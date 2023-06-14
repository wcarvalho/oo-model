
from absl import app
from absl import flags

import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds

from experiments import babyai_env_utils
from tqdm import tqdm
import tensorflow as tf
import jax

from acme import wrappers as acme_wrappers
from acme import types
from acme.wrappers import base
from acme.utils import paths
from acme.jax import utils as jax_utils
import dm_env
from envs.babyai_kitchen.bot import KitchenBot

from experiments import experiment_builders  # for FLAGS
# from envs.babyai_kitchen.multilevel import MultiLevel
# from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
# from envs.babyai_kitchen import babyai_utils


FLAGS = flags.FLAGS

# Agent flags
# flags.DEFINE_string('data_file', '', 'data_file')
# flags.DEFINE_bool('debug', False, 'whether to debug script')
flags.DEFINE_string('tasks_file', 'pickup_sanity', 'tasks_file')
flags.DEFINE_string('size', 'large', 'small=1e4, medium=1e5, large=1e6, xl=1e7')
flags.DEFINE_integer('room_size', 5, 'room size')
flags.DEFINE_bool('partial_obs', False, 'partial observability')


def collect_episode(gym_env, dm_env, idx=None):
  """Collect 1 episode by using bot."""
  steps = 0
  dm_env.reset()
  bot = KitchenBot(gym_env)
  actions_taken = []
  action_taken = None
  subgoals = iter(bot.subgoals)
  current_subgoal = next(subgoals)
  
  while bot.stack:
    # plan action
    try:
      action = bot.replan(action_taken)
    except Exception as e:
      print('-'*30, idx)
      print(f"Episode failed after {steps} steps.")
      print(e)
      return
    if action == gym_env.actions.done:
      break
    # observe consequences
    dm_env.step(action)
    actions_taken.append(gym_env.idx2action[int(action)])
    action_taken = action
    steps += 1

    object_infront = gym_env.grid.get(*gym_env.front_pos)
    if object_infront and object_infront.type == current_subgoal.goto.type:

      for action_str in current_subgoal.actions:
        interaction = gym_env.actiondict[action_str]
        timestep = dm_env.step(interaction)
        actions_taken.append(action_str)

      # update subgoal if any are left
      try:
        current_subgoal = next(subgoals)
      except:
        pass
    
    if steps > 100:
      return # failed
  
  if not timestep.reward > 0.0:
    print('-'*30, idx)
    print(f"Episode terminated with no reward after {steps} steps.")

def get_episodes(size, debug: bool = False):
  if debug: return 100
  sizes = dict(
    small=int(1e4),
    medium=int(1e5),
    large=int(1e6),
    xl=int(1e7),
  )
  return sizes[size]

def get_name(nepisodes):
  return {
    int(1e2) : "1e2",
    int(1e3) : "1e3",
    int(1e4) : "1e4",
    int(1e5) : "1e5",
    int(1e6) : "1e6",
    int(1e7) : "1e7",
  }[nepisodes]

def directory_name(tasks_file,
                   room_size,
                   partial_obs,
                   nepisodes: int = None,
                   evaluation: bool = False,
                   **kwargs):

  prefix=f"{tasks_file},r={room_size},p={partial_obs}"
  suffix = 'test' if evaluation else 'train'
  data_directory=f'data/babyai_kitchen/{prefix}/{suffix}_{get_name(nepisodes)}'

  return data_directory

def named_tuple_to_dict(nt):
    """Recursively convert to namedtuple to dictionary"""
    if isinstance(nt, tuple):
        if hasattr(nt, '_fields'):
            return {k: named_tuple_to_dict(v) for k, v in nt._asdict().items()}
        else:
            return tuple(named_tuple_to_dict(x) for x in nt)
    else:
        return nt


class EnvLoggerWrapper(base.EnvironmentWrapper):

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    new_timestep = self._augment_timestep(timestep)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_timestep = self._augment_timestep(timestep)
    return new_timestep

  def _augment_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    timestep = timestep._replace(observation=named_tuple_to_dict(timestep.observation))
    # timestep = timestep._replace(reward=float(timestep.reward))
    return timestep

  def observation_spec(self):
      return named_tuple_to_dict(self._environment.observation_spec())

def make_dataset(env_kwargs: dict, nepisodes: int, debug: bool = False):
  environment = babyai_env_utils.make_kitchen_environment(
      **env_kwargs,
      wrapper_list=[
          EnvLoggerWrapper],
      evaluation=False)


  tf_tensor = lambda x: tfds.features.Tensor(shape=x.shape, dtype=x.dtype)
  obs_spec = environment.observation_spec()
  tf_obs_spec = jax.tree_map(lambda v: tf_tensor(v), obs_spec)
  dataset_config = tfds.rlds.rlds_base.DatasetConfig(
      name=env_kwargs['tasks_file'],
      observation_info=tf_obs_spec,
      action_info=tf.int64,
      reward_info=tf.float64,
      discount_info=tf.float64,
      )

  def step_fn(timestep, action, env):
    return {}

  for evaluation in [False, True]:
    dm_env, gym_env = babyai_env_utils.make_kitchen_environment(
        **env_kwargs,
        wrapper_list=[EnvLoggerWrapper],
        evaluation=evaluation,
        return_gym_env=True)

    data_directory = directory_name(
      **env_kwargs, evaluation=evaluation, nepisodes=nepisodes, debug=debug)
    paths.process_path(data_directory)  # create directory

    with envlogger.EnvLogger(
      dm_env,
      step_fn=step_fn,
      backend=tfds_backend_writer.TFDSBackendWriter(
        data_directory=data_directory,
        split_name='test' if evaluation else 'train',
        max_episodes_per_file=nepisodes,
        metadata=dict(env_kwargs=env_kwargs),
        ds_config=dataset_config)) as dm_env:

      for episode_idx in tqdm(range(nepisodes)):
        collect_episode(gym_env, dm_env, idx=episode_idx)

def main(unused_argv):

  env_kwargs = dict(
      tasks_file=FLAGS.tasks_file,
      room_size=FLAGS.room_size,
      partial_obs=FLAGS.partial_obs,
  )
  nepisodes = get_episodes(FLAGS.size, FLAGS.debug)
  make_dataset(env_kwargs=env_kwargs,
               nepisodes=nepisodes)


if __name__ == '__main__':
  app.run(main)
