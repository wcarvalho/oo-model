
from absl import app
from absl import flags

import envlogger

from experiments import helpers
from tqdm import tqdm

from acme.utils import paths
from envs.babyai_kitchen.bot import KitchenBot
# from envs.babyai_kitchen.multilevel import MultiLevel
# from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
# from envs.babyai_kitchen import babyai_utils


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
# Agent flags
flags.DEFINE_string('tasks_file', 'place_split_hard', 'tasks_file')
# flags.DEFINE_string('data_file', '', 'data_file')
flags.DEFINE_bool('debug', False, 'whether to debug script')
flags.DEFINE_integer('episodes', int(1e5), 'number of episodes')


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


def make_directory(tasks_file, evaluation: bool = False, debug: bool = False):
  suffix = 'test' if evaluation else 'train'
  data_directory=f'data/babyai_kitchen_{tasks_file}/{suffix}'
  if debug:
    data_directory+="_debug"
  return data_directory

def main(unused_argv):

  for evaluation in [False, True]:
    dm_env, gym_env = helpers.make_kitchen_environment(
      tasks_file=FLAGS.tasks_file,
      evaluation=evaluation,
      return_gym_env=True)

    data_directory = make_directory(FLAGS.tasks_file, evaluation, FLAGS.debug)
    paths.process_path(data_directory)  # create directory

    with envlogger.EnvLogger(dm_env, data_directory=data_directory) as dm_env:
      nepisodes = 100 if FLAGS.debug else FLAGS.episodes
      for episode_idx in tqdm(range(nepisodes)):
        collect_episode(gym_env, dm_env, idx=episode_idx)


if __name__ == '__main__':
  app.run(main)
