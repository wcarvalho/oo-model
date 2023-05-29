
import functools 

from absl import flags
from absl import app
from ray import tune
from absl import logging
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme import specs
from acme.jax import experiments
from acme.utils import paths
import dm_env

from envs.multitask_kitchen import Observation
from experiments.observers import LevelAvgReturnObserver, ExitObserver
from experiments import config_utils
from experiments import train_many
from experiments import experiment_builders
from experiments import babyai_env_utils
from experiments import collect_data
from experiments import dataset_utils


flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_bool(
    'train_single', False, 'Run many or 1 experiments')
FLAGS = flags.FLAGS


def setup_experiment_inputs(
    agent : str,
    path: str = '.',
    agent_config_kwargs: dict=None,
    agent_config_file: str=None,
    env_kwargs: dict=None,
    env_config_file: str=None,
    debug: bool = False,
  ):
  """Setup."""

  # -----------------------
  # load agent and environment kwargs (potentially from files)
  # -----------------------
  config_kwargs = agent_config_kwargs or dict()
  if agent_config_file:
    config_kwargs = config_utils.load_config(agent_config_file)
  config_kwargs.update(  # custom for offline
    dict(importance_sampling_exponent=0.0,
         use_stored_lstm_state=False))
  logging.info(f'config_kwargs: {str(config_kwargs)}')

  env_kwargs = env_kwargs or dict()
  if env_config_file:
    env_kwargs = config_utils.load_config(env_config_file)
  logging.info(f'env_kwargs: {str(env_kwargs)}')

  # -----------------------
  # setup environment
  # -----------------------
  def environment_factory(seed: int,
                          evaluation: bool = False) -> dm_env.Environment:
    del seed
    return babyai_env_utils.make_kitchen_environment(
        path=path,
        debug=debug,
        evaluation=evaluation,
        **env_kwargs)

  # Create an environment and make the environment spec
  environment = environment_factory(0)
  environment_spec = specs.make_environment_spec(environment)

  # Define the demonstrations factory.
  data_directory = collect_data.directory_name(
      tasks_file=env_kwargs['tasks_file'],
      evaluation=False,
      debug=debug)

  demonstration_dataset_factory = dataset_utils.make_demonstration_dataset_factory(
      data_directory,
      obs_constructor=Observation,
      batch_size=config.batch_size,
      trace_length=config.trace_length)

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  if agent == 'r2d2':
    from experiments import babyai_rd2d2
    config, builder, network_factory = babyai_rd2d2.setup(
      debug=debug,
      config_kwargs=config_kwargs)
  elif agent == 'muzero':
    from experiments import babyai_muzero
    config, builder, network_factory = babyai_muzero.setup(
      debug=debug,
      config_kwargs=config_kwargs)
  elif agent == 'factored':
    from experiments import babyai_factored_muzero
    room_size = env_kwargs['room_size']
    config, builder, network_factory = babyai_factored_muzero.setup(
      debug=debug,
      config_kwargs=config_kwargs,
      network_kwargs=dict(num_spatial_vectors=room_size**2))
  else:
    raise NotImplementedError

  # -----------------------
  # setup observer factory for environment
  # -----------------------
  def observer_factory():
    return [
      LevelAvgReturnObserver(
              get_task_name=lambda env: str(env.env.current_levelname),
              reset=50 if not debug else 5),
          # ExitObserver(window_length=500, exit_at_success=.99),  # will exit at certain success rate
      ]

  return experiment_builders.OfflineExperimentConfigInputs(
    agent_config=config,
    final_env_kwargs=env_kwargs,
    builder=builder,
    network_factory=network_factory,
    environment_factory=environment_factory,
    observer_factory=observer_factory,
    demonstration_dataset_factory=demonstration_dataset_factory,
    environment_spec=environment_spec,
  )

def train_single(
    default_env_kwargs: dict,
    wandb_init_kwargs: dict = None,
    **kwargs
):
  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    agent=FLAGS.agent,
    path=FLAGS.path,
    agent_config_file=FLAGS.agent_config,
    env_kwargs=default_env_kwargs,
    env_config_file=FLAGS.env_config,
    debug=debug)

  experiment = experiment_builders.build_offline_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    agent=FLAGS.agent,
    debug=debug,
    log_dir=FLAGS.folder,
    wandb_init_kwargs=wandb_init_kwargs,
    debug=debug,
    **kwargs
  )
  if FLAGS.run_distributed:
    program = experiments.make_distributed_offline_experiment(
        experiment=experiment)

    local_resources = {
        "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
    }
    lp.launch(program,
              lp.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal='current_terminal',
              local_resources=local_resources)
  else:
    experiments.run_offline_experiment(
      experiment=experiment,
      )



def sweep(search: str = 'default', agent: str = 'muzero'):
  if search == 'default':
    space = [
        {
            "seed": tune.grid_search([1]),
            "agent": tune.grid_search([agent]),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space


def main(_):

  # -----------------------
  # wandb setup
  # -----------------------
  use_wandb = FLAGS.use_wandb
  search = FLAGS.search or 'default'
  if FLAGS.train_single:
    group = FLAGS.wandb_group if FLAGS.wandb_group else FLAGS.agent  # overall group
  else:
    group = FLAGS.wandb_group if FLAGS.wandb_group else search  # overall group
  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      group=group,  # overall group
      notes=FLAGS.wandb_notes,
      save_code=True,
  )
  if FLAGS.wandb_name:
    wandb_init_kwargs['name'] = FLAGS.wandb_name

  # -----------------------
  # env setup
  # -----------------------
  default_env_kwargs = dict(
      tasks_file='place_split_hard',
      room_size=5,
      num_dists=1,
      partial_obs=False,
  )
  run_distributed = FLAGS.run_distributed
  num_actors = FLAGS.num_actors
  if FLAGS.train_single:
    train_single(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs)
  else:
    train_many.run(
      name='babyai_offline_trainer',
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      use_wandb=use_wandb,
      debug=FLAGS.debug,
      space=sweep(search, FLAGS.agent),
      make_program_command=functools.partial(
        train_many.make_program_command,
        filename='experiments/babyai_offline_trainer.py',
        run_distributed=run_distributed,
        num_actors=num_actors,
        ),
    )

if __name__ == '__main__':
  app.run(main)