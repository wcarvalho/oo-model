
import functools 

from absl import flags
from absl import app
from ray import tune
from absl import logging
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp
import os.path

from acme import specs
from acme.jax import experiments
from acme.utils import paths
import dm_env
import rlds

from envs.multitask_kitchen import Observation
from experiments.observers import LevelAvgReturnObserver, ExitObserver
from experiments import config_utils
from experiments import train_many
from experiments import experiment_builders
from experiments import babyai_env_utils
from experiments import babyai_collect_data
from experiments import dataset_utils
from experiments import logger as wandb_logger
from experiments import offline_configs

flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_bool(
    'train_single', False, 'Run many or 1 experiments')
flags.DEFINE_bool(
    'make_path', False, 'Create a path under `FLAGS.folder` for the experiment')
flags.DEFINE_bool(
    'make_dataset', False, 'Make dataset if does not exist.')
flags.DEFINE_bool(
    'auto_name_wandb', False, 'automatically name wandb.')

FLAGS = flags.FLAGS


def setup_experiment_inputs(
    agent : str,
    path: str = '.',
    agent_config_kwargs: dict=None,
    agent_config_file: str=None,
    env_kwargs: dict=None,
    env_config_file: str=None,
    debug: bool = False,
    make_dataset: bool = False,
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

  config, builder, network_factory = experiment_builders.default_setup_agents(
    agent=agent,
    debug=debug,
    config_kwargs=config_kwargs,
    env_kwargs=env_kwargs,
    update_logger_kwargs=dict(
        action_names=['left', 'right', 'forward', 'pickup_1',
                      'pickup_2', 'place', 'toggle', 'slice'],
    )
  )
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
        evaluate_train_test=True,
        **env_kwargs)

  # Create an environment and make the environment spec
  environment = environment_factory(0)
  environment_spec = specs.make_environment_spec(environment)

  # Define the demonstrations factory.
  data_directory = babyai_collect_data.directory_name(
      tasks_file=env_kwargs['tasks_file'],
      room_size=env_kwargs['room_size'],
      num_dists=env_kwargs['num_dists'],
      partial_obs=env_kwargs['partial_obs'],
      evaluation=False,
      debug=debug)
  
  dataset_info_path = os.path.join(data_directory, 'dataset_info.json')
  if not os.path.exists(dataset_info_path):
    if make_dataset:
      logging.info(f'MAKING DATASET: {data_directory}')
      import ipdb; ipdb.set_trace()
      babyai_collect_data.make_dataset(
        env_kwargs=env_kwargs,
        nepisodes=100 if debug else int(1e5),
        debug=debug,
      )
    else:
      raise RuntimeError(f"Does not exist: {dataset_info_path}")


  demonstration_dataset_factory = dataset_utils.make_demonstration_dataset_factory(
      data_directory=data_directory,
      obs_constructor=Observation,
      batch_size=config.batch_size,
      trace_length=config.trace_length)

  # -----------------------
  # setup observer factory for environment
  # -----------------------
  observers = [
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
    observers=observers,
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
    make_dataset=FLAGS.make_dataset,
    debug=debug)

  def custom_steps_keys(name: str):
    return f'{name}_steps'
  log_dir = FLAGS.folder

  if FLAGS.make_path:
    log_dir = wandb_logger.gen_log_dir(
      base_dir=log_dir,
      hourminute=True,
      date=True,
      )
    if FLAGS.auto_name_wandb and wandb_init_kwargs is not None:
      date_time = wandb_logger.datetime(time=True)
      logging.info(f'wandb name: {str(date_time)}')
      wandb_init_kwargs['name'] = date_time

  experiment = experiment_builders.build_offline_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    agent=FLAGS.agent,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    debug=debug,
    logger_factory_kwargs=dict(
      custom_steps_keys=custom_steps_keys,
    ),
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
    # NOTE: DEBUGGING ONLY. otherwise change settings below
    experiments.run_offline_experiment(
      experiment=experiment,
      eval_every=5,
      num_eval_episodes=5,
      )



def sweep(search: str = 'default', agent: str = 'muzero'):
  if search == 'default':
    space = [
        {
            "seed": tune.grid_search([3]),
            "agent": tune.grid_search([agent]),
            "tasks_file": tune.grid_search([
                'place_split_easy', 'place_split_hard']),
        }
    ]
  elif search == 'benchmark':
    space = [
        {
            "seed": tune.grid_search([4]),
            "group": tune.grid_search(['benchmark4']),
            "agent": tune.grid_search(['muzero', 'factored']),
            "tasks_file": tune.grid_search([
                'place_split_easy']),
        }
    ]
  elif search == 'muzero':
    space = [
        {
            "seed": tune.grid_search([2]),
            "group": tune.grid_search(['muzero8']),
            "agent": tune.grid_search(['muzero']),
            "num_learner_steps": tune.grid_search([int(1e5)]),
            "v_target_source": tune.grid_search(['reanalyze']),
            "tasks_file": tune.grid_search(['place_split_easy']),
            # "reanalyze_ratio": tune.grid_search([0.0, 0.5, 1.0]),
            # "num_bins": tune.grid_search([101, 301]),
            # "max_scalar_value": tune.grid_search([5.0, 10.0]),
        }
    ]
  elif search == 'factored':
    space = [
        {
            "seed": tune.grid_search([1]),
            # "group": tune.grid_search(['benchmark2']),
            "agent": tune.grid_search(['factored']),
            "tasks_file": tune.grid_search(['pickup']),
        }
    ]
  elif search == 'attn_1':
    space = [
        {
            "seed": tune.grid_search([1]),
            # "group": tune.grid_search(['benchmark2']),
            "agent": tune.grid_search(['factored']),
            "show_gradients": tune.grid_search(['factored']),
            "tasks_file": tune.grid_search([
                'pickup'
            ]),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space


def main(_):

  # -----------------------
  # wandb setup
  # -----------------------
  search = FLAGS.search or 'default'
  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      dir=FLAGS.wandb_dir,
      save_code=False,
  )
  if FLAGS.train_single:
    # overall group
    wandb_init_kwargs['group'] = FLAGS.wandb_group or f"{search}_{FLAGS.agent}"
  else:
    if FLAGS.wandb_group:
      logging.info(
          f'IGNORING `wandb_group`. This will be set using the current `search`')
    wandb_init_kwargs['group'] = search

  if FLAGS.wandb_name:
    wandb_init_kwargs['name'] = FLAGS.wandb_name

  use_wandb = FLAGS.use_wandb
  if not use_wandb:
    wandb_init_kwargs = None

  # -----------------------
  # env setup
  # -----------------------
  default_env_kwargs = dict(
      tasks_file=FLAGS.tasks_file,
      room_size=FLAGS.room_size,
      num_dists=FLAGS.num_dists,
      partial_obs=FLAGS.partial_obs,
  )
  if FLAGS.train_single:
    train_single(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs)
  else:
    run_distributed = FLAGS.run_distributed
    num_actors = FLAGS.num_actors
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