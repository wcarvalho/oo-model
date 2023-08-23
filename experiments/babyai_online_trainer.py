
import functools 

from absl import flags
from absl import app
from ray import tune
from absl import logging
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme.jax import experiments
from acme.utils import paths
import dm_env

from experiments import config_utils
from experiments import train_many
from experiments import experiment_builders
from experiments import babyai_env_utils
from experiments.observers import LevelAvgReturnObserver
from experiments import logger as wandb_logger
from r2d2 import R2D2Config

flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_bool(
    'train_single', False, 'Run many or 1 experiments')
flags.DEFINE_bool(
    'make_path', False, 'Create a path under `FLAGS>folder` for the experiment')
flags.DEFINE_bool(
    'auto_name_wandb', False, 'automatically name wandb.')
FLAGS = flags.FLAGS


def setup_agents(
    agent: str,
    config_kwargs: dict = None,
    env_kwargs: dict = None,
    debug: bool = False,
    update_logger_kwargs: dict = None,
    setup_kwargs: dict = None,
    config_class: R2D2Config = None,
):
  config_kwargs = config_kwargs or dict()
  update_logger_kwargs = update_logger_kwargs or dict()
  setup_kwargs = setup_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  if agent == 'r2d2':
    from experiments import babyai_rd2d2
    config, builder, network_factory = babyai_rd2d2.setup(
        debug=debug,
        config_kwargs=config_kwargs,
        config_class=config_class,
        **setup_kwargs)
  elif agent == 'muzero':
    from experiments import babyai_muzero
    from muzero import learner_logger

    config = babyai_muzero.load_config(
      config_class=config_class,
      config_kwargs=config_kwargs)

    builder_kwargs = dict(
        visualization_logger=learner_logger.LearnerLogger(
            label='MuZeroLearnerLogger',
            log_frequency=5 if debug else 4000,
            discount=config.discount,
            **update_logger_kwargs,
        ),
    )
    builder, network_factory = babyai_muzero.setup(
        config=config,
        config_class=config_class,
        builder_kwargs=builder_kwargs,
        config_kwargs=config_kwargs,
        **setup_kwargs)

  elif agent in ('factored', 'branched'):
    from experiments import babyai_factored_muzero
    from factored_muzero.analysis_actor import VisualizeActor, AttnLogger
    from factored_muzero import learner_logger

    config = babyai_factored_muzero.load_config(
        config_class=config_class,
        config_kwargs=config_kwargs)

    builder_kwargs = dict(
        # actorCls=functools.partial(
        #     VisualizeActor,
        #     logger=AttnLogger(),
        #     log_frequency=50 if debug else 4000),
        visualization_logger=learner_logger.LearnerLogger(
            label='FactoredMuZeroLearnerLogger',
            log_frequency=5 if debug else 4000,
            discount=config.discount,
            **update_logger_kwargs,
        ),

    )
    assert env_kwargs is not None
    room_size = env_kwargs['room_size']
    builder, network_factory = babyai_factored_muzero.setup(
        config=config,
        agent_name=agent,
        network_kwargs=dict(num_spatial_vectors=room_size**2),
        builder_kwargs=builder_kwargs,
        **setup_kwargs)

  else:
    raise NotImplementedError

  return config, builder, network_factory


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
  logging.info(f'config_kwargs: {str(config_kwargs)}')

  env_kwargs = env_kwargs or dict()
  if env_config_file:
    env_kwargs = config_utils.load_config(env_config_file)
  logging.info(f'env_kwargs: {str(env_kwargs)}')

  # -----------------------
  # setup environment factory
  # -----------------------
  def environment_factory(seed: int,
                          evaluation: bool = False) -> dm_env.Environment:
    del seed
    return babyai_env_utils.make_kitchen_environment(
        path=path,
        debug=debug,
        evaluation=evaluation,
        **env_kwargs)

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  # Configure the agent & update with config kwargs

  config, builder, network_factory = setup_agents(
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
  # setup observer factory for environment
  # -----------------------
  observers = [
      LevelAvgReturnObserver(
              get_task_name=lambda env: str(env.env.current_levelname),
              reset=50 if not debug else 5),
          # ExitObserver(window_length=500, exit_at_success=.99),  # will exit at certain success rate
      ]

  return experiment_builders.OnlineExperimentConfigInputs(
    agent_config=config,
    final_env_kwargs=env_kwargs,
    builder=builder,
    network_factory=network_factory,
    environment_factory=environment_factory,
    observers=observers,
  )

def train_single(
    default_env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    **kwargs,
):

  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    agent=FLAGS.agent,
    path=FLAGS.path,
    agent_config_kwargs=agent_config_kwargs,
    agent_config_file=FLAGS.agent_config,
    env_kwargs=default_env_kwargs,
    env_config_file=FLAGS.env_config,
    debug=debug)

  log_dir = FLAGS.folder
  if FLAGS.make_path:
    log_dir = wandb_logger.gen_log_dir(
        base_dir=log_dir,
        hourminute=True,
        date=True,
    )
    if FLAGS.auto_name_wandb and wandb_init_kwargs is not None:
      date_time = wandb_logger.date_time(time=True)
      logging.info(f'wandb name: {str(date_time)}')
      wandb_init_kwargs['name'] = date_time

  tasks_file = experiment_config_inputs.final_env_kwargs['tasks_file']
  logger_factory_kwargs = dict(
    actor_label=f"actor_{tasks_file}",
    evaluator_label=f"evaluator_{tasks_file}",
    learner_label=f"learner_{FLAGS.agent}",
  )

  experiment = experiment_builders.build_online_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    agent=FLAGS.agent,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    logger_factory_kwargs=logger_factory_kwargs,
    debug=debug,
    **kwargs,
  )
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=experiment,
        num_actors=FLAGS.num_actors)

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
    experiments.run_experiment(experiment=experiment)



def sweep(search: str = 'default', agent: str = 'muzero'):
  settings=dict(
    place5=dict(tasks_file='place_split_medium', room_size=5, num_steps=3e6),
    place6=dict(tasks_file='place_split_medium', room_size=6, num_steps=5e6),
    place7=dict(tasks_file='place_split_medium', room_size=7, num_steps=7e6),
    medium5=dict(tasks_file='medium_medium', room_size=5, num_steps=4e6),
    medium6=dict(tasks_file='medium_medium', room_size=6, num_steps=6e6),
    medium7=dict(tasks_file='medium_medium', room_size=7, num_steps=8e6),
    long5=dict(tasks_file='long_medium', room_size=5, num_steps=5e6),
    long6=dict(tasks_file='long_medium', room_size=6, num_steps=7e6),
    long7=dict(tasks_file='long_medium', room_size=7, num_steps=9e6),
  )
  if search == 'default':
    space = [
        {
            "seed": tune.grid_search([1]),
            "agent": tune.grid_search([agent]),
        }
    ]
  elif search == 'benchmark':
    shared = {
      "seed": tune.grid_search([1]),
      "group": tune.grid_search(['benchmark6']),
      "partial_obs": tune.grid_search([True]),
    }
    space = [
        {**shared, **settings['place5'],
          # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
          "agent": tune.grid_search(['muzero']),
        },
        {**shared, **settings['place7'],
          # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
          "agent": tune.grid_search(['muzero']),
        },
        {**shared, **settings['medium5'],
          # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
          "agent": tune.grid_search(['muzero']),
        },
        {**shared, **settings['medium7'],
          # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
          "agent": tune.grid_search(['muzero']),
        },
    ]
  elif search == 'muzero':
    space = [
        {
            "seed": tune.grid_search([1]),
            "group": tune.grid_search(['benchmark6']),
            "agent": tune.grid_search(['muzero']),
            "room_size": tune.grid_search([5,7]),
            "tasks_file": tune.grid_search([
                'place_split_medium',
                'medium_medium',
                'long_medium',
                ]),
            # 'root_value_coef': tune.grid_search([.25]),
            # 'model_policy_coef': tune.grid_search([10.0]),
            # 'model_value_coef': tune.grid_search([10.0, 2.5]),
            # 'mask_model': tune.grid_search([True]),
            # 'clip_probs': tune.grid_search([True, False]),
        }
    ]
  elif search == 'factored1':
    shared = {
        "seed": tune.grid_search([4]),
        "group": tune.grid_search(['factored2']),
        "agent": tune.grid_search(['branched']),
        "partial_obs": tune.grid_search([True]),
         **settings['place5'],
    }
    space = [
        {**shared,
         # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
         },
        {
            **shared,
            "recon_coeff": tune.grid_search([1.0, 1e-1, 1e-2, 1e-3]),
        },
        # {
        #     **shared,
        #     "savi_mlp_size": tune.grid_search([64]),
        # },

    ]
  elif search == 'factored2':
    shared = {
        "seed": tune.grid_search([4]),
        "group": tune.grid_search(['factored2']),
        "agent": tune.grid_search(['branched']),
        "partial_obs": tune.grid_search([True]),
         **settings['place5'],
    }
    space = [
        {**shared,
         # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
         },
        {
            **shared,
            "max_grad_norm": tune.grid_search([80.0, 5., .5, .05]),
        },
        # {
        #     **shared,
        #     "savi_mlp_size": tune.grid_search([64]),
        # },

    ]
  else:
    raise NotImplementedError(search)

  return space


def main(_):
  agent_config_kwargs = dict()
  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      show_gradients=1,
      samples_per_insert=1,
      min_replay_size=100,
    ))
  # -----------------------
  # wandb setup
  # -----------------------
  search = FLAGS.search or 'default'
  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      save_code=False,
  )
  if FLAGS.train_single:
    # overall group
    wandb_init_kwargs['group'] = FLAGS.wandb_group if FLAGS.wandb_group else f"{search}_{FLAGS.agent}"
  else:
    if FLAGS.wandb_group:
      logging.info(f'IGNORING `wandb_group`. This will be set using the current `search`')
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
      num_dists=1,
      partial_obs=False,
  )
  run_distributed = FLAGS.run_distributed
  num_actors = FLAGS.num_actors
  if FLAGS.train_single:
    train_single(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      agent_config_kwargs=agent_config_kwargs)
  else:
    train_many.run(
      name='babyai_online_trainer',
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      use_wandb=use_wandb,
      debug=FLAGS.debug,
      space=sweep(search, FLAGS.agent),
      make_program_command=functools.partial(
        train_many.make_program_command,
        filename='experiments/babyai_online_trainer.py',
        run_distributed=run_distributed,
        num_actors=num_actors,
        ),
    )

if __name__ == '__main__':
  app.run(main)