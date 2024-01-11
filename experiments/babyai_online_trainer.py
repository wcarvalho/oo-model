"""
Running experiments:
--------------------

# DEBUGGING, single stream
python -m ipdb -c continue experiments/babyai_online_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --search='baselines'

# DEBUGGING, single stream, disable just-in-time compilation
JAX_DISABLE_JIT=1 python -m ipdb -c continue experiments/babyai_online_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --search='baselines'

# DEBUGGING, launching jobs in parallel with ray: see `sweep` fn
python -m ipdb -c continue experiments/babyai_online_trainer.py \
  --parallel='ray' \
  --debug_parallel=True \
  --run_distributed=False \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=factored_muzero2_debug \
  --search='baselines'


# launching jobs in parallel with ray: see `sweep` fn
python -m ipdb -c continue experiments/babyai_online_trainer.py \
  --parallel='ray' \
  --run_distributed=True \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=factored_muzero2 \
  --search='muzero'

"""
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
from experiments import parallel
from experiments import experiment_builders
from experiments import babyai_env_utils
from experiments.observers import LevelAvgReturnObserver
from experiments import logger as wandb_logger
from r2d2 import R2D2Config

flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')
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
    strict_config: bool = True,
    config_class: R2D2Config = None,
):
  config_kwargs = config_kwargs or dict()
  update_logger_kwargs = update_logger_kwargs or dict()
  setup_kwargs = setup_kwargs or dict()
  del env_kwargs
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
  elif agent in ('conv_muzero', 'muzero'):
    from experiments import babyai_muzero
    from muzero import learner_logger

    config = babyai_muzero.load_config(
      config_class=config_class,
      config_kwargs=config_kwargs,
      strict_config=strict_config)

    builder_kwargs = dict(
        visualization_logger=learner_logger.LearnerLogger(
            label='MuZeroLearnerLogger',
            log_frequency=5 if debug else 4000,
            discount=config.discount,
            **update_logger_kwargs,
        ),
    )
    builder, network_factory = babyai_muzero.setup(
        agent=agent,
        config=config,
        config_class=config_class,
        builder_kwargs=builder_kwargs,
        config_kwargs=config_kwargs,
        **setup_kwargs)

  elif agent in ('factored', 'branched', 'conv_factored'):
    from experiments import babyai_factored_muzero
    from factored_muzero.analysis_actor import VisualizeActor, AttnLogger
    from factored_muzero import learner_logger

    config = babyai_factored_muzero.load_config(
        config_class=config_class,
        config_kwargs=config_kwargs,
        strict_config=strict_config)

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
    builder, network_factory = babyai_factored_muzero.setup(
        config=config,
        agent_name=agent,
        builder_kwargs=builder_kwargs,
        **setup_kwargs)

  else:
    raise NotImplementedError

  return config, builder, network_factory

def setup_experiment_inputs(
    agent_config_kwargs: dict=None,
    env_kwargs: dict=None,
    debug: bool = False,
    strict_config: bool = True,
    path: str = '.',
  ):
  """Setup."""
  config_kwargs = agent_config_kwargs or dict()
  env_kwargs = env_kwargs or dict()

  agent = config_kwargs.get("agent", 'agent')
  assert agent != 'agent', "no agent selected"

  # -----------------------
  # setup environment factory
  # -----------------------
  def environment_factory(seed: int,
                          evaluation: bool = False) -> dm_env.Environment:
    del seed
    return babyai_env_utils.make_kitchen_environment(
        path=path,
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
      strict_config=strict_config,
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
    env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    log_dir: str = None,
    num_actors: int = 1,
    run_distributed: bool = False,
):
  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    agent_config_kwargs=agent_config_kwargs,
    env_kwargs=env_kwargs,
    debug=debug)

  tasks_file = experiment_config_inputs.final_env_kwargs['tasks_file']
  logger_factory_kwargs = dict(
    actor_label=f"actor_{tasks_file}",
    evaluator_label=f"evaluator_{tasks_file}",
  )

  experiment = experiment_builders.build_online_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    logger_factory_kwargs=logger_factory_kwargs,
    debug=debug,
    run_distributed=run_distributed,
  )

  if run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=experiment,
        num_actors=num_actors)

    local_resources = {
        "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
    }
    controller = lp.launch(program,
              lp.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal='current_terminal',
              local_resources=local_resources)
    controller.wait(return_on_first_completed=True)
    controller._kill()

  else:
    experiments.run_experiment(
      experiment=experiment,
      num_eval_episodes=10)


def setup_wandb_init_kwargs():
  if not FLAGS.use_wandb:
    return dict()

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      name=FLAGS.wandb_name,
      group=FLAGS.search,
      save_code=False,
  )
  return wandb_init_kwargs

def run_single():
  ########################
  # default settings
  ########################
  env_kwargs = dict()
  agent_config_kwargs = dict()
  num_actors = FLAGS.num_actors
  run_distributed = FLAGS.run_distributed
  wandb_init_kwargs = setup_wandb_init_kwargs()
  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      samples_per_insert=1.0,
      min_replay_size=100,
    ))
    env_kwargs.update(dict(
    ))

  import os
  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results'

  if FLAGS.make_path:
    # i.e. ${folder}/runs/${date_time}/
    folder = parallel.gen_log_dir(
        base_dir=os.path.join(folder, 'rl_results'),
        hourminute=True,
        date=True,
    )

  ########################
  # override with config settings, e.g. from parallel run
  ########################
  if FLAGS.config_file:
    configs = config_utils.load_config(FLAGS.config_file)
    if isinstance(configs, list):
      config = configs[FLAGS.config_idx-1]  # starts at 1 with SLURM
    elif isinstance(configs, dict):
      config = configs
    else:
      raise NotImplementedError(type(configs))
    logging.info(f'loaded config: {str(config)}')

    agent_config_kwargs.update(config['agent_config'])
    env_kwargs.update(config['env_config'])
    folder = config['folder']

    num_actors = config['num_actors']
    run_distributed = config['run_distributed']

    wandb_init_kwargs['group'] = config['wandb_group']
    wandb_init_kwargs['name'] = config['wandb_name']
    wandb_init_kwargs['project'] = config['wandb_project']
    wandb_init_kwargs['entity'] = config['wandb_entity']

    if not config['use_wandb']:
      wandb_init_kwargs = dict()

  if FLAGS.debug and not FLAGS.subprocess:
      configs = parallel.get_all_configurations(spaces=sweep(FLAGS.search))
      first_agent_config, first_env_config = parallel.get_agent_env_configs(
          config=configs[0])
      agent_config_kwargs.update(first_agent_config)
      env_kwargs.update(first_env_config)

  train_single(
    wandb_init_kwargs=wandb_init_kwargs,
    env_kwargs=env_kwargs,
    agent_config_kwargs=agent_config_kwargs,
    log_dir=folder,
    num_actors=num_actors,
    run_distributed=run_distributed
    )


def run_many():
  wandb_init_kwargs = setup_wandb_init_kwargs()

  import os
  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results_dir'

  assert FLAGS.debug is False, 'only run debug if not running many things in parallel'

  if FLAGS.parallel == 'ray':
    parallel.run_ray(
      trainer_filename=__file__,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      folder=folder,
      run_distributed=FLAGS.run_distributed,
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep(FLAGS.search),
      num_actors=FLAGS.num_actors
    )
  elif FLAGS.parallel == 'sbatch':
    parallel.run_sbatch(
      trainer_filename=__file__,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      folder=folder,
      run_distributed=FLAGS.run_distributed,
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep(FLAGS.search),
      num_actors=FLAGS.num_actors)


def sweep(search: str = 'default', **kwargs):
  settings=dict(
    pickup5={
        'env.tasks_file':'pickup',
        'env.room_size':5,
        'num_steps':3e6},
    place5={
        'env.tasks_file':'place_split_medium',
        'env.room_size':5,
        'num_steps':3e6},
    place6={
        'env.tasks_file':'place_split_medium',
        'env.room_size':6,
        'num_steps':5e6},
    place7={
        'env.tasks_file':'place_split_medium',
        'env.room_size':7,
        'num_steps':7e6},
    medium5={
        'env.tasks_file':'medium_medium',
        'env.room_size':5,
        'num_steps':4e6},
    medium6={
        'env.tasks_file':'medium_medium',
        'env.room_size':6,
        'num_steps':6e6},
    medium7={
        'env.tasks_file':'medium_medium',
        'env.room_size':7,
        'num_steps':8e6},
    long5={
        'env.tasks_file':'long_medium',
        'env.room_size':5,
        'num_steps':5e6},
    long6={
        'env.tasks_file':'long_medium',
        'env.room_size':6,
        'num_steps':7e6},
    long7={
        'env.tasks_file':'long_medium',
        'env.room_size':7,
        'num_steps':9e6},
  )
  if search == 'benchmark':
    shared = {
      "seed": tune.grid_search([1]),
      "group": tune.grid_search(['benchmark6']),
      "env.partial_obs": tune.grid_search([True]),
    }
    space = [
        {**shared, **settings['place5'],
          # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
          "agent": tune.grid_search(['conv_muzero']),
        },
        # {**shared, **settings['place7'],
        #   # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
        #   "agent": tune.grid_search(['muzero']),
        # },
        # {**shared, **settings['medium5'],
        #   # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
        #   "agent": tune.grid_search(['muzero']),
        # },
        # {**shared, **settings['medium7'],
        #   # "agent": tune.grid_search(['muzero', 'factored', 'branched']),
        #   "agent": tune.grid_search(['muzero']),
        # },
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
        # "seed": tune.grid_search([3]),
        "partial_obs": True,
         **settings['place7'],
    }
    space = [
        {
            **shared, #4
            "group": 'B40-contrastive',
            "agent": tune.grid_search(['branched']),
            "seed": tune.grid_search([3,4]),
            "learned_weights": tune.grid_search(['softmax']),
            "relation_dim": tune.grid_search([512]),
            "savi_epsilon": tune.grid_search([1e-5, 1e-8]),
            "extra_contrast": tune.grid_search([40, 80]),
        },
        # {
        #     **shared, #4
        #     "group": 'B40-target',
        #     "agent": tune.grid_search(['branched']),
        #     "seed": tune.grid_search([3]),
        #     "learned_weights": tune.grid_search(['softmax']),
        #     "reanalyze_ratio": tune.grid_search([0.0, .25]),
        #     "root_target": tune.grid_search(['mcts']),
        #     "relation_dim": tune.grid_search([256]),
        #     "savi_epsilon": tune.grid_search([1e-5, 1e-8]),
        # },
        # {
        #     **shared, #4
        #     "group": 'B40-relation',
        #     "agent": tune.grid_search(['branched']),
        #     "seed": tune.grid_search([3]),
        #     "learned_weights": tune.grid_search(['softmax']),
        #     "relation_dim": tune.grid_search([256, 512]),
        #     "savi_epsilon": tune.grid_search([1e-5, 1e-8]),
        # },
    ]
  elif search == 'replicate':
    shared = {
        "seed": tune.grid_search([5]),
        "env.partial_obs": True,
         **settings['place7'],
    }
    space = [
        # {
        #     **shared, #4
        #     "group": 'B35-replicate-terminate',
        #     "agent": tune.grid_search(['muzero', 'branched']),
        #     "timeout_truncate": tune.grid_search([True, False]),
        # },
        {
            **shared, #4
            "group": 'B37-replicate-6',
            "agent": tune.grid_search(['branched']),
            "learned_weights": tune.grid_search(
              ['none', 'softmax']),
            # "seperate_model_nets": tune.grid_search(
            #   [True, False]),
            # "staircase_decay": tune.grid_search([True, False]),
            "reanalyze_ratio": tune.grid_search([.25]),
        },
    ]

  elif search == 'conv_muzero':
    shared = {
        "seed": tune.grid_search([5]),
        "env.partial_obs": True,
         **settings['place7'],
    }
    space = [
        {
            **shared, #4
            "group": 'conv-muzero-4',
            "agent": tune.grid_search([
              'muzero',
              'conv_muzero']),
        },
    ]
  elif search == 'conv_factored':
    shared = {
        "seed": tune.grid_search([5]),
        "env.partial_obs": True,
         **settings['place5'],
    }
    space = [
        {
            **shared, #4
            "group": 'conv-factored-6',
            "agent": tune.grid_search(['conv_factored']),
            "num_slots": tune.grid_search([1, 4]),
            # "seperate_model_nets": tune.grid_search([False, True]),
            "transition_blocks": tune.grid_search([2]),
            "conv_lstm_dim": tune.grid_search([32, 64]),
            "context_slot_dim": tune.grid_search([0]),
            "learned_weights": tune.grid_search(['none', 'softmax']),
            # "vpi_mlps": tune.grid_search([[128, 32], [256, 256]]),
        },
    ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  assert FLAGS.parallel in ('ray', 'sbatch', 'none')
  if FLAGS.parallel in ('ray', 'sbatch'):
    run_many()
  else:
    run_single()

if __name__ == '__main__':
  app.run(main)