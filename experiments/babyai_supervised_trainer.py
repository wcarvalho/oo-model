
import functools 

from typing import Optional, Callable

from absl import flags
from absl import app
from absl import logging
import collections
from ray import tune


import acme
from acme import specs
from acme.jax import experiments
from acme.jax.experiments import config
from acme.utils import counting
from acme.tf import savers
from acme.utils import paths

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from pprint import pprint

from envs.multitask_kitchen import Observation
from experiments import config_utils
from experiments import train_many
from experiments import experiment_builders
from experiments import babyai_env_utils
from experiments import babyai_collect_data
from experiments import dataset_utils
from experiments import logger as wandb_logger
from experiments import offline_configs
from experiments.babyai_online_trainer import setup_agents

FLAGS = flags.FLAGS


def setup_logger_factory(
    agent_config,
    save_config_dict: dict = None,
    log_dir: str = None,
    log_every: int = 30.0,
    custom_steps_keys: Optional[Callable[[str], str]] = None,
    wandb_init_kwargs=None,
):
  """Builds experiment config."""

  assert log_dir, 'provide directory for logging experiments via FLAGS.folder'
  paths.process_path(log_dir)
  config_utils.save_config(f'{log_dir}/config.pkl', agent_config.__dict__)
  # -----------------------
  # wandb setup
  # -----------------------
  wandb_init_kwargs = wandb_init_kwargs or dict()
  save_config_dict = save_config_dict or dict()

  use_wandb = len(wandb_init_kwargs)
  if use_wandb:
    import wandb

    # add config to wandb
    wandb_config = wandb_init_kwargs.get("config", {})
    save_config_dict = save_config_dict or dict()
    save_config_dict.update(agent_config.__dict__)
    wandb_config.update(save_config_dict)

    wandb_init_kwargs['config'] = wandb_config
    wandb_init_kwargs['dir'] = log_dir
    wandb_init_kwargs['reinit'] = True
    wandb_init_kwargs['settings'] = wandb.Settings(
        code_dir=log_dir,
        start_method="fork")
    wandb.init(**wandb_init_kwargs)

  # -----------------------
  # create logger factory
  # -----------------------
  def logger_factory(
      name: str,
      steps_key: Optional[str] = None,
  ):
    if custom_steps_keys is not None:
      steps_key = custom_steps_keys(name)
    return wandb_logger.make_logger(
        log_dir=log_dir,
        label=name,
        time_delta=log_every,
        steps_key=steps_key,
        use_wandb=use_wandb,
        asynchronous=True)
  return logger_factory


def run_experiment(
    agent : str,
    nepisode_dataset: int,
    path: str = '.',
    agent_config_kwargs: dict=None,
    agent_config_file: str=None,
    env_kwargs: dict=None,
    env_config_file: str=None,
    debug: bool = False,
    log_dir: str = None,
    wandb_init_kwargs=None,
    dataset_percent: int = 100,
    min_validation: float = 1e10,
    tolerance: int = 10,
    train_batches: int = int(1e4),
    eval_batches: int = int(1e2),
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
  # setup agent
  # -----------------------
  if agent == 'r2d2':
    config_class = offline_configs.R2D2Config
  elif agent == 'muzero':
    config_class = offline_configs.MuZeroConfig
    loss_kwargs = dict(behavior_clone=True)

  elif agent == 'factored':
    config_class = offline_configs.FactoredMuZeroConfig
    loss_kwargs = dict(behavior_clone=True)

  env_actions = ['left', 'right', 'forward', 'pickup_container',
                 'pickup_contents', 'place', 'toggle', 'slice']
  tasks = babyai_env_utils.open_kitchen_tasks_file(
    tasks_file=env_kwargs['tasks_file'], path=path)
  valid_actions = [(a in tasks['valid_actions']) for a in env_actions]
  invalid_actions = 1-jnp.array(valid_actions, dtype=jnp.float32)

  config, builder, network_factory = setup_agents(
    agent=agent,
    debug=debug,
    config_kwargs=config_kwargs,
    env_kwargs=env_kwargs,
    config_class=config_class,
    setup_kwargs=dict(
      loss_kwargs=loss_kwargs,
      invalid_actions=invalid_actions,
    ),
    update_logger_kwargs=dict(
        action_names=['left', 'right', 'forward', 'pickup_1',
                  'pickup_c', 'place', 'toggle', 'slice'],
        invalid_actions=np.asarray(invalid_actions),
    )
  )
  logging.info(f'Config')
  pprint(config.__dict__)


  environment = babyai_env_utils.make_kitchen_environment(**env_kwargs)
  environment_spec = specs.make_environment_spec(environment)
  # Create the networks and policy.
  networks = network_factory(environment_spec)


  # Define the demonstrations factory.
  def make_data_directory(eval, nepisodes):
    return babyai_collect_data.directory_name(
      tasks_file=env_kwargs['tasks_file'],
      room_size=env_kwargs['room_size'],
      partial_obs=env_kwargs['partial_obs'],
      nepisodes=nepisodes,
      evaluation=eval,
      debug=debug)

  def make_iterator(data_directory: str,
                    split: str,
                    buffer_size: int = 10_000):
    dataset_factory = dataset_utils.make_demonstration_dataset_factory(
      data_directory=data_directory,
      obs_constructor=Observation,
      batch_size=config.batch_size,
      trace_length=config.trace_length)
    return dataset_factory(0,
                           split=split,
                           buffer_size=buffer_size)
  
  train_split = 'train'
  if dataset_percent < 100:
    train_split = 'train[:{dataset_percent}%]'
  train_iterator = make_iterator(
    data_directory=make_data_directory(
      eval=False, nepisodes=nepisode_dataset),
    split=train_split)
  validation_iterator = make_iterator(
    data_directory=make_data_directory(
      eval=False, nepisodes=int(1e5)),
    split='train')
  eval_iterator = make_iterator(
    data_directory=make_data_directory(
      eval=True, nepisodes=int(1e5)),
    split='test')


  # Create learners.
  parent_counter = counting.Counter(time_delta=0.)
  logger_fn = setup_logger_factory(
    agent_config=config,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
  )

  key = jax.random.PRNGKey(config.seed)
  learner_key, key = jax.random.split(key)
  learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=train_iterator,
        logger_fn=logger_fn,
        environment_spec=environment_spec,
        counter=counting.Counter(parent_counter,
                                 prefix='learner',
                                 time_delta=0.))
  validation_logger = logger_fn('valid_loss')
  eval_logger = logger_fn('eval_loss')

  # -----------------------
  # training loop
  # -----------------------
  valid_increases = 0
  steps = 0
  num_learner_steps = config.num_learner_steps
  if debug:
    train_batches = eval_batches = 1
    num_learner_steps = 10

  while True:
    losses = collections.defaultdict(list)
    update_visualizations = True
    for _ in range(eval_batches):
      learner_key, key = jax.random.split(key)
      valid_loss, metrics = learner.compute_loss(
        sample=next(validation_iterator),
        loss_key=learner_key,
        update_visualizations=update_visualizations,
        log_label='validation')
      validation_logger.write(metrics)
      losses['valid_loss'].append(valid_loss)

      learner_key, key = jax.random.split(key)
      eval_loss, metrics = learner.compute_loss(
        sample=next(eval_iterator),
        loss_key=learner_key,
        update_visualizations=update_visualizations,
        log_label='eval')
      eval_logger.write(metrics)
      losses['eval_loss'].append(eval_loss)
      update_logger = False

    for _ in range(train_batches):
      learner.step()
      steps += 1

      if steps > num_learner_steps:
        break

    valid_loss = np.array(losses['valid_loss']).mean()

    if valid_loss < min_validation:
      min_validation = valid_loss
      valid_increases = 0
    else:
      valid_increases += 1

    if valid_increases >= tolerance:
      return

    if steps > num_learner_steps:
      return

def train_single(
    default_env_kwargs: dict,
    wandb_init_kwargs: dict = None,
    terminal: str = 'output_to_files',
    **kwargs
):
  debug = FLAGS.debug

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

  run_experiment(
    agent=FLAGS.agent,
    path=FLAGS.path,
    agent_config_file=FLAGS.agent_config,
    env_kwargs=default_env_kwargs,
    env_config_file=FLAGS.env_config,
    nepisode_dataset=babyai_collect_data.get_episodes(FLAGS.size, debug),
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    debug=debug)


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
            "group": tune.grid_search(['bc_muzero5']),
            "agent": tune.grid_search(['muzero']),
            "num_learner_steps": tune.grid_search([int(1e5)]),
            # "v_target_source": tune.grid_search(['reanalyze']),
            "tasks_file": tune.grid_search(['place_split_easy']),
            # "warmup_steps": tune.grid_search([1_000, 10_000, 100_000]),
            # "lr_transition_steps": tune.grid_search([1_000, 10_000, 100_000]),
            # "target_update_period": tune.grid_search([100, 2500]),
            "learning_rate": tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6]),
            # "action_source": tune.grid_search(['policy']),
            # "output_init": tune.grid_search([None, 0.0]),
            # "mask_model": tune.grid_search([True, False]),
        }
    ]
  elif search == 'muzero2':
    space = [
        {
            "seed": tune.grid_search([2]),
            "group": tune.grid_search(['muzero16']),
            "agent": tune.grid_search(['muzero']),
            "num_learner_steps": tune.grid_search([int(1e5)]),
            # "v_target_source": tune.grid_search(['reanalyze']),
            "tasks_file": tune.grid_search(['place_split_easy']),

            "warmup_steps": tune.grid_search([0]),
            "lr_transition_steps": tune.grid_search([1_000, 10_000, 100_000]),
            "learning_rate": tune.grid_search([1e-3]),
            # "scalar_step_size": tune.grid_search([.2, .1, .05]),
            # # "num_bins": tune.grid_search([51, 101, 201]),
            # # "target_update_period": tune.grid_search([100, 2500]),
            # # "learning_rate": tune.grid_search([1e-3, 1e-4]),
            # "action_source": tune.grid_search(['value', 'policy']),
            # # "output_init": tune.grid_search([None, 0.0]),
            # "gumbel_scale": tune.grid_search([.01, 1.0]),
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
      partial_obs=FLAGS.partial_obs,
  )
  if FLAGS.train_single:
    train_single(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs)
  else:
    run_distributed = FLAGS.run_distributed
    train_many.run(
      name='babyai_supervised_trainer',
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      use_wandb=use_wandb,
      debug=FLAGS.debug,
      space=sweep(search, FLAGS.agent),
      make_program_command=functools.partial(
        train_many.make_program_command,
        filename='experiments/babyai_supervised_trainer.py',
        run_distributed=run_distributed,
        num_actors=1,
        ),
    )

if __name__ == '__main__':
  app.run(main)