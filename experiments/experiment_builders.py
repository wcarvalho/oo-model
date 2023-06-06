# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example runing model-based agents on discrete control tasks.

Copied from: https://github.com/deepmind/acme/blob/master/examples/baselines/rl_discrete/run_r2d2.py
"""
# Do not preallocate GPU memory for JAX.
from launchpad.nodes.python.local_multi_processing import PythonProcess
import pickle
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# https://github.com/google/jax/issues/8302
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import dataclasses
import datetime
from typing import Optional, NamedTuple, Any, Callable, List, Iterator


from functools import partial
from absl import flags
from absl import app
from acme.jax import experiments
from acme.utils import loggers
from acme.utils import paths
from acme.jax import types
from acme.agents.jax import builders
from acme import specs
from acme.utils.observers import EnvLoopObserver

import dm_env
import launchpad as lp

from ray import tune


from experiments import logger as wandb_logger 
from experiments import config_utils


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'muzero', 'which agent.')

# Flags which modify the behavior of the launcher.
flags.DEFINE_string('agent_config', '', 'config file')
flags.DEFINE_string('env_config', '', 'config file')
flags.DEFINE_string('path', '.', 'config file')
# flags.DEFINE_string('tasks_file', 'pickup_place', 'tasks_file')
flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 1_000_000,
                     'Number of environment steps to run for.')
flags.DEFINE_integer('debug', 0, 'Random seed (experiment).')


# -----------------------
# wandb
# -----------------------
flags.DEFINE_bool('use_wandb', False, 'whether to log.')
flags.DEFINE_string('wandb_project', None, 'wand project.')
flags.DEFINE_string('wandb_entity', None, 'wandb entity')
flags.DEFINE_string('wandb_dir', None, 'wandb directory')
flags.DEFINE_string('wandb_group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('wandb_name', '', 'name of run. way to group runs.')
flags.DEFINE_string('wandb_notes', '', 'notes for wandb.')
flags.DEFINE_string('folder', '', 'folder for experiments.')

FLAGS = flags.FLAGS

Seed = int
Eval = bool

def default_setup_agents(
    agent: str,
    config_kwargs: dict = None,
    env_kwargs: dict = None,
    debug: bool = False,
    update_logger_kwargs = None,
):
  config_kwargs = config_kwargs or dict()
  update_logger_kwargs = update_logger_kwargs or dict()
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
    from muzero import learner_logger

    config = babyai_muzero.load_config(
      config_kwargs=config_kwargs)

    builder_kwargs = dict(
      update_logger=learner_logger.LearnerLogger(
          label='MuZeroLearnerLogger',
          log_frequency=5 if debug else 2500,
          discount=config.discount,
          **update_logger_kwargs,
      ),
    )
    builder, network_factory = babyai_muzero.setup(
      config=config,
      builder_kwargs=builder_kwargs,
      config_kwargs=config_kwargs)

  elif agent == 'factored':
    from experiments import babyai_factored_muzero
    from factored_muzero.analysis_actor import VisualizeActor, AttnLogger
    from factored_muzero import learner_logger

    config = babyai_factored_muzero.load_config(
        config_kwargs=config_kwargs)

    builder_kwargs = dict(
      actorCls=partial(
        VisualizeActor,
        logger=AttnLogger(),
        log_frequency=5 if debug else 2500),
      update_logger=learner_logger.LearnerLogger(
          label='FactoredMuZeroLearnerLogger',
          log_frequency=5 if debug else 2500,
          discount=config.discount,
          **update_logger_kwargs,
      ),
    )
    assert env_kwargs is not None
    room_size = env_kwargs['room_size']
    return babyai_factored_muzero.setup(
      config=config,
      network_kwargs=dict(num_spatial_vectors=room_size**2),
      builder_kwargs=builder_kwargs)
  else:
    raise NotImplementedError
  
  return config, builder, network_factory


def setup_logger_factory(
    agent_config,
    debug: bool = False,
    save_config_dict: dict = None,
    log_dir: str = None,
    log_every: int = 30.0,
    log_with_key: Optional[str] = 'log_data',
    actor_label: str = 'actor',
    evaluator_label: str = 'evaluator',
    learner_label: str = 'learner',
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
      task_id: Optional[int] = None,
  ) -> loggers.Logger:
    if custom_steps_keys is not None:
      steps_key = custom_steps_keys(name)
    if use_wandb:
      wandb.init(**wandb_init_kwargs)

    if name == 'actor':
      return wandb_logger.make_logger(
          log_dir=log_dir,
          label=actor_label,
          time_delta=0.0,
          log_with_key=log_with_key,
          steps_key=steps_key,
          save_data=task_id == 0,
          use_wandb=use_wandb)
    elif name == 'evaluator':
      return wandb_logger.make_logger(
          log_dir=log_dir,
          label=evaluator_label,
          time_delta=0.0,
          log_with_key=log_with_key,
          steps_key=steps_key,
          use_wandb=use_wandb)
    elif name == 'learner':
      return wandb_logger.make_logger(
          log_dir=log_dir,
          label=learner_label,
          time_delta=log_every,
          steps_key=steps_key,
          use_wandb=use_wandb,
          asynchronous=True)
  return logger_factory


def setup_evaluator_factories(
        builder,
        environment_factory,
        network_factory,
        logger_factory,
        observers):
 # -----------------------
  # create evaluator factory
  # -----------------------
  def eval_policy_factory(networks: builders.Networks,
                          environment_spec: specs.EnvironmentSpec,
                          evaluation: bool) -> builders.Policy:
    del evaluation
    return builder.make_policy(
        networks=networks,
        environment_spec=environment_spec,
        evaluation=True)

  return [
      experiments.default_evaluator_factory(
          environment_factory=partial(environment_factory,
                                      evaluation=True),  # Key difference
          network_factory=network_factory,
          policy_factory=eval_policy_factory,
          logger_factory=logger_factory,
          observers=observers)
  ]


class OnlineExperimentConfigInputs(NamedTuple):
  agent_config: dict
  final_env_kwargs: dict
  builder: Any
  network_factory: Callable[[specs.EnvironmentSpec], builders.Networks]
  environment_factory: Callable[[Seed, Eval], dm_env.Environment]
  observers: Optional[List[EnvLoopObserver]] = None


def build_online_experiment_config(
  experiment_config_inputs: OnlineExperimentConfigInputs,
  agent: str,
  debug: bool = False,
  save_config_dict: dict = None,
  log_dir: str = None,
  log_every: int = 30.0,
  log_with_key: Optional[str] = 'log_data',
  observers: Optional[List[EnvLoopObserver]] = None,
  wandb_init_kwargs: dict = None,
  logger_factory_kwargs: dict = None
  ):
  """Builds experiment config."""
  agent_config = experiment_config_inputs.agent_config
  builder = experiment_config_inputs.builder
  network_factory = experiment_config_inputs.network_factory
  environment_factory = experiment_config_inputs.environment_factory
  env_kwargs = experiment_config_inputs.final_env_kwargs
  observers = experiment_config_inputs.observers or ()
  logger_factory_kwargs = logger_factory_kwargs or dict()
  wandb_init_kwargs = wandb_init_kwargs or dict()

  assert log_dir, 'provide directory for logging experiments via FLAGS.folder'
  paths.process_path(log_dir)
  config_utils.save_config(f'{log_dir}/config.pkl', agent_config.__dict__)

  save_config_dict = save_config_dict or dict()
  save_config_dict.update(
        agent=agent,
        group=wandb_init_kwargs.get('group', None),
        **env_kwargs,
      )
  logger_factory = setup_logger_factory(
      agent_config,
      debug=debug,
      save_config_dict=save_config_dict,
      log_dir=log_dir,
      log_every=log_every,
      log_with_key=log_with_key,
      wandb_init_kwargs=wandb_init_kwargs,
      **logger_factory_kwargs,
  )

  evaluator_factories = setup_evaluator_factories(
      builder=builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
      logger_factory=logger_factory,
      observers=observers)

  return experiments.ExperimentConfig(
      builder=builder,
      network_factory=network_factory,
      environment_factory=environment_factory,
      seed=agent_config.seed,
      max_num_actor_steps=agent_config.num_steps,
      observers=observers,
      logger_factory=logger_factory,
      evaluator_factories=evaluator_factories,
      checkpointing=experiments.CheckpointingConfig(
          directory=log_dir,
          max_to_keep=5,
          add_uid=False,
          checkpoint_ttl_seconds=int(datetime.timedelta(days=30).total_seconds()))
      )


class OfflineExperimentConfigInputs(NamedTuple):
  agent_config: dict
  final_env_kwargs: dict
  builder: Any
  network_factory: Callable[[specs.EnvironmentSpec], builders.Networks]
  demonstration_dataset_factory: Callable[[types.PRNGKey],
                                          Iterator[builders.Sample]]
  environment_spec: specs.EnvironmentSpec
  environment_factory: Callable[[Seed, Eval], dm_env.Environment]
  observers: Optional[List[EnvLoopObserver]] = None


def build_offline_experiment_config(
    experiment_config_inputs: OfflineExperimentConfigInputs,
    agent: str,
    debug: bool=False,
    save_config_dict: dict = None,
    log_dir: str = None,
    log_every: int = 30.0,
    log_with_key: Optional[str] = 'log_data',
    observers: Optional[List[EnvLoopObserver]] = None,
    wandb_init_kwargs: dict = None,
    logger_factory_kwargs: dict = None
    ):
  """Returns a config for BC experiments."""
  observers = observers or ()
  agent_config = experiment_config_inputs.agent_config
  builder = experiment_config_inputs.builder
  network_factory = experiment_config_inputs.network_factory
  demonstration_dataset_factory = experiment_config_inputs.demonstration_dataset_factory
  environment_factory = experiment_config_inputs.environment_factory
  environment_spec = experiment_config_inputs.environment_spec
  env_kwargs = experiment_config_inputs.final_env_kwargs
  observers = experiment_config_inputs.observers or ()
  logger_factory_kwargs = logger_factory_kwargs or dict()
  wandb_init_kwargs = wandb_init_kwargs or dict()

  assert log_dir, 'provide directory for logging experiments via FLAGS.folder'
  paths.process_path(log_dir)
  config_utils.save_config(f'{log_dir}/config.pkl', agent_config.__dict__)

  save_config_dict = save_config_dict or dict()
  save_config_dict.update(
      agent=agent,
      group=wandb_init_kwargs.get('group', None),
      **env_kwargs,
  )

  logger_factory = setup_logger_factory(
      agent_config,
      debug=debug,
      save_config_dict=save_config_dict,
      log_dir=log_dir,
      log_every=log_every,
      log_with_key=log_with_key,
      wandb_init_kwargs=wandb_init_kwargs,
      **logger_factory_kwargs,
  )
  evaluator_factories = setup_evaluator_factories(
      builder=builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
      logger_factory=logger_factory,
      observers=observers)

  return experiments.OfflineExperimentConfig(
      builder=builder,
      network_factory=network_factory,
      demonstration_dataset_factory=demonstration_dataset_factory,
      environment_factory=environment_factory,
      max_num_learner_steps=agent_config.num_learner_steps,
      seed=agent_config.seed,
      environment_spec=environment_spec,
      observers=observers,
      logger_factory=logger_factory,
      evaluator_factories=evaluator_factories,
      checkpointing=experiments.CheckpointingConfig(
          directory=log_dir,
          max_to_keep=5,
          add_uid=False,
          checkpoint_ttl_seconds=int(datetime.timedelta(days=30).total_seconds()))
  )