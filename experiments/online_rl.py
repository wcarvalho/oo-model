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
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# https://github.com/google/jax/issues/8302
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import dataclasses

from typing import Optional

from absl import flags
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
from acme.utils import experiment_utils
from acme.utils import loggers
import dm_env
import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

import functools

from acme.agents.jax import r2d2

from muzero.config import MuZeroConfig
from muzero.builder import MuZeroBuilder
from muzero import networks
# from muzero import config

from experiments import helpers
from experiments import logger as wandb_logger 


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'muzero', 'which agent.')
# flags.DEFINE_string('env', 'fruitbot', 'which environment.')
# flags.DEFINE_string('env_setting', '', 'which environment setting.')
# flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_bool('test', True, 'whether testing.')
# flags.DEFINE_bool('evaluate', True, 'whether to use evaluation policy.')
# flags.DEFINE_bool('init_only', False, 'whether to only init arch.')

# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('tasks_file', 'place_split_hard', 'tasks_file')
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
flags.DEFINE_string('wandb_group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('wandb_notes', '', 'notes for wandb.')
flags.DEFINE_string('folder', '', 'notes for wandb.')

FLAGS = flags.FLAGS


def build_experiment_config(launch=False,
                            agent='muzero',
                            config_kwargs: dict = None,
                            env_kwargs: dict = None,
                            log_dir: str = None,
                            path: str = '.',
                            log_every: int = 30.0,
                            log_with_key: Optional[str] = None,
                            wandb_init_kwargs = None):
  """Builds R2D2 experiment config which can be executed in different ways."""

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_kwargs = env_kwargs or dict()
  if not launch: #DEBUG
    config_kwargs['min_replay_size'] = 1000
    config_kwargs['seperate_model_nets'] = False
    config_kwargs['burn_in_length'] = 0
    config_kwargs['simulation_steps'] = 8
    config_kwargs['action_source'] = 'value'
  # tasks_file = FLAGS.tasks_file

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_kitchen_environment(
      path=path,
      **env_kwargs)

  # Configure the agent.
  config = MuZeroConfig(
      # burn_in_length=8 if launch else 4,
      # trace_length=40 if launch else 10,
      # # sequence_period=20 if launch else 10,
      # min_replay_size=10_000 if launch else 100,
      # batch_size=32,
      # prefetch_size=1,
      # samples_per_insert=1.0,
      # evaluation_epsilon=1e-3,
      # learning_rate=1e-4,
      # target_update_period=1200,
      # variable_update_period=100,
      # num_simulations = 50 if launch else 2
  )
  # update with config kwargs
  for k, v in config_kwargs.items():
    setattr(config, k, v)

  config.trace_length = config.burn_in_length + config.simulation_steps + config.td_steps + 1
  config.sequence_period = config.trace_length - 1
  # TODO: implacing conv_kwargs
  # TODO: swapping based on agent

  # -----------------------
  # wandb setup
  # -----------------------
  use_wandb = wandb_init_kwargs is not None
  if use_wandb:
    import wandb
    # add config to wandb
    wandb_config = wandb_init_kwargs.get("config", {})
    # wandb_config.update(save_config_dict)
    wandb_init_kwargs['config'] = wandb_config
    if launch:  # distributed
      wandb.init(
        settings=wandb.Settings(start_method="fork"),
        reinit=True, **wandb_init_kwargs)
    else:
      wandb.init(**wandb_init_kwargs)

  # -----------------------
  # create logger factory
  # -----------------------
  def logger_factory(
    name: str,
    steps_key: Optional[str] = None,
    task_id: Optional[int] = None,
    ) -> loggers.Logger:
    if use_wandb and launch:
      wandb.init(
        settings=wandb.Settings(start_method="fork"),
        reinit=True, **wandb_init_kwargs)
    if name == 'actor':
      return wandb_logger.make_logger(
          log_dir=log_dir,
          label='actor',
          time_delta=log_every,
          log_with_key=log_with_key,
          steps_key=steps_key,
          save_data=task_id == 0,
          use_wandb=use_wandb)
    elif name == 'evaluator':
      return wandb_logger.make_logger(
          log_dir=log_dir,
          label='evaluator',
          time_delta=log_every,
          log_with_key=log_with_key,
          steps_key=steps_key,
          use_wandb=use_wandb)
    elif name == 'learner':
      return wandb_logger.make_logger(
          log_dir=log_dir,
          label='learner',
          time_delta=log_every,
          log_with_key=log_with_key,
          steps_key=steps_key,
          use_wandb=use_wandb,
          asynchronous=True)

  return experiments.ExperimentConfig(
      builder=MuZeroBuilder(config),
      network_factory=functools.partial(
        networks.make_babyai_networks, config=config),
      environment_factory=environment_factory,
      seed=config.seed,
      max_num_actor_steps=config.num_steps,
      logger_factory=logger_factory)

def make_distributed_program(num_actors: int = 4, **kwargs):
  config = build_experiment_config(launch=True, **kwargs)
  program = experiments.make_distributed_experiment(
      experiment=config,
      num_actors=num_actors)

  local_resources = {
      "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
  }

  return program, local_resources

def main(_):
  wandb_init_kwargs = dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=FLAGS.wandb_group if FLAGS.wandb_group else FLAGS.agent,
    notes=FLAGS.wandb_notes,
  )

  wandb_init_kwargs = wandb_init_kwargs if FLAGS.use_wandb else None
  log_dir_fn = lambda b: wandb_logger.gen_log_dir(
        base_dir=b,
        date=True,
        hourminute=True,
        agent=FLAGS.seed,
        seed=FLAGS.seed,
        return_kwpath=True)
  if FLAGS.run_distributed:
    log_dir, config_path_str = log_dir_fn("results/babyai/debug_async")
  else:
    log_dir, config_path_str = log_dir_fn("results/babyai/debug_async")
  if wandb_init_kwargs is not None:
    wandb_init_kwargs['name'] = config_path_str

  env_kwargs = dict(
      tasks_file=FLAGS.tasks_file,
  )
  config_kwargs = dict(
      num_steps=FLAGS.num_steps,
  )
  if FLAGS.run_distributed:
    program, local_resources = make_distributed_program(
      log_dir=log_dir,
      wandb_init_kwargs=wandb_init_kwargs,
      env_kwargs=env_kwargs,
      config_kwargs=config_kwargs,
      )
    lp.launch(program,
              lp.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal='current_terminal',
              local_resources=local_resources)
  else:
    config = build_experiment_config(
      launch=False,
      log_dir=log_dir,
      wandb_init_kwargs=wandb_init_kwargs,
      env_kwargs=env_kwargs,
      config_kwargs=config_kwargs,
      )
    experiments.run_experiment(experiment=config)


if __name__ == '__main__':
  app.run(main)
