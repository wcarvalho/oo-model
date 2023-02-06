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

from experiments import helpers
from absl import flags
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import dm_env
import launchpad as lp

import functools

from acme.agents.jax import r2d2

from muzero.config import MuZeroConfig
from muzero.builder import MuZeroBuilder
from muzero import networks
# from muzero import config

from experiments import network_defs

# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('tasks_file', 'place_split_hard', 'tasks_file')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 1_000_000,
                     'Number of environment steps to run for.')

FLAGS = flags.FLAGS



def build_experiment_config(launch=False):
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  tasks_file = FLAGS.tasks_file

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_kitchen_environment(
      tasks_file=tasks_file,
      evaluation=False)

  # Configure the agent.
  config = MuZeroConfig(
      burn_in_length=8 if launch else 4,
      trace_length=40 if launch else 10,
      sequence_period=20 if launch else 10,
      min_replay_size=10_000 if launch else 100,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=1.0,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
  )

  config.vocab_size = 50
  config.word_dim = 32
  config.sentence_dim = 32
  config.resnet_transition_dim = 256
  config.num_blocks = 8
  config.num_bins = 301
  config.simulation_steps = 4
  config.model_state_extract_fn = lambda state: state.hidden
  config.num_simulations = 50 if launch else 2
  config.maxvisit_init = 50
  config.gumbel_scale = 1.0
  config.td_steps = 4

  return experiments.ExperimentConfig(
      builder=MuZeroBuilder(config),
      network_factory=functools.partial(
        networks.make_babyai_networks, config=config),
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  if FLAGS.run_distributed:
    config = build_experiment_config(launch=True)
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4 if lp_utils.is_local_run() else 80)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    config = build_experiment_config(launch=False)
    experiments.run_experiment(experiment=config)


if __name__ == '__main__':
  app.run(main)
