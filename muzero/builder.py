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

"""MuZero Builder.

Changes from R2D2:
- loss_fn is input.
  This allows for a custom loss function. Here `ValueEquivalentLoss` loss function.

- visualization_logger is input.
  This is used to visualize learner outputs.

- defaults to `LearnableStateActor` actor class.
  This allows for having a learning state function.

- builder accepts networks factory as input.
  This is used to create dummy parameters for making a dummy actor_state.
  The assumption of this class is that the intial_state function for the RNN requires parameters.
  This is used in (1) make_replay_tables & (2) make_adder

- make_learner: custom MuZero learner.

- make_policy: custom MuZero actor.

"""
from typing import Generic, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import structured
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb
from reverb import structured_writer as sw
import tensorflow as tf
import tree

from acme.agents.jax import r2d2

from muzero.learning import MuZeroLearner
from muzero import actor as muzero_actor
from muzero.ve_losses import ValueEquivalentLoss
from muzero import learner_logger
from muzero import r2d2_builder

class MuZeroBuilder(r2d2_builder.R2D2Builder):
  """MuZero Builder.

  This constructs all the pieces of MuZero.
  """

  def __init__(
      self,
      config: r2d2_config.R2D2Config,
      loss_fn: ValueEquivalentLoss,
      learnerCls: MuZeroLearner=MuZeroLearner,
      network_factory = None,
      actorCls: actors.GenericActor=muzero_actor.LearnableStateActor,
      visualization_logger: Optional[learner_logger.BaseLogger] = None,
      **kwargs):
    """Creates a R2D2 learner, a behavior policy and an eval actor."""
    super().__init__(config=config, actorCls=actorCls, **kwargs)
    self._loss_fn = loss_fn
    self._use_stored_lstm_state = config.use_stored_lstm_state
    self._visualization_logger = visualization_logger
    self._network_factory = network_factory
    self._learnerCls = learnerCls

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    # The learner updates the parameters (and initializes them).
    return self._learnerCls(
        networks=networks,
        batch_size=self._batch_size_per_device,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        LossFn=self._loss_fn,
        use_core_state=self._use_stored_lstm_state,
        replay_client=replay_client,
        counter=counter,
        config=self._config,
        logger=logger_fn('learner'),
        visualization_logger=self._visualization_logger)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: muzero_actor.R2D2Policy,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    # HACK to get parameters as input for initialization.
    dummy_key = jax.random.PRNGKey(0)
    dummy_networks = self._network_factory(environment_spec)
    dummy_params = dummy_networks.unroll_init(dummy_key)
    dummy_actor_state = policy.init(dummy_key, dummy_params)
    del dummy_params
    del dummy_networks
    # end hack

    extras_spec = policy.get_extras(dummy_actor_state)
    step_spec = structured.create_step_spec(
        environment_spec=environment_spec, extras_spec=extras_spec)
    if self._config.samples_per_insert:
      samples_per_insert_tolerance = (
          self._config.samples_per_insert_tolerance_rate *
          self._config.samples_per_insert)
      error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._config.min_replay_size,
          samples_per_insert=self._config.samples_per_insert,
          error_buffer=error_buffer)
    else:
      limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Prioritized(
                self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=sw.infer_signature(
                configs=r2d2_builder._make_adder_config(
                    step_spec, self._sequence_length,
                    self._config.sequence_period),
                step_spec=step_spec))
    ]

  def make_adder(
          self, replay_client: reverb.Client,
          environment_spec: Optional[specs.EnvironmentSpec],
          policy: Optional[muzero_actor.R2D2Policy]) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment."""
    if environment_spec is None or policy is None:
      raise ValueError('`environment_spec` and `policy` cannot be None.')
    # HACK to get parameters as input for initialization.
    dummy_key = jax.random.PRNGKey(0)
    dummy_networks = self._network_factory(environment_spec)
    dummy_params = dummy_networks.unroll_init(dummy_key)
    dummy_actor_state = policy.init(dummy_key, dummy_params)
    del dummy_params
    del dummy_networks
    # end hack

    extras_spec = policy.get_extras(dummy_actor_state)
    step_spec = structured.create_step_spec(
        environment_spec=environment_spec, extras_spec=extras_spec)
    return structured.StructuredAdder(
        client=replay_client,
        max_in_flight_items=5,
        configs=r2d2_builder._make_adder_config(step_spec, self._sequence_length,
                                                self._config.sequence_period),
        step_spec=step_spec)

  def make_policy(self,
                  networks: r2d2_networks.R2D2Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> muzero_actor.R2D2Policy:
    del environment_spec
    return muzero_actor.get_actor_core(networks,
                                       evaluation=evaluation,
                                       config=self._config)
