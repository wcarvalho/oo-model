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

class MuZeroBuilder(r2d2.R2D2Builder):
  """MuZero Builder.

  This constructs all the pieces of MuZero.
  """

  def __init__(
      self,
      config: r2d2_config.R2D2Config,
      loss_fn: ValueEquivalentLoss,
      actorCls: actors.GenericActor=muzero_actor.LearnableStateActor,
      visualization_logger: Optional[learner_logger.BaseLogger] = None,
      **kwargs):
    """Creates a R2D2 learner, a behavior policy and an eval actor."""
    super().__init__(config=config, actorCls=actorCls, **kwargs)
    self._loss_fn = loss_fn
    self._use_stored_lstm_state = config.use_stored_lstm_state
    self._visualization_logger = visualization_logger

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
    return MuZeroLearner(
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

  def make_policy(self,
                  networks: r2d2_networks.R2D2Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> muzero_actor.R2D2Policy:
    del environment_spec
    return muzero_actor.get_actor_core(networks,
                                       evaluation=evaluation,
                                       config=self._config)
