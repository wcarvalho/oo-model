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

"""MuZero learner implementation.

Built off of R2D2 since value-based recurrent method. This adds learning the MuZero model.
"""

import dataclasses

import functools
import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

from absl import logging
import acme
from acme import types as acme_types
from acme.adders import reverb as adders
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree

from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.agents.jax.dqn import learning_lib
from muzero import types as muzero_types
from muzero.config import MuZeroConfig
from muzero.loss import muzero_loss

from pprint import pprint
aprint = lambda a: pprint(jax.tree_map(lambda x:x.shape, a))

_PMAP_AXIS_NAME = 'data'
# This type allows splitting a sample between the host and device, which avoids
# putting item keys (uint64) on device for the purposes of priority updating.
R2D2ReplaySample = utils.PrefetchingSplit

class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: muzero_types.MuZeroParams
  target_params: muzero_types.MuZeroParams
  opt_state: optax.OptState
  steps: int
  random_key: networks_lib.PRNGKey


class MuZeroLearner(acme.Learner):
  """R2D2 learner."""

  def __init__(self,
               networks: r2d2_networks.R2D2Networks,
               batch_size: int,
               random_key: networks_lib.PRNGKey,
               burn_in_length: int,
               discount: float,
               importance_sampling_exponent: float,
               max_priority_weight: float,
               target_update_period: int,
               iterator: Iterator[R2D2ReplaySample],
               optimizer: optax.GradientTransformation,
               config: MuZeroConfig,
               bootstrap_n: int = 5,
               tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
               clip_rewards: bool = False,
               max_abs_reward: float = 1.,
               use_core_state: bool = True,
               prefetch_size: int = 2,
               loss_fn = muzero_loss,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    """Initializes the learner."""

    simulation_steps = config.simulation_steps
    num_bins = config.num_bins
    model_state_extract_fn = config.model_state_extract_fn

    num_simulations = config.num_simulations
    maxvisit_init = config.maxvisit_init
    gumbel_scale = config.gumbel_scale

    td_steps = config.td_steps

    def loss(
        params: muzero_types.MuZeroParams,
        target_params: muzero_types.MuZeroParams,
        key_grad: networks_lib.PRNGKey,
        sample: reverb.ReplaySample
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
      """Computes loss for a batch of sequences."""

      #####################
      # Initialize + Burn-in state
      #####################
      # Get core state & warm it up on observations for a burn-in period.
      if use_core_state:
        # Replay core state.
        # NOTE: We may need to recover the type of the hk.LSTMState if the user
        # specifies a dynamically unrolled RNN as it will strictly enforce the
        # match between input/output state types.
        online_state = utils.maybe_recover_lstm_type(
            sample.data.extras.get('core_state'))
      else:
        key_grad, initial_state_rng = jax.random.split(key_grad)
        online_state = networks.init_recurrent_state(initial_state_rng,
                                                     batch_size)
      target_state = online_state

      # Convert sample data to sequence-major format [T, B, ...].
      data = utils.batch_to_sequence(sample.data)
      # Maybe burn the core state in.
      if burn_in_length:
        burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
        key_grad, key1, key2 = jax.random.split(key_grad, 3)
        _, online_state = networks.unroll(params.unroll, key1, burn_obs, online_state)
        _, target_state = networks.unroll(target_params.unroll, key2, burn_obs,
                                          target_state)

      #####################
      # Compute state + quantities for rest of trajectory
      #####################
      # Only get data to learn on from after the end of the burn in period.
      data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

      # Unroll on sequences to get online and target Q-Values.
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      online_outputs, online_state = networks.unroll(
        params.unroll, key1, data.observation, online_state)
      target_outputs, target_state = networks.unroll(
        target_params.unroll, key2, data.observation, target_state)

      loss_fn_ = functools.partial(loss_fn,
        networks=networks,
        params=params,
        target_params=target_params,
        rng_key=key_grad,
        simulation_steps=simulation_steps,
        num_bins=num_bins,
        model_state_extract_fn=model_state_extract_fn,
        num_simulations=num_simulations,
        maxvisit_init=maxvisit_init,
        gumbel_scale=gumbel_scale,
        discount=discount,
        td_steps=td_steps,
        )
      loss_fn_ = jax.vmap(loss_fn_, in_axes=(1, 1, 1, 0, 0), out_axes=1) # vmap over batch dimension
      # [T, B]
      batch_loss, metrics = loss_fn_(data,
                               online_outputs,
                               target_outputs,
                               online_state,
                               target_state)

      # Importance weighting.
      probs = sample.info.probability
      importance_weights = (1. / (probs + 1e-6)).astype(batch_loss.dtype)
      importance_weights **= importance_sampling_exponent
      importance_weights /= jnp.max(importance_weights)
      mean_loss = jnp.mean(importance_weights * batch_loss)

      # Calculate priorities as a mixture of max and mean sequence errors.
      value_target = metrics['value_target']
      value_prediction = metrics['value_prediction']
      batch_td_error = value_target - value_prediction

      abs_td_error = jnp.abs(batch_td_error).astype(batch_loss.dtype)
      max_priority = max_priority_weight * jnp.max(abs_td_error, axis=0)
      mean_priority = (1 - max_priority_weight) * jnp.mean(abs_td_error, axis=0)
      priorities = (max_priority + mean_priority)
      # priorities = jnp.zeros_like(mean_loss)

      metrics = jax.tree_map(lambda x: x.mean(), metrics)
      extra = learning_lib.LossExtra(metrics=metrics,
                                     reverb_priorities=priorities)

      return mean_loss, extra

    def sgd_step(
        state: TrainingState,
        samples: reverb.ReplaySample
    ) -> Tuple[TrainingState, jnp.ndarray, Dict[str, jnp.ndarray]]:
      """Performs an update step, averaging over pmap replicas."""

      # Compute loss and gradients.
      grad_fn = jax.value_and_grad(loss, has_aux=True)
      key, key_grad = jax.random.split(state.random_key)
      (loss_value, extra), gradients = grad_fn(
        state.params,
        state.target_params,
        key_grad,
        samples)

      metrics = extra.metrics
      priorities = extra.reverb_priorities

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

      # Apply optimizer updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1
      target_params = optax.periodic_update(new_params, state.target_params,
                                            steps, self._target_update_period)

      new_state = TrainingState(
          params=new_params,
          target_params=target_params,
          opt_state=new_opt_state,
          steps=steps,
          random_key=key)

      return new_state, priorities, metrics

    def update_priorities(
        keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
      keys, priorities = keys_and_priorities
      keys, priorities = tree.map_structure(
          # Fetch array and combine device and batch dimensions.
          lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
          (keys, priorities))
      replay_client.mutate_priorities(  # pytype: disable=attribute-error
          table=adders.DEFAULT_PRIORITY_TABLE,
          updates=dict(zip(keys, priorities)))

    # Internalise components, hyperparameters, logger, counter, and methods.
    self._iterator = iterator
    self._replay_client = replay_client
    self._target_update_period = target_update_period
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        time_delta=1.,
        steps_key=self._counter.get_steps_key())

    self._sgd_step = jax.pmap(sgd_step, axis_name=_PMAP_AXIS_NAME)
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

    # Initialise and internalise training state (parameters/optimiser state).
    random_key, key_init1, key_init2 = jax.random.split(random_key, 3)
    initial_params = muzero_types.MuZeroParams(
      unroll=networks.unroll_init(key_init1),
      model= networks.model_init(key_init2),
    )
    opt_state = optimizer.init(initial_params)

    # Log how many parameters the network has.
    sizes = tree.map_structure(jnp.size, initial_params)
    logging.info('Total number of params: %.3g',
                 sum(tree.flatten(sizes.unroll.values())) + 
                 sum(tree.flatten(sizes.model.values())))

    state = TrainingState(
        params=initial_params,
        target_params=initial_params,
        opt_state=opt_state,
        steps=jnp.array(0),
        random_key=random_key)
    # Replicate parameters.
    self._state = utils.replicate_in_all_devices(state)

  def step(self):
    prefetching_split = next(self._iterator)
    # The split_sample method passed to utils.sharded_prefetch specifies what
    # parts of the objects returned by the original iterator are kept in the
    # host and what parts are prefetched on-device.
    # In this case the host property of the prefetching split contains only the
    # replay keys and the device property is the prefetched full original
    # sample.
    keys = prefetching_split.host
    samples: reverb.ReplaySample = prefetching_split.device

    # Do a batch of SGD.
    start = time.time()
    self._state, priorities, metrics = self._sgd_step(self._state, samples)
    # Take metrics from first replica.
    metrics = utils.get_from_first_device(metrics)
    # Update our counts and record it.
    time_elapsed = time.time() - start
    metrics['LearnerDuraction'] = time_elapsed
    counts = self._counter.increment(steps=1, time_elapsed=time_elapsed)

    # Update priorities in replay.
    if self._replay_client:
      self._async_priority_updater.put((keys, priorities))

    # Attempt to write logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[muzero_types.MuZeroParams]:
    del names  # There's only one available set of params in this agent.
    # Return first replica of parameters.
    return utils.get_from_first_device([self._state.params])

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return utils.get_from_first_device(self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state)
