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

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree
import mctx

from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.agents.jax.dqn import learning_lib
from muzero import types as muzero_types
from muzero.config import MuZeroConfig
from muzero import ve_losses
from muzero import utils as muzero_utils

from pprint import pprint
aprint = lambda a: pprint(jax.tree_map(lambda x:x.shape, a))

_PMAP_AXIS_NAME = 'data'
# This type allows splitting a sample between the host and device, which avoids
# putting item keys (uint64) on device for the purposes of priority updating.
R2D2ReplaySample = utils.PrefetchingSplit


def episode_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    batch_loss = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    batch_loss = jnp.multiply(x, mask)
  return (batch_loss.sum(0))/(mask.sum(0)+1e-5)

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
               config: MuZeroConfig,
               discretizer: muzero_utils.Discretizer,
               use_core_state: bool = True,
               LossFn = ve_losses.ValueEquivalentLoss,
               get_unroll_params: Callable[[networks_lib.Params], networks_lib.Params] = lambda params: params.unroll,
               get_model_params: Callable[[networks_lib.Params], networks_lib.Params] = lambda params: params.model,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    """Initializes the learner."""
    self._config = config
    simulation_steps = config.simulation_steps
    num_simulations = config.num_simulations
    td_steps = config.td_steps
    policy_coef = config.policy_coef
    value_coef = config.value_coef
    reward_coef = config.reward_coef
    ema_update = config.ema_update
    show_gradients = config.show_gradients > 0

    assert config.muzero_policy in ["muzero", "gumbel_muzero"]
    if config.muzero_policy == "muzero":
      muzero_policy = functools.partial(
          mctx.muzero_policy,
          dirichlet_fraction=config.dirichlet_fraction,
          dirichlet_alpha=config.dirichlet_alpha,
          pb_c_init=config.pb_c_init,
          pb_c_base=config.pb_c_base,
          temperature=config.temperature)
      policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)
    elif config.muzero_policy == "gumbel_muzero":
      muzero_policy = functools.partial(
          mctx.gumbel_muzero_policy,
          gumbel_scale=config.gumbel_scale)
      # def kl_divergence(p, l):
        # return distrax.Categorical(probs=p).kl_divergence(distrax.Categorical(logits=l))
        # return distrax.Categorical(logits=l).kl_divergence(distrax.Categorical(probs=p))
      # policy_loss_fn = jax.vmap(kl_divergence)
      policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)
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
        _, online_state = networks.unroll(
          get_unroll_params(params), key1, burn_obs, online_state)
        _, target_state = networks.unroll(
          get_unroll_params(target_params), key2, burn_obs, target_state)

      #####################
      # Compute state + quantities for rest of trajectory
      #####################
      # Only get data to learn on from after the end of the burn in period.
      data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

      # Unroll on sequences to get online and target Q-Values.
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      online_outputs, online_state = networks.unroll(
        get_unroll_params(params), key1, data.observation, online_state)
      target_outputs, target_state = networks.unroll(
        get_unroll_params(target_params), key2, data.observation, target_state)

      ve_loss_fn = LossFn(
        networks=networks,
        params=params,
        target_params=target_params,
        simulation_steps=simulation_steps,
        discretizer=discretizer,
        num_simulations=num_simulations,
        discount=discount,
        td_steps=td_steps,
        muzero_policy=muzero_policy,
        model_coef=config.model_coef,
        policy_coef=policy_coef,
        value_coef=value_coef,
        reward_coef=reward_coef,
        policy_loss_fn=policy_loss_fn,
        get_model_params=get_model_params,
        v_target_source=config.v_target_source,
        metrics=config.metrics,
      )
      ve_loss_fn = jax.vmap(ve_loss_fn, in_axes=(1, 1, 1, 1, 1, 0, 0, None), out_axes=0) # vmap over batch dimension

      # [T, B]
      B = data.discount.shape[1]
      in_episode = jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
      is_terminal_mask = jnp.concatenate(
        (jnp.ones((1, B)), data.discount[:-1]), axis=0) == 0.0
      
      # [B], B
      batch_loss, metrics = ve_loss_fn(data,
                                       in_episode,
                                       is_terminal_mask,
                                       online_outputs,
                                       target_outputs,
                                       online_state,
                                       target_state,
                                       key_grad)


      if importance_sampling_exponent == 0.0:
        priorities = jnp.ones_like(batch_loss)
        mean_loss = jnp.mean(batch_loss)
      else:
        # Importance weighting.
        probs = sample.info.probability
        importance_weights = (1. / (probs + 1e-6)).astype(batch_loss.dtype)
        importance_weights **= importance_sampling_exponent
        importance_weights /= jnp.max(importance_weights)
        mean_loss = jnp.mean(importance_weights * batch_loss)

        # Calculate priorities as a mixture of max and mean sequence errors.
        value_target = metrics['2.value_target']
        value_prediction = metrics['2.value_prediction']
        batch_td_error = value_target - value_prediction

        abs_td_error = jnp.abs(batch_td_error).astype(batch_loss.dtype)
        max_priority = max_priority_weight * jnp.max(abs_td_error, axis=0)
        mean_priority = (1 - max_priority_weight) * jnp.mean(abs_td_error, axis=0)
        priorities = (max_priority + mean_priority)

      metrics = jax.tree_map(lambda x: x.mean(), metrics)
      metrics.update(in_episode=in_episode.mean())
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
      updates, new_opt_state = optimizer.update(gradients, state.opt_state, state.params)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1
      if ema_update is not None and ema_update > 0:
        target_params = optax.incremental_update(new_params, state.target_params,
                                                 ema_update)
      else:
        target_params = optax.periodic_update(new_params, state.target_params,
                                              steps, self._target_update_period)

      new_state = TrainingState(
          params=new_params,
          target_params=target_params,
          opt_state=new_opt_state,
          steps=steps,
          random_key=key)

      metrics.update({
        "0.0.total_loss": loss_value,
        '0.grad_norm': optax.global_norm(gradients),
        '0.update_norm': optax.global_norm(updates),
        '0.param_norm': optax.global_norm(state.params),
      })
      if show_gradients:
        metrics['mean_grad'] = jax.tree_map(lambda x:x.mean(), gradients)

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

    ########################
    # Initialise and internalise training state (parameters/optimiser state).
    ########################
    random_key, key_init1, key_init2 = jax.random.split(random_key, 3)

    if config.seperate_model_nets:
      initial_params = muzero_types.MuZeroParams(
        unroll=networks.unroll_init(key_init1),
        model=networks.model_init(key_init2),
      )
      weight_decay_mask = muzero_types.MuZeroParams(
        unroll=hk.data_structures.map(
            lambda module_name, name, value: True if name == "w" else False,
            initial_params.unroll),
        model=hk.data_structures.map(
            lambda module_name, name, value: True if name == "w" else False,
            initial_params.model)
      )
      sizes = tree.map_structure(jnp.size, initial_params)
      unroll_size = sum(tree.flatten(sizes.unroll.values()))
      model_size = sum(tree.flatten(sizes.model.values()))
      total_size = unroll_size + model_size
      logging.info('Total number of unroll params: %.3g', unroll_size)
      logging.info('Total number of model params: %.3g', model_size)
      logging.info('Total params ALLTOGETHER: %.3g', total_size)
    else:
      initial_params = hk.data_structures.merge(
          networks.model_init(key_init2),
          networks.unroll_init(key_init1),
          )
      
      weight_decay_mask = hk.data_structures.map(
          lambda module_name, name, value: True if name == "w" else False,
          initial_params)
      sizes = tree.map_structure(jnp.size, initial_params)
      from pprint import pprint
      pprint(sizes)
      import ipdb; ipdb.set_trace()
      logging.info('Total number of params: %.3g',
                   sum(tree.flatten(sizes.values())))

    if config.warmup_steps > 0:
      learning_rate = optax.warmup_exponential_decay_schedule(
          init_value=0.0,
          peak_value=config.learning_rate,
          warmup_steps=config.warmup_steps,
          transition_steps=config.lr_transition_steps,
          decay_rate=config.learning_rate_decay,
          staircase=True,
      )
    else:
      learning_rate = config.learning_rate
    if config.weight_decay > 0.0:
      optimizer = optax.adamw(
          learning_rate=learning_rate,
          eps=config.adam_eps,
          weight_decay=config.weight_decay,
          mask=weight_decay_mask,
      )
    else:
      optimizer = optax.adam(learning_rate=learning_rate, eps=config.adam_eps)
    if config.max_grad_norm:
      optimizer = optax.chain(optax.clip_by_global_norm(
          config.max_grad_norm), optimizer)

    opt_state = optimizer.init(initial_params)



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

    if self._config.show_gradients:
      if self._state.steps[0] % self._config.show_gradients == 0:
        for k, v in metrics['mean_grad'].items():
          metrics['mean_grad'][k] = float(next(iter(v.values())))
      else:
        if 'mean_grad' in metrics:
          metrics.pop('mean_grad')

    # Update our counts and record it.
    time_elapsed = time.time() - start
    metrics['LearnerDuration'] = time_elapsed
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
