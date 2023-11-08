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
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable, TypeVar

import collections
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
from jax.example_libraries import optimizers as jax_optim
import optax
import reverb
import tree
import numpy as np

from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.agents.jax.dqn import learning_lib
from muzero import types as muzero_types
from muzero.config import MuZeroConfig
from muzero import ve_losses
from muzero import learner_logger

from pprint import pprint
aprint = lambda a: pprint(jax.tree_map(lambda x:x.shape, a))

_PMAP_AXIS_NAME = 'data'
# This type allows splitting a sample between the host and device, which avoids
# putting item keys (uint64) on device for the purposes of priority updating.
R2D2ReplaySample = utils.PrefetchingSplit

def sample_maybe(prob_: float, key):
  if prob_  > .999:
    sample = True
    is_conditional = False
  elif prob_ < 1e-4:
    sample = False
    is_conditional = False
  else:
    sample = distrax.Bernoulli(
        probs=prob_).sample(seed=key)
    is_conditional = True
  return sample, is_conditional

def flatten_dict(dictionary, parent_key='', separator='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, separator))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

def group_named_values(dictionary: dict, name_fn = None):
  if name_fn is None:
    def name_fn(key): return key.split("/")[0]

  new_dict = collections.defaultdict(list)
  for key, named_values in dictionary.items():
    name = name_fn(key)
    new_dict[name].append(sum(named_values.values()))

  return new_dict



def savi_clip_grads(grad_tree, max_norm: float, epsilon: float = 1e-6):
  """Gradient clipping with epsilon.

  Adapted from jax.example_libraries.optimizers.clip_grads.

  Args:
    grad_tree: ArrayTree of gradients.
    max_norm: A float, clip gradients above this maximum norm.
    epsilon: A float, offset to avoid division by zero.

  Returns:
    ArrayTree of clipped gradients.
  """
  norm = jax_optim.l2_norm(grad_tree)
  clip_coef = max_norm / (norm + epsilon)
  normalize = lambda g: jnp.where(clip_coef < 1., g * clip_coef, g)
  return jax.tree_map(normalize, grad_tree)

def muzero_clip_grads(grad_tree, max_norm: float):
  """
  Returns:
    ArrayTree of clipped gradients.
  """
  g_norm = optax.global_norm(grad_tree)
  trigger = jnp.squeeze(g_norm < max_norm)

  def clip_fn(t):
    return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

  return jax.tree_util.tree_map(clip_fn, grad_tree)


def split_clip_by_norm(norm1, norm2, muzero_params, grad_fn1, grad_fn2):
  def init_fn(params):
      del params
      return optax.ClipByGlobalNormState()

  def update_fn(updates, state, params=None):
      del params

      def clip_fn(module_name, name, grad):
        found_muzero_param = sum([p in module_name for p in muzero_params])
        if found_muzero_param:
          norm = norm1
          return grad_fn1(grad, norm)
        else:
          norm = norm2
          return grad_fn2(grad, norm)

      updates = hk.data_structures.map(clip_fn, updates)
      return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def param_sizes(params):

  sizes = group_named_values(
    dictionary=tree.map_structure(jnp.size, params))

  format = lambda number: f"{number:,}"
  return {k: format(sum(v)) for k,v in sizes.items()}





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


class FactoredMuZeroLearner(acme.Learner):
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
               num_sgd_steps_per_step: int = 1,
               use_core_state: bool = True,
               LossFn = ve_losses.ValueEquivalentLoss,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               visualization_logger: Optional[learner_logger.BaseLogger] = None,
               ):
    """Initializes the learner."""
    self._config = config
    ema_update = config.ema_update
    show_gradients = config.show_gradients > 0
    self._num_sgd_steps_per_step = num_sgd_steps_per_step
    self._model_unroll_share_params = True
    self._visualization_logger = visualization_logger

    def loss(
        params: muzero_types.MuZeroParams,
        target_params: muzero_types.MuZeroParams,
        key_grad: networks_lib.PRNGKey,
        sample: reverb.ReplaySample
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
      """Computes loss for a batch of sequences."""

      if self._model_unroll_share_params:
        params_unroll = params
        target_params_unroll = target_params
      else:
        params_unroll = params.unroll
        target_params_unroll = target_params.unroll

      if config.step_penalty > 0.0:
        data = data._replace(reward=data.reward - jnp.abs(config.step_penalty))

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
        online_state = networks.init_recurrent_state(
          params_unroll, initial_state_rng,
          batch_size//self._num_sgd_steps_per_step)
      target_state = online_state

      # Convert sample data to sequence-major format [T, B, ...].
      data = utils.batch_to_sequence(sample.data)
      # Maybe burn the core state in.
      if burn_in_length:
        burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
        key_grad, key1, key2 = jax.random.split(key_grad, 3)
        _, online_state = networks.unroll(
          params_unroll, key1, burn_obs, online_state)
        _, target_state = networks.unroll(
          target_params_unroll, key2, burn_obs, target_state)

      #####################
      # Compute state + quantities for rest of trajectory
      #####################
      # Only get data to learn on from after the end of the burn in period.
      data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

      # Unroll on sequences to get online and target Q-Values.
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      online_outputs, online_state = networks.unroll(
        params_unroll, key1, data.observation, online_state)
      target_outputs, target_state = networks.unroll(
        target_params_unroll, key2, data.observation, target_state)

      # computing this outside of loss is important because only 1 branch is executed.
      # if computed inside function, vmap would lead all branches to be executed

      key_grad, sample_key = jax.random.split(key_grad)
      learn_model, conditional_learn_model = sample_maybe(
        prob_=config.model_learn_prob,
        key=sample_key)

      key_grad, sample_key = jax.random.split(key_grad)
      reanalyze, _ = sample_maybe(
        prob_=config.reanalyze_ratio,
        key=sample_key)

      ve_loss_fn = LossFn(
        networks=networks,
        params=params,
        target_params=target_params,
        conditional_learn_model=conditional_learn_model,
      )
      # vmap over batch dimension
      ve_loss_fn = jax.vmap(ve_loss_fn, in_axes=(1, 1, 1, 1, 1, 0, 0, None, None, None, None), out_axes=0)

      # [T, B]
      B = data.discount.shape[1]
      in_episode = jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
      is_terminal_mask = jnp.concatenate(
        (jnp.ones((1, B)), data.discount[:-1]), axis=0) == 0.0

      # [B], B
      batch_loss, loss_metrics, visualize_metrics, value_target, value_pred = ve_loss_fn(
        data,
        in_episode,
        is_terminal_mask,
        online_outputs,
        target_outputs,
        online_state,
        target_state,
        online_outputs,
        key_grad,
        learn_model,
        reanalyze)
      metrics=dict(
        loss_metrics=loss_metrics,
        visualize_metrics=visualize_metrics,
      )
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
        nsteps = min(value_target.shape[1], value_pred.shape[1])

        # [B, T]
        batch_td_error = value_target[:, :nsteps] - value_pred[:, :nsteps]

        abs_td_error = jnp.abs(batch_td_error).astype(batch_loss.dtype)
        max_priority = max_priority_weight * jnp.max(abs_td_error, axis=1)
        mean_priority = (1 - max_priority_weight) * jnp.mean(abs_td_error, axis=1)
        priorities = (max_priority + mean_priority)
      
      extra = learning_lib.LossExtra(metrics=metrics,
                                     reverb_priorities=priorities)

      return mean_loss, extra

    def evaluate_model(
        params: muzero_types.MuZeroParams,
        target_params: muzero_types.MuZeroParams,
        key_grad: networks_lib.PRNGKey,
        sample: reverb.ReplaySample):
      if self._model_unroll_share_params:
        params_unroll = params
        target_params_unroll = target_params
      else:
        params_unroll = params.unroll
        target_params_unroll = target_params.unroll

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
        online_state = networks.init_recurrent_state(
          params_unroll, initial_state_rng,
          batch_size//self._num_sgd_steps_per_step)
      target_state = online_state

      # Convert sample data to sequence-major format [T, B, ...].
      data = utils.batch_to_sequence(sample.data)
      # Maybe burn the core state in.
      if burn_in_length:
        burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
        key_grad, key1, key2 = jax.random.split(key_grad, 3)
        _, online_state = networks.unroll(
          params_unroll, key1, burn_obs, online_state)
        _, target_state = networks.unroll(
          target_params_unroll, key2, burn_obs, target_state)

      #####################
      # Compute state + quantities for rest of trajectory
      #####################
      # Only get data to learn on from after the end of the burn in period.
      data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

      # Unroll on sequences to get online and target Q-Values.
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      online_outputs, online_state = networks.unroll(
        params_unroll, key1, data.observation, online_state)
      target_outputs, target_state = networks.unroll(
        target_params_unroll, key2, data.observation, target_state)

      ve_loss_fn = LossFn(
        networks=networks,
        params=params,
        target_params=target_params,
      )
      evaluate_model = ve_loss_fn.evaluate_model
      # vmap over batch dimension
      evaluate_model = jax.vmap(evaluate_model,
                                in_axes=(1, 1, 1, None), out_axes=0)

      return evaluate_model(
        data,
        online_outputs,
        target_outputs,
        key_grad)

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
      metrics['loss_metrics']['0.0.final_loss'] = loss_value
      metrics['loss_metrics'] = jax.tree_map(lambda x: x.mean(), metrics['loss_metrics'])
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

      metrics['loss_metrics'].update({
        '0.grad_norm': optax.global_norm(gradients),
        '0.update_norm': optax.global_norm(updates),
        '0.param_norm': optax.global_norm(state.params),
      })
      if show_gradients:
        metrics['loss_metrics']['mean_grad'] = jax.tree_map(lambda x:x.mean(), gradients)

      return new_state, (priorities, metrics)

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

    if self._num_sgd_steps_per_step > 1:
      sgd_step = utils.process_multiple_batches(
        sgd_step,
        num_sgd_steps_per_step,
        # get aux from 1st batch
        postprocess_aux=lambda aux: jax.tree_map(lambda x: x[0], aux))

    self._loss = jax.pmap(loss,
                          axis_name=_PMAP_AXIS_NAME)
    self._sgd_step = jax.pmap(sgd_step,
                              axis_name=_PMAP_AXIS_NAME)
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

    self._evaluate_model = jax.pmap(evaluate_model,
                                    axis_name=_PMAP_AXIS_NAME)

    ########################
    # Initialise and internalise training state (parameters/optimiser state).
    ########################
    random_key, key_init1, key_init2 = jax.random.split(random_key, 3)

    if self._model_unroll_share_params:
      initial_params = networks.unroll_init(key_init1)
      default_decay = lambda module_name, name, value: True if name == "w" else False
      def decay_except_savi(module_name, name, value):
        del value
        if "slot_attention" in module_name: return False
        elif "observation_fn" in module_name: return False
        elif name == "w": return True
        return False
      if config.weight_decay_fn == "default":
        decay_fn = default_decay
      elif config.weight_decay_fn == "except_savi":
        decay_fn = decay_except_savi
      weight_decay_mask = hk.data_structures.map(decay_fn, initial_params)

      if config.warmup_steps > 0 or config.lr_transition_steps > 0:
        warmup_steps = config.warmup_steps
        if config.warmup_steps == 0:
          warmup_steps = 1
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            end_value=config.lr_end_value,
            transition_steps=config.lr_transition_steps,
            decay_rate=config.learning_rate_decay,
            staircase=config.staircase_decay,
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
        muzero_params = ('pred')
        if config.muzero_grad_model:
          muzero_params += ('model')

        if config.grad_fn == 'muzero':
          grad_fn1=muzero_clip_grads
          grad_fn2=muzero_clip_grads
        elif config.grad_fn == 'savi':
          grad_fn1=savi_clip_grads
          grad_fn2=savi_clip_grads
        elif config.grad_fn == 'muzero_savi':
          grad_fn1 = muzero_clip_grads
          grad_fn2 = savi_clip_grads
        else:
          raise NotImplementedError

        clip_grad_op = split_clip_by_norm(
          config.max_grad_norm, config.savi_grad_norm,
          muzero_params,
          grad_fn1=grad_fn1,
          grad_fn2=grad_fn2)
        optimizer = optax.chain(clip_grad_op, optimizer)

      self._learning_rate = learning_rate
      opt_state = optimizer.init(initial_params)
  
      # Log how many parameters the network has.
      sizes = tree.map_structure(jnp.size, initial_params)
      total_params =  sum(tree.flatten(sizes.values()))
      logging.info('Total number of params: %.3g', total_params)
      print(f"{total_params:,}")
      pprint(param_sizes(initial_params))
    else:
      raise NotImplementedError

    state = TrainingState(
        params=initial_params,
        target_params=initial_params,
        opt_state=opt_state,
        steps=jnp.array(0),
        random_key=random_key)
    # Replicate parameters.
    self._state = utils.replicate_in_all_devices(state)

  def evaluate_model(self,
                   sample: reverb.ReplaySample,
                   loss_key: networks_lib.PRNGKey):

    # key, key_grad = jax.random.split(state.random_key)
    results = self._evaluate_model(
      params=self._state.params, 
      target_params=self._state.target_params,
      key_grad=utils.replicate_in_all_devices(loss_key),
      sample=sample)
    return jax.tree_map(lambda x:x.mean(), results)

  def compute_loss(self,
                   sample: reverb.ReplaySample,
                   loss_key: networks_lib.PRNGKey,
                   vis_label: str = None,
                   update_visualizations: bool = False):

    # key, key_grad = jax.random.split(state.random_key)
    loss_value, extra = self._loss(
      params=self._state.params, 
      target_params=self._state.target_params,
      key_grad=utils.replicate_in_all_devices(loss_key),
      sample=sample)
    metrics = extra.metrics
    # Take metrics from first replica.
    metrics = utils.get_from_first_device(metrics)

    if update_visualizations:
      self._visualization_logger.log_metrics(metrics, label=vis_label)
    loss_metrics = metrics.pop('loss_metrics', {})
    loss_metrics = jax.tree_map(lambda x: x.mean(), loss_metrics)

    return loss_value, loss_metrics

  def step(self, vis_label: str = None):
    prefetching_split = next(self._iterator)
    # The split_sample method passed to utils.sharded_prefetch specifies what
    # parts of the objects returned by the original iterator are kept in the
    # host and what parts are prefetched on-device.
    # In this case the host property of the prefetching split contains only the
    # replay keys and the device property is the prefetched full original
    # sample.
    if hasattr(prefetching_split, 'host'):
      keys = prefetching_split.host
      samples: reverb.ReplaySample = prefetching_split.device
    else:
      samples: reverb.ReplaySample = prefetching_split
    # Do a batch of SGD.
    start = time.time()
    self._state, (priorities, metrics) = self._sgd_step(self._state, samples)
    # Take metrics from first replica.
    metrics = utils.get_from_first_device(metrics)

    ###############################
    # log learning rate
    ###############################
    if callable(self._learning_rate):
      count = self._state.opt_state[1][2].count[0]
      lr = self._learning_rate(count)
    else:
      lr = self.learning_rate
    metrics['loss_metrics']['z.learning_rate'] = np.array(lr)

    ###############################
    # checking for nans
    ###############################
    if self._config.check_nan:
      def nan_runtime(x):
        if jnp.isnan(x): raise ValueError
      means = jax.tree_map(lambda x: x.mean(), metrics)

      try:
        jax.tree_map(lambda x: nan_runtime(x), means)
      except ValueError as e:
        logging.warning("Nans found")
        means = flatten_dict(means)
        for k, v in means.items():
          if jnp.isscalar(v) and jnp.isnan(v):
            print(k, v)
        raise RuntimeError("exiting")

    ###############################
    # updating
    ###############################
    self._visualization_logger.log_metrics(metrics, label=vis_label)
    self._visualization_logger.step()
    loss_metrics = metrics.pop('loss_metrics', {})
    loss_metrics = jax.tree_map(lambda x: x.mean(), loss_metrics)

    if self._config.show_gradients:
      hit_show_gradients = self._state.steps[0] % self._config.show_gradients == 0
      grad_norm = loss_metrics['0.grad_norm']
      if hit_show_gradients or grad_norm.mean() > 40:
        flat_mean_grads = None
        if grad_norm.mean() > 40:
          flat_mean_grads = flatten_dict(loss_metrics['mean_grad'])
        mean_grad = group_named_values(
            dictionary=loss_metrics['mean_grad'])

        new_mean_grad = {}
        for k, v in mean_grad.items():
          new_norm = float(np.linalg.norm(v, ord=1))
          new_mean_grad[f'{k}_norm'] = new_norm
          if flat_mean_grads and new_norm > 40:
            for individual_key, individual_norm in flat_mean_grads.items():
              if k in individual_key and '.w' in individual_key:  # i.e. weights, not bias
                new_mean_grad[f'{individual_key}_norm'] = individual_norm

        loss_metrics['mean_grad'] = new_mean_grad

      else:
        if 'mean_grad' in loss_metrics:
          loss_metrics.pop('mean_grad')

    # Update our counts and record it.
    time_elapsed = time.time() - start
    loss_metrics['Learnerduration'] = time_elapsed
    steps_per_sec = (self._num_sgd_steps_per_step / time_elapsed) if time_elapsed else 0
    loss_metrics['steps_per_second'] = steps_per_sec
    counts = self._counter.increment(steps=self._num_sgd_steps_per_step,
                                     time_elapsed=time_elapsed)

    # Update priorities in replay.
    if self._replay_client:
      self._async_priority_updater.put((keys, priorities))

    # Attempt to write logs.
    self._logger.write({**loss_metrics, **counts})

  def get_variables(self, names: List[str]) -> List[muzero_types.MuZeroParams]:
    del names  # There's only one available set of params in this agent.
    # Return first replica of parameters.
    return utils.get_from_first_device([self._state.params])

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return utils.get_from_first_device(self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state)
