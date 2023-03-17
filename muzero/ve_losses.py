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

"""Implementation of value-equivalent model.

Built off of R2D2 since value-based recurrent method. This adds learning the MuZero model.
"""

from pprint import pprint

import functools
from functools import partial
from typing import Tuple, Callable, Union

import chex
from acme import types as acme_types
from acme.jax import networks as networks_lib
import distrax
import jax
import jax.numpy as jnp
from jax import jit
import rlax
import mctx

from muzero import types as muzero_types
from muzero.utils import (
    scale_gradient,
    Discretizer,
)

Array = acme_types.NestedArray
State = acme_types.NestedArray

def aprint(a): return pprint(jax.tree_map(lambda x: x.shape, a))

@partial(jit, static_argnums=(1,))
def rolling_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

def masked_mean(x, mask):
  z = jnp.multiply(x, mask)
  return (z.sum(0))/(mask.sum(0)+1e-5)

class ValueEquivalentLoss:

  def __init__(self,
               networks: muzero_types.MuZeroNetworks,
               params: muzero_types.MuZeroParams,
               target_params: muzero_types.MuZeroParams,
               discretizer: Discretizer,
               muzero_policy: Union[mctx.muzero_policy, mctx.gumbel_muzero_policy],
               simulation_steps: int,
               num_simulations: int,
               td_steps: int,
               policy_loss_fn: Callable[[Array, Array], Array],
               discount: float = 0.99,
               policy_coef: float = 1.0,
               value_coef: float = 1.0,
               reward_coef: float = 1.0,
               model_coef: float = 1.0,
               get_model_params: Callable[[networks_lib.Params], networks_lib.Params] = lambda params: params.model,
               v_target_source: str = 'return',
               metrics: str = 'sparse',
               ):
    self._networks = networks
    self._params = params
    self._target_params = target_params
    self._simulation_steps = simulation_steps
    self._discretizer = discretizer
    self._num_simulations = num_simulations
    self._discount = discount
    self._td_steps = td_steps
    self._muzero_policy = muzero_policy
    self._policy_coef = policy_coef
    self._model_coef = model_coef
    self._value_coef = value_coef
    self._reward_coef = reward_coef
    self._policy_loss_fn = policy_loss_fn
    self._v_target_source = v_target_source
    self._get_model_params = get_model_params
    self._metrics = metrics
    assert metrics in ('sparse', 'dense')
    assert v_target_source in ('mcts', 'return', 'q_learning')

  def __call__(self,
               data: acme_types.NestedArray,
               in_episode: Array,
               is_terminal_mask: Array,
               online_outputs: muzero_types.RootOutput,
               target_outputs: muzero_types.RootOutput,
               online_state: State,
               target_state: State,
               rng_key: networks_lib.PRNGKey,
               ):
    del online_state
    del target_state
    nsteps = data.reward.shape[0]  # [T]

    metrics = {}
    # [T], [T/T-1], [T]
    policy_probs_target, value_probs_target, reward_probs_target, target_metrics = self.compute_target(
      data=data,
      in_episode=in_episode,
      is_terminal_mask=is_terminal_mask,
      online_outputs=online_outputs,
      target_outputs=target_outputs,
      rng_key=rng_key,
    )
    metrics.update(target_metrics)

    ###############################
    # Root losses
    ###############################
    # [T/T-1]
    dim_return = len(value_probs_target)
    root_value_loss = jax.vmap(rlax.categorical_cross_entropy)(
        value_probs_target, online_outputs.value_logits[:dim_return])
    # []
    root_value_loss = masked_mean(root_value_loss, in_episode[:dim_return])

    # [T]
    root_policy_loss = self._policy_loss_fn(policy_probs_target, online_outputs.policy_logits)
    # []
    root_policy_loss = masked_mean(root_policy_loss, in_episode)

    ###############################
    # Model losses
    ###############################
    #------------
    # prepare targets for model predictions
    #------------
    # add dummy values to targets for out-of-bounds simulation predictions
    npreds = nsteps + self._simulation_steps
    num_actions = online_outputs.policy_logits.shape[-1]
    uniform_policy = jnp.ones((self._simulation_steps, num_actions)) / num_actions
    policy_model_target = jnp.concatenate(
        (policy_probs_target, uniform_policy))

    dummy_zeros = self._discretizer.scalar_to_probs(jnp.zeros(self._simulation_steps-1))
    reward_model_target = jnp.concatenate((reward_probs_target, dummy_zeros))

    if dim_return < nsteps:
      nz = self._simulation_steps+1
    else:
      nz = self._simulation_steps
    dummy_zeros = self._discretizer.scalar_to_probs(jnp.zeros(nz))
    value_model_target = jnp.concatenate((value_probs_target, dummy_zeros))

    # for every timestep t=0,...T,  we have predictions for t+1, ..., t+k where k = simulation_steps
    # use rolling window to create T x k prediction targets
    vmap_roll = jax.vmap(functools.partial(rolling_window, size=self._simulation_steps), 1,2)
    policy_model_target = vmap_roll(policy_model_target[1:])  # [T, k, actions]
    value_model_target = vmap_roll(value_model_target[1:])    # [T, k, bins]
    reward_model_target = vmap_roll(reward_model_target)      # [T, k, bins]

    #------------
    # get masks for losses
    #------------
    # if dim_return is LESS than number of predictions, then have extra target, so mask it
    extra_v = self._simulation_steps + int(dim_return < npreds)
    value_mask = jnp.concatenate((in_episode[1:dim_return], jnp.zeros(extra_v)))
    policy_mask = jnp.concatenate((in_episode[1:], jnp.zeros(self._simulation_steps)))
    reward_mask = jnp.concatenate((in_episode, jnp.zeros(self._simulation_steps-1)))

    #------------
    # get simulation actions
    #------------
    rng_key, action_key = jax.random.split(rng_key)
    # unroll_actions = all_actions[start_i:start_i+simulation_steps]
    random_actions = jax.random.choice(
        action_key, num_actions, data.action.shape, replace=True)
    # for time-steps at the end of an episode, generate random actions from the last state
    simulation_actions = jax.lax.select(
        is_terminal_mask, random_actions, data.action)
    # expand simulations to account for model at end
    simulation_actions = jnp.concatenate(
      (simulation_actions, 
       jnp.zeros(self._simulation_steps-1, dtype=simulation_actions.dtype))
    )
    simulation_actions = rolling_window(simulation_actions, self._simulation_steps)

    #------------
    # compute loss
    #------------
    model_loss_fn = jax.vmap(self.model_loss,
                             in_axes=(0,0,0,0,0,0,0,0,None), out_axes=0)

    rng_key, model_key = jax.random.split(rng_key)
    model_reward_loss, model_value_loss, model_policy_loss, reward_logits, model_metrics = model_loss_fn(
      online_outputs,      # [T]
      simulation_actions,  # [T]
      policy_model_target, # [T]
      value_model_target,  # [T]
      reward_model_target, # [T]
      rolling_window(reward_mask, self._simulation_steps),
      rolling_window(value_mask, self._simulation_steps),
      rolling_window(policy_mask, self._simulation_steps),
      model_key,
    )
    metrics.update(model_metrics)

    # all are []
    model_policy_loss = self._model_coef * \
        masked_mean(model_policy_loss, policy_mask[:nsteps])
    model_value_loss = self._model_coef * \
        masked_mean(model_value_loss, value_mask[:nsteps])
    policy_loss = root_policy_loss + model_policy_loss
    value_loss = root_value_loss + model_value_loss
    reward_loss = masked_mean(model_reward_loss, reward_mask[:nsteps])
    total_loss = (
        self._reward_coef * reward_loss +
        self._value_coef * value_loss + 
        self._policy_coef * policy_loss)
    
    if self._metrics == 'sparse':
      metrics = {
        '0.1.root_policy_loss': root_policy_loss, # T
        '0.1.model_policy_loss': model_policy_loss,  # T
        '0.2.model_reward_loss': reward_loss,  # T
        '0.3.root_value_loss': root_value_loss,  # T
        '0.3.model_value_loss': model_value_loss,  # T
      }
      return total_loss, metrics

    # metrics
    max_entropy = distrax.Categorical(probs=uniform_policy[0]).entropy()
    entropy_fn_p = jax.vmap(lambda p: distrax.Categorical(probs=p).entropy())
    policy_target_entropy = entropy_fn_p(policy_probs_target)/(1e-5+max_entropy)
    policy_target_entropy = masked_mean(policy_target_entropy, policy_mask[:nsteps])

    entropy_fn_l = jax.vmap(lambda l: distrax.Categorical(logits=l).entropy())
    policy_entropy = entropy_fn_l(online_outputs.policy_logits)/(1e-5+max_entropy)
    policy_entropy = masked_mean(policy_entropy, policy_mask[:nsteps])

    prediction_metrics = {
      '1.0.policy_entropy': policy_entropy,
      '1.0.policy_target_entropy': policy_target_entropy,
      '1.policy_logits': masked_mean(online_outputs.policy_logits.mean(-1), policy_mask[:nsteps]),
      '2.value_logits': masked_mean(online_outputs.value_logits.mean(-1), value_mask[:nsteps]),
      '3.reward_target': masked_mean(data.reward, in_episode),  # T
      '3.reward_prediction': masked_mean(self._discretizer.logits_to_scalar(reward_logits), reward_mask[:nsteps]),
    }
    metrics.update(prediction_metrics)

    metrics.update({
      '0.1.root_policy_loss': root_policy_loss, # T
      '0.3.root_value_loss': root_value_loss,  # T
    })

    return total_loss, metrics

  def compute_target(self,
                     data: Array,
                     is_terminal_mask: Array,
                     in_episode: Array,
                     online_outputs: muzero_types.RootOutput,
                     target_outputs: muzero_types.RootOutput,
                     rng_key: networks_lib.PRNGKey,
    ):
    #---------------
    # reward
    #---------------
    reward_target = data.reward
    reward_probs_target = self._discretizer.scalar_to_probs(reward_target)
    reward_probs_target = jax.lax.stop_gradient(reward_probs_target)
    #---------------
    # policy
    #---------------
    # for policy, need initial policy + value estimates for each time-step
    # these will be used by MCTS
    target_values = self._discretizer.logits_to_scalar(target_outputs.value_logits)
    # search_roots = jax.tree_map(lambda t: t[:num_target_steps], target_outputs)
    # target_values = jax.tree_map(lambda t: t[:num_target_steps], target_values)
    roots = mctx.RootFnOutput(prior_logits=target_outputs.policy_logits,
                              value=target_values,
                              embedding=target_outputs.state)

    # 1 step of policy improvement
    rng_key, improve_key = jax.random.split(rng_key)
    mcts_outputs = self._muzero_policy(
        params=self._get_model_params(self._target_params),
        rng_key=improve_key,
        root=roots,
        recurrent_fn=functools.partial(
            model_step,
            discount=jnp.full(target_values.shape, self._discount),
            networks=self._networks,
            discretizer=self._discretizer),
        num_simulations=self._num_simulations)

    # policy_target = action_probs(mcts_outputs.search_tree.summary().visit_counts)
    policy_target = mcts_outputs.action_weights
    num_actions = policy_target.shape[-1]
    uniform_policy = jnp.ones_like(policy_target) / num_actions
    random_policy_mask = jnp.broadcast_to(
        is_terminal_mask[:, None], policy_target.shape
    )
    policy_probs_target = jax.lax.select(
        random_policy_mask, uniform_policy, policy_target
    )
    policy_probs_target = jax.lax.stop_gradient(policy_probs_target)

    #---------------
    # Value
    #---------------
    # discounts = (1.0 - trajectory.is_last[1:]) * discount_factor
    discounts = (data.discount[:-1] * self._discount).astype(target_values.dtype)
    if self._v_target_source == 'mcts':
      returns = mcts_outputs.search_tree.summary().value
    elif self._v_target_source == 'return':
      returns = rlax.n_step_bootstrapped_returns(
          data.reward[:-1], discounts, target_values[1:], self._td_steps)

    elif self._v_target_source == "q_learning":
      # these will already have been scaled...
      target_q_t = target_outputs.q_value[1:]
      online_q_t = online_outputs.q_value[1:]
      max_action = jnp.argmax(online_q_t, -1)

      # final q-learning loss (same as rlax.transformed_n_step_q_learning)
      v_t = rlax.batched_index(target_q_t, max_action)
      returns = rlax.transformed_n_step_returns(
          self._discretizer._tx_pair, data.reward[:-1], discounts, v_t, self._td_steps)

    returns = jax.lax.stop_gradient(returns)

    # Value targets for the absorbing state and the states after are 0.
    dim_return = returns.shape[0]
    value_target = jax.lax.select(
        is_terminal_mask[:dim_return], jnp.zeros_like(returns), returns)
    value_probs_target = self._discretizer.scalar_to_probs(value_target)
    value_probs_target = jax.lax.stop_gradient(value_probs_target)

    value_metrics = {}
    if self._metrics == 'dense':
      value_metrics = {
        '2.value_prediction': self._discretizer.logits_to_scalar(online_outputs.value_logits[:-1]),  # T
        '2.value_target': value_target,  # T-1
        '2.raw_return': returns,  # T-1
      }
      value_metrics = jax.tree_map(
          lambda x: masked_mean(x, in_episode[:-1]), value_metrics)

    return policy_probs_target, value_probs_target, reward_probs_target, value_metrics

  def model_loss(self,
                 starting_outputs: Array,
                 simulation_actions: Array,
                 policy_probs_target: Array,
                 value_probs_target: Array,
                 reward_probs_target: Array,
                 reward_mask: Array,
                 value_mask: Array,
                 policy_mask: Array,
                 rng_key: networks_lib.PRNGKey,
                 ):
    # 2) unroll model and get predictions
    rng_key, model_key = jax.random.split(rng_key)
    state = starting_outputs.state
    model_output, _ = self._networks.unroll_model(
        self._get_model_params(self._params),
        model_key, state, simulation_actions,
    )
    _batch_categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)

    reward_loss = _batch_categorical_cross_entropy(
        reward_probs_target, model_output.reward_logits)
    reward_loss = masked_mean(reward_loss, reward_mask)

    value_loss = _batch_categorical_cross_entropy(
        value_probs_target, model_output.value_logits)
    value_loss = masked_mean(value_loss, value_mask)

    policy_loss = self._policy_loss_fn(policy_probs_target, model_output.policy_logits)
    policy_loss = masked_mean(policy_loss, policy_mask)
    metrics = {}
    if self._metrics != 'sparse':
      metrics = {
          '0.1.model_policy_loss': self._model_coef*policy_loss, # T
          '0.2.model_reward_loss': reward_loss, # T
          '0.3.model_value_loss': self._model_coef*value_loss, # T
          # [T, A] --> T
          '3.reward_logits': masked_mean(model_output.reward_logits.mean(-1), reward_mask),
      }

    return reward_loss, value_loss, policy_loss, model_output.reward_logits[0], metrics


def model_step(params: networks_lib.Params,
               rng_key: chex.Array,
               action: chex.Array,
               state: chex.Array,
               discount: chex.Array,
               networks: muzero_types.MuZeroNetworks,
               discretizer: Discretizer):
  """One simulation step in MCTS."""
  rng_key, model_key = jax.random.split(rng_key)
  model_output, next_state = networks.apply_model(
      params, model_key, state, action,
  )
  reward = discretizer.logits_to_scalar(model_output.reward_logits)
  value = discretizer.logits_to_scalar(model_output.value_logits)

  recurrent_fn_output = mctx.RecurrentFnOutput(
      reward=reward,
      discount=discount,
      prior_logits=model_output.policy_logits,
      value=value,
  )
  return recurrent_fn_output, next_state
