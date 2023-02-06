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

from pprint import pprint
import dataclasses

import functools
import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

from absl import logging
import acme
import chex
from acme import types as acme_types
from acme.adders import reverb as adders
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
import distrax
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree
import mctx

from acme.agents.jax.r2d2 import networks as r2d2_networks
from muzero.config import MuZeroConfig
from muzero import types as muzero_types
from muzero.utils import (
    inv_value_transform,
    logits_to_scalar,
    scalar_to_two_hot,
    scale_gradient,
    value_transform,
)

Array = acme_types.NestedArray


def aprint(a): return pprint(jax.tree_map(lambda x: x.shape, a))

def muzero_loss(data: acme_types.NestedArray,
                online_outputs: muzero_types.RootOutput,
                target_outputs: muzero_types.RootOutput,
                online_state: muzero_types.MuZeroState,
                target_state: muzero_types.MuZeroState,
                networks: muzero_types.MuZeroNetworks,
                params: muzero_types.MuZeroParams,
                target_params: muzero_types.MuZeroParams,
                rng_key: networks_lib.PRNGKey,
                simulation_steps: int,
                num_bins: int,
                num_simulations: int,
                td_steps: int,
                maxvisit_init: int = 50,
                gumbel_scale: float = 1.0,
                discount: float = 0.99,
                policy_coef: float = 1.0,
                value_coef: float = 1.0,
                reward_coef: float = 1.0,
                model_state_extract_fn: Callable[[acme_types.NestedArray],
                                                 jnp.ndarray] = lambda state: state.hidden,
                ):
  """MuZero loss. Assumes data only has time-dimension, not batch-dimension."""

  #####################
  # Unroll model with action sequence from data
  # - outputs:
  #   - policy, value predictions for t = 0, ..., T (root + model)
  #   - reward for t = 1, ..., T (model)
  #####################
  # 1) compute value + policy at root
  # root_state = jax.tree_map(lambda t: t[:1], online_outputs.state)
  learner_root = jax.tree_map(lambda t: t[:1], online_outputs)
  # reward = logits_to_scalar(learner_root.reward_logits, num_bins)
  # reward = inv_value_transform(reward)
  # learner_root = compute_root_values(networks, params, num_bins, root_state)
  # learner_root: AgentOutput = jax.tree_map(lambda t: t[0], learner_root)

  # 2) compute actions that will be used for simulation
  unroll_data = jax.tree_map(lambda t: t[: simulation_steps + 1], data)
  random_action_mask = (
      jnp.cumprod(1.0 - unroll_data.start_of_episode[1:]) == 0.0
  )
  action_sequence = unroll_data.action[:simulation_steps]
  num_actions = learner_root.policy_logits.shape[-1]
  rng_key, action_key = jax.random.split(rng_key)
  random_actions = jax.random.choice(
      action_key, num_actions, action_sequence.shape, replace=True
  )
  simulation_actions = jax.lax.select(
      random_action_mask, random_actions, action_sequence
  )

  # 2) unroll model and get predictions
  _, model_output = model_unroll(
      networks=networks,
      params=params,
      rng_key=rng_key,
      state=model_state_extract_fn(online_state),
      action_sequence=simulation_actions,
  )

  #####################
  # construct targets
  #####################
  # 2) Model learning targets.
  # prepare values
  target_values = logits_to_scalar(target_outputs.value_logits, num_bins)
  target_values = inv_value_transform(target_values)

  #---------------
  # reward
  #---------------
  rewards = data.reward
  reward_target = jax.lax.select(
      random_action_mask,
      jnp.zeros_like(rewards[:simulation_steps]),
      rewards[:simulation_steps],
  )
  reward_target_transformed = value_transform(reward_target)
  reward_logits_target = scalar_to_two_hot(
      reward_target_transformed, num_bins
  )
  #---------------
  # policy
  #---------------
  # for policy, need initial policy + value estimates for each time-step
  # these will be used by MCTS
  search_roots = jax.tree_map(lambda t: t[:simulation_steps+1], target_outputs)
  root_values = jax.tree_map(lambda t: t[:simulation_steps+1], target_values)

  rng_key, improve_key = jax.random.split(rng_key)
  roots = mctx.RootFnOutput(prior_logits=search_roots.policy_logits,
                            value=root_values,
                            embedding=search_roots.state)
  
  # 1 step of policy improvement
  search_data = jax.tree_map(lambda t: t[:simulation_steps+1], data)
  mcts_outputs = mctx.gumbel_muzero_policy(
      params=params,
      rng_key=rng_key,
      root=roots,
      recurrent_fn=functools.partial(
          model_step,
          discount=(search_data.discount * discount).astype(root_values.dtype),
          networks=networks,
          num_bins=num_bins,
          discount_factor=discount),
      num_simulations=num_simulations,
      # invalid_actions=jax.vmap(lambda e: e.invalid_actions())(env),
      qtransform=functools.partial(
          mctx.qtransform_completed_by_mix_value,
          value_scale=0.1,
          maxvisit_init=maxvisit_init,
          rescale_values=True,
      ),
      gumbel_scale=gumbel_scale,
  )
  # policy_target = action_probs(mcts_outputs.search_tree.summary().visit_counts)
  policy_target = mcts_outputs.action_weights
  uniform_policy = jnp.ones_like(policy_target) / num_actions
  is_terminal_mask = jnp.cumprod(unroll_data.discount) == 0.0
  random_policy_mask = jnp.broadcast_to(
      is_terminal_mask[:, None], policy_target.shape
  )
  policy_target = jax.lax.select(
      random_policy_mask, uniform_policy, policy_target
  )
  policy_target = jax.lax.stop_gradient(policy_target)

  #---------------
  # Value
  #---------------
  # discounts = (1.0 - trajectory.is_last[1:]) * discount_factor
  discounts = (data.discount[1:] * discount).astype(root_values.dtype)
  v_bootstrap = target_values

  def n_step_return(i: int) -> jnp.ndarray:
      bootstrap_value = jax.tree_map(lambda t: t[i + td_steps], v_bootstrap)
      _rewards = jnp.concatenate(
          [rewards[i : i + td_steps], bootstrap_value[None]], axis=0
      )
      _discounts = jnp.concatenate(
          [jnp.ones((1,)), jnp.cumprod(discounts[i : i + td_steps])],
          axis=0,
      )
      return jnp.sum(_rewards * _discounts)

  returns = []
  for i in range(simulation_steps + 1):
      returns.append(n_step_return(i))
  returns = jnp.stack(returns)
  # Value targets for the absorbing state and the states after are 0.
  zero_return_mask = is_terminal_mask
  value_target = jax.lax.select(
      zero_return_mask, jnp.zeros_like(returns), returns
  )
  value_target_transformed = value_transform(value_target)
  value_logits_target = scalar_to_two_hot(value_target_transformed, num_bins)
  value_logits_target = jax.lax.stop_gradient(value_logits_target)

  #####################
  # Compute losses
  #####################
  # reward
  _batch_categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)
  reward_loss = _batch_categorical_cross_entropy(
      reward_logits_target, model_output.reward_logits)
  reward_loss = jnp.concatenate((jnp.zeros(1), reward_loss))

  # value
  value_logits = jnp.concatenate((learner_root.value_logits, model_output.value_logits), axis=0)
  value_loss = _batch_categorical_cross_entropy(value_logits_target, value_logits)

  policy_logits = jnp.concatenate([learner_root.policy_logits, model_output.policy_logits], axis=0)
  policy_loss = _batch_categorical_cross_entropy(policy_target, policy_logits)
  
  total_loss = (
    reward_coef * reward_loss +
    value_coef * value_loss + 
    policy_coef * policy_loss)
  
  # metrics
  entropy_fn = lambda l: distrax.Categorical(logits=l).entropy()
  entropy_fn = jax.vmap(entropy_fn)
  policy_target_entropy = entropy_fn(policy_target)
  policy_entropy = entropy_fn(policy_logits)

  # 3) organize predictions
  # [0, ...]
  value_preds = logits_to_scalar(value_logits, num_bins)
  value_preds = inv_value_transform(value_preds)

  # [1, ...]
  reward_preds = logits_to_scalar(model_output.reward_logits, num_bins)
  reward_preds = inv_value_transform(reward_preds)

  # reward_target = jnp.concatenate((jnp.zeros(1), reward_target))
  # reward_preds = jnp.concatenate((jnp.zeros(1), reward_preds))

  metrics = {
      'reward_target': reward_target,
      'reward_prediction': reward_preds,
      'value_target': value_target,
      'value_prediction': value_preds,
      'policy_entropy': policy_entropy,
      'policy_target_entropy': policy_target_entropy,
      'reward_loss': reward_loss,
      'value_loss': value_loss,
      'policy_loss': policy_loss,
      'total_loss': total_loss,
  }

  return total_loss, metrics


def model_unroll(
    networks: muzero_types.MuZeroNetworks,
    params: muzero_types.Params,
    rng_key: networks_lib.PRNGKey,
    # num_bins: int,
    state: Array,
    action_sequence: Array,
) -> muzero_types.ModelOutput:
    """Unroll the learned model with a sequence of actions."""

    def fn(state: Array,
           action: Array,
           rng_key: networks_lib.PRNGKey) -> Tuple[Array, Array]:
        """Dynamics fun for scan."""
        rng_key, model_key = jax.random.split(rng_key)
        model_outputs, next_state = networks.apply_model(
            params.model, model_key, state, action,
        )
        next_state = scale_gradient(next_state, 0.5)
        return next_state, model_outputs

    fn = functools.partial(fn, rng_key=rng_key)
    return jax.lax.scan(fn, state, action_sequence)


def model_step(params: muzero_types.MuZeroParams,
               rng_key: chex.Array,
               action: chex.Array,
               state: chex.Array,
               discount: chex.Array,
               networks: muzero_types.MuZeroNetworks,
               num_bins: int,
               discount_factor: float = .99):
  """One simulation step in MCTS."""
  rng_key, model_key = jax.random.split(rng_key)
  model_output, next_state = networks.apply_model(
      params.model, model_key, state, action,
  )
  reward = logits_to_scalar(model_output.reward_logits, num_bins)
  reward = inv_value_transform(reward)

  value = logits_to_scalar(model_output.value_logits, num_bins)
  value = inv_value_transform(value)

  recurrent_fn_output = mctx.RecurrentFnOutput(
      reward=reward,
      discount=discount*discount_factor,
      prior_logits=model_output.policy_logits,
      value=value,
  )
  return recurrent_fn_output, next_state
