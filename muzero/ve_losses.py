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
from typing import Tuple, Callable, Union, Optional

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
from muzero.utils import Discretizer
from muzero.learner_logger import LearnerLogger

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
               root_policy_coef: float = 1.0,
               root_value_coef: float = 1.0,
               model_policy_coef: float = 1.0,
               model_value_coef: float = 1.0,
               model_reward_coef: float = 1.0,
               conditional_learn_model: bool = False,
               mask_model: bool = False,
               v_target_source: str = 'returns',
               invalid_actions: Optional[chex.Array] = None,
               behavior_clone: bool = False,
               **kwargs,
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
    self._root_policy_coef = root_policy_coef
    self._root_value_coef = root_value_coef
    self._model_policy_coef = model_policy_coef
    self._model_value_coef = model_value_coef
    self._model_reward_coef = model_reward_coef
    self._policy_loss_fn = policy_loss_fn
    self._v_target_source = v_target_source
    self._conditional_learn_model = conditional_learn_model
    self._mask_model = mask_model
    self._invalid_actions = invalid_actions
    self._behavior_clone = behavior_clone

    assert v_target_source in ('mcts',
                               'return',
                               'q_learning',
                               'eff_zero',  # efficient-zero strategy
                               'reanalyze'), v_target_source  # original re-analyze strategy

  def __call__(self,
               data: acme_types.NestedArray,
               in_episode: Array,
               is_terminal_mask: Array,
               online_outputs: muzero_types.RootOutput,
               target_outputs: muzero_types.RootOutput,
               online_state: State,
               target_state: State,
               rng_key: networks_lib.PRNGKey,
               learn_model: bool = True,
               reanalyze: bool = True,
               logger: Optional[LearnerLogger] = None,
               ):

    # applying this ensures no gradients are computed
    def stop_grad_pytree(pytree):
      return jax.tree_map(lambda x: jax.lax.stop_gradient(x), pytree)

    # [T], [T/T-1], [T]
    policy_probs_target, value_probs_target, reward_probs_target, returns, mcts_values = self.compute_target(
      data=stop_grad_pytree(data),
      in_episode=stop_grad_pytree(in_episode),
      is_terminal_mask=stop_grad_pytree(is_terminal_mask),
      online_outputs=stop_grad_pytree(online_outputs),
      target_outputs=stop_grad_pytree(target_outputs),
      rng_key=rng_key,
      reanalyze=reanalyze,
    )


    learn_fn = functools.partial(
      self.learn,
      data=data,
      in_episode=in_episode,
      is_terminal_mask=is_terminal_mask,
      online_outputs=online_outputs,
      target_outputs=target_outputs,
      online_state=online_state,
      target_state=target_state,
      rng_key=rng_key,
      policy_probs_target=policy_probs_target,
      value_probs_target=value_probs_target,
      reward_probs_target=reward_probs_target,
      returns=returns,
      mcts_values=mcts_values,
      logger=logger,
    )

    if self._conditional_learn_model:
      return jax.lax.cond(
          learn_model,
        lambda: learn_fn(learn_model=True),
        lambda: learn_fn(learn_model=False))
    else:
      return learn_fn(learn_model=learn_model)

  def get_invalid_actions(self, batch_size):
    invalid_actions = self._invalid_actions
    if invalid_actions is None:
      return None
    if invalid_actions.ndim < 2:
      invalid_actions = jax.numpy.tile(
          invalid_actions, (batch_size, 1))
    return invalid_actions

  def compute_target(self,
                     data: Array,
                     is_terminal_mask: Array,
                     in_episode: Array,
                     online_outputs: muzero_types.RootOutput,
                     target_outputs: muzero_types.RootOutput,
                     rng_key: networks_lib.PRNGKey,
                     reanalyze: bool = True,
                     ):
    # target_metrics = {}
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
    target_values = self._discretizer.logits_to_scalar(
        target_outputs.value_logits)
    roots = mctx.RootFnOutput(prior_logits=target_outputs.policy_logits,
                              value=target_values,
                              embedding=target_outputs.state)

    invalid_actions = self.get_invalid_actions(
      batch_size=target_values.shape[0])

    # 1 step of policy improvement
    rng_key, improve_key = jax.random.split(rng_key)
    mcts_outputs = self._muzero_policy(
        params=self._target_params,
        rng_key=improve_key,
        root=roots,
        invalid_actions=invalid_actions,
        recurrent_fn=functools.partial(
            model_step,
            discount=jnp.full(target_values.shape, self._discount),
            networks=self._networks,
            discretizer=self._discretizer,
            ),
        num_simulations=self._num_simulations)

    # policy_target = action_probs(mcts_outputs.search_tree.summary().visit_counts)
    num_actions = target_outputs.policy_logits.shape[-1]
    if self._behavior_clone:
      policy_target = jax.nn.one_hot(data.action, num_classes=num_actions)
    else:
      policy_target = mcts_outputs.action_weights

    uniform_policy = jnp.ones_like(policy_target) / num_actions
    if invalid_actions is not None:
      valid_actions = 1 - invalid_actions
      uniform_policy = valid_actions/valid_actions.sum(-1, keepdims=True)

    random_policy_mask = jnp.broadcast_to(
        is_terminal_mask[:, None], policy_target.shape
    )
    policy_probs_target = jax.lax.select(
        random_policy_mask, uniform_policy, policy_target
    )
    policy_probs_target = jax.lax.stop_gradient(policy_probs_target)

    #---------------
    # Values
    #---------------
    discounts = (data.discount[:-1] *
                 self._discount).astype(target_values.dtype)
    mcts_values = mcts_outputs.search_tree.summary().value
    if self._v_target_source == 'mcts':
      returns = mcts_values
    elif self._v_target_source == 'return':
      returns = rlax.n_step_bootstrapped_returns(
          data.reward[:-1], discounts, target_values[1:], self._td_steps)

    elif self._v_target_source == "q_learning":
      # these will already have been scaled...
      raise NotImplementedError
      # target_q_t = target_outputs.q_value[1:]
      # online_q_t = online_outputs.q_value[1:]
      # max_action = jnp.argmax(online_q_t, -1)

      # # final q-learning loss (same as rlax.transformed_n_step_q_learning)
      # v_t = rlax.batched_index(target_q_t, max_action)
      # returns = rlax.transformed_n_step_returns(
      #     self._discretizer._tx_pair, data.reward[:-1], discounts, v_t, self._td_steps)
    elif self._v_target_source == 'eff_zero':
      def return_fn(v): return rlax.n_step_bootstrapped_returns(
          data.reward[:-1], discounts, v[1:], self._td_steps)
      returns = jax.lax.cond(
          reanalyze > 0,
          lambda: return_fn(mcts_values),
          lambda: return_fn(target_values))
    elif self._v_target_source == 'reanalyze':
      def return_fn(v): return rlax.n_step_bootstrapped_returns(
          data.reward[:-1], discounts, v[1:], self._td_steps)
      dim_return = data.reward.shape[0] - 1
      returns = jax.lax.cond(
          reanalyze > 0,
          lambda: mcts_values[:dim_return],
          lambda: return_fn(target_values))

    # Value targets for the absorbing state and the states after are 0.
    dim_return = returns.shape[0]
    value_target = jax.lax.select(
        is_terminal_mask[:dim_return], jnp.zeros_like(returns), returns)
    value_probs_target = self._discretizer.scalar_to_probs(value_target)
    value_probs_target = jax.lax.stop_gradient(value_probs_target)

    return policy_probs_target, value_probs_target, reward_probs_target, returns, mcts_values

  def learn(self,
            data: acme_types.NestedArray,
            in_episode: Array,
            is_terminal_mask: Array,
            online_outputs: muzero_types.RootOutput,
            target_outputs: muzero_types.RootOutput,
            online_state: State,
            target_state: State,
            rng_key: networks_lib.PRNGKey,
            policy_probs_target: Array,
            value_probs_target: Array,
            reward_probs_target: Array,
            returns: Array,
            mcts_values: Array,
            learn_model: bool = True,
            logger: Optional[LearnerLogger] = None,
            ):
    del target_outputs
    del online_state
    del target_state
    nsteps = data.reward.shape[0]  # [T]
    loss_metrics = {}
    visualize_metrics = {}

    ###############################
    # Root losses
    ###############################
    # [T/T-1]
    dim_return = value_probs_target.shape[0]
    root_value_ce = jax.vmap(rlax.categorical_cross_entropy)(
        value_probs_target, online_outputs.value_logits[:dim_return])
    # []
    root_value_loss = masked_mean(root_value_ce, in_episode[:dim_return])
    root_value_loss = self._root_value_coef*root_value_loss


    # [T]
    root_policy_ce = self._policy_loss_fn(policy_probs_target, online_outputs.policy_logits)
    # []
    root_policy_loss = masked_mean(root_policy_ce, in_episode)
    root_policy_loss = self._root_policy_coef*root_policy_loss

    #------------
    # root metrics
    #------------
    value_root_prediction = self._discretizer.logits_to_scalar(
                online_outputs.value_logits[:dim_return])
    visualize_metrics['visualize_root_data'] = dict(
        # episode data
        data=data,  # after any processing
        is_terminal=is_terminal_mask,
        in_episode=in_episode,
        online_state=online_outputs.state,
        online_outputs=online_outputs,
        # root policy
        policy_root_ce=root_policy_ce,
        policy_root_mask=in_episode,
        policy_root_target=policy_probs_target,
        policy_root_logits=online_outputs.policy_logits,
        policy_root_prediction=jax.nn.softmax(online_outputs.policy_logits),
        # root value
        value_root_ce=root_value_ce,
        value_root_mask=in_episode[:dim_return],
        value_root_target=returns,
        value_root_target_probs=value_probs_target,
        value_root_logits=online_outputs.value_logits[:dim_return],
        value_root_prediction=value_root_prediction,
        # extras
        mcts_values=mcts_values[:dim_return],  # needs to caches
    )

    loss_metrics.update({
      '0.1.policy_root_loss': root_policy_loss, # T
      '0.3.value_root_loss': root_value_loss,  # T
    })

    if not learn_model:
      total_loss = root_value_loss + root_policy_loss
      return total_loss, loss_metrics, visualize_metrics, returns, mcts_values

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
    invalid_actions = self.get_invalid_actions(
      batch_size=self._simulation_steps)
    if invalid_actions is not None:
      valid_actions = 1 - invalid_actions
      uniform_policy = valid_actions/valid_actions.sum(-1, keepdims=True)

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
    policy_mask = jnp.concatenate((in_episode[1:], jnp.zeros(self._simulation_steps)))
    if self._mask_model:
      value_mask = jnp.concatenate((in_episode[1:dim_return], jnp.zeros(extra_v)))
      reward_mask = policy_mask
    else:
      reward_mask = value_mask = jnp.ones_like(policy_mask)
    reward_model_mask = rolling_window(reward_mask, self._simulation_steps)
    value_model_mask = rolling_window(value_mask, self._simulation_steps)
    policy_model_mask = rolling_window(policy_mask, self._simulation_steps)

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
    # model outputs
    #------------
    def model_unroll(key, state, actions):
      key, model_key = jax.random.split(key)
      model_output, _ = self._networks.unroll_model(
          self._params, model_key, state, actions)
      return model_output
    model_unroll = jax.vmap(model_unroll, in_axes=(0,0,0), out_axes=0)

    keys = jax.random.split(rng_key, num=len(policy_model_target)+1)
    model_keys = keys[1:]
    # T, |simulation_actions|, ...
    model_outputs = model_unroll(
        model_keys, online_outputs.state, simulation_actions,
    )

    #------------
    # compute loss
    #------------
    def compute_losses(
        model_outputs_,
        reward_target_, value_target_, policy_target_,
        reward_mask_, value_mask_, policy_mask_):
      _batch_categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)
      reward_ce = _batch_categorical_cross_entropy(
          reward_target_, model_outputs_.reward_logits)
      reward_loss = masked_mean(reward_ce, reward_mask_)

      value_ce = _batch_categorical_cross_entropy(
          value_target_, model_outputs_.value_logits)
      value_loss = masked_mean(value_ce, value_mask_)

      policy_ce = self._policy_loss_fn(
          policy_target_, model_outputs_.policy_logits)
      policy_loss = masked_mean(policy_ce, policy_mask_)

      return reward_ce, value_ce, policy_ce, reward_loss, value_loss, policy_loss

    _ = [
      reward_model_ce,
      value_model_ce,
      policy_model_ce,
      model_reward_loss,
      model_value_loss,
      model_policy_loss] = jax.vmap(compute_losses)(
        model_outputs,
        reward_model_target, value_model_target, policy_model_target,
        reward_model_mask, value_model_mask, policy_model_mask)

    #------------
    # model metrics
    #------------
    # just use simulations starting from timestep = 0
    visualize_metrics['visualize_model_data_t0'] = dict(
      model_outputs=jax.tree_map(lambda x:x[0], model_outputs),
      simulation_actions=simulation_actions,
      # model reward
      reward_model_ce=reward_model_ce[0],
      reward_model_mask=reward_model_mask[0],
      reward_model_target=self._discretizer.probs_to_scalar(reward_model_target[0]),
      reward_model_prediction=self._discretizer.logits_to_scalar(model_outputs.reward_logits[0]),
      # model policy
      policy_model_ce=policy_model_ce[0],
      policy_model_mask=policy_model_mask[0],
      policy_model_target=policy_model_target[0],
      policy_model_prediction=jax.nn.softmax(model_outputs.policy_logits[0]),
      # model value
      value_model_ce=value_model_ce[0],
      value_model_mask=value_model_mask[0],
      value_model_target=jnp.concatenate((returns[1:], jnp.zeros(nz)))[:self._simulation_steps],
      value_model_prediction=self._discretizer.logits_to_scalar(model_outputs.value_logits[0]),
    )

    visualize_metrics['visualize_model_data_t_all'] = dict(
      # model reward
      reward_model_ce=model_reward_loss,
      reward_model_mask=reward_mask[:nsteps],
      # model policy
      policy_model_ce=model_policy_loss,
      policy_model_mask=policy_mask[:nsteps],
      # model value
      value_model_ce=model_value_loss,
      value_model_mask=value_mask[:nsteps],
    )

    # all are []
    model_policy_loss = self._model_policy_coef * \
        masked_mean(model_policy_loss, policy_mask[:nsteps])
    model_value_loss = self._model_value_coef * \
        masked_mean(model_value_loss, value_mask[:nsteps])
    reward_loss = self._model_reward_coef * \
        masked_mean(model_reward_loss, reward_mask[:nsteps])

    loss_metrics.update({
        '0.1.policy_model_loss': model_policy_loss,
        '0.2.model_reward_loss': reward_loss, # T
        '0.3.value_model_loss': model_value_loss, # T
    })

    total_loss = (
        reward_loss +
        root_value_loss + model_value_loss + 
        root_policy_loss + model_policy_loss)

    return total_loss, loss_metrics, visualize_metrics, returns, mcts_values



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
