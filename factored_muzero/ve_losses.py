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
import numpy as np

from muzero import types as muzero_types
from muzero.utils import Discretizer
from muzero.learner_logger import LearnerLogger
from muzero import ve_losses
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

unit = lambda a: (a / (1e-5+jnp.linalg.norm(a, axis=-1, keepdims=True)))
dot = lambda a,b: jnp.sum(a[:, None] * b[None], axis=-1)  # [A, B]


def graph_penalty(x, y):
    norm = lambda a: jnp.linalg.norm(a, axis=-1, keepdims=False)
    return jnp.square(dot(x,y)) - norm(x) - norm(y)
    # # Step 1: Compute the dot-products between x and y
    # z = dot(x, y)

    # # Step 2: Square each entry of the z matrix
    # z_squared = jnp.square(z)

    # # Step 3: Sum the off-diagonal entries while subtracting the diagonal entries
    # upper_diag_indices = jnp.triu_indices(z.shape[0], k=1)

    # dot_same = jnp.sum(jnp.diag(z_squared))
    # dot_diff = jnp.sum(z_squared[upper_diag_indices])

    # return dot_diff - dot_same


def squared_l2_norm(preds: Array, targets: Array,
                    reduction_type: str = "mean") -> Array:
  if reduction_type == "sum":
    return jnp.sum(jnp.square(preds - targets))
  elif reduction_type == "mean":
    return jnp.mean(jnp.square(preds - targets))
  else:
    raise ValueError(f"Unsupported reduction_type: {reduction_type}")

class ValueEquivalentLoss(ve_losses.ValueEquivalentLoss):

  def __init__(self,
               state_loss: str = 'dot_contrast',
               get_factors = None,
               extra_contrast: int = 5,
               contrast_gamma: float = 1.0,
               contrast_temp: float = 1.0,
               cswm_spread: float =.5,
               state_model_coef: float = 1.0,
               attention_penalty: float = 0.0,
               recon_coeff: float = 1.0,
               **kwargs,
               ):
    super().__init__(**kwargs)
    self.state_loss = state_loss
    self.extra_contrast = extra_contrast
    self.contrast_gamma = contrast_gamma
    self.contrast_temp = contrast_temp
    self.cswm_spread = cswm_spread
    self._state_model_coef = state_model_coef
    self._attention_penalty = attention_penalty
    self._recon_coeff = recon_coeff

    if get_factors is None:
      get_factors = lambda state: state.rep.factors
    self.get_factors = get_factors

  def learn(self,
            data: acme_types.NestedArray,
            in_episode: Array,
            is_terminal_mask: Array,
            online_outputs: muzero_types.RootOutput,
            target_outputs: muzero_types.RootOutput,
            online_state: State,
            target_state: State,
            all_online_outputs: muzero_types.RootOutput,
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
    raw_root_value_loss = masked_mean(root_value_ce, in_episode[:dim_return])
    root_value_loss = self._root_value_coef*raw_root_value_loss


    # [T]
    root_policy_ce = self._policy_loss_fn(policy_probs_target, online_outputs.policy_logits)
    # []
    raw_root_policy_loss = masked_mean(root_policy_ce, in_episode)
    root_policy_loss = self._root_policy_coef*raw_root_policy_loss

    recon_loss = raw_recon_loss = 0.0
    if self._recon_coeff:
      images = data.observation.observation.image
      images = images.astype(jnp.float32)/255.0
      prediction = online_outputs.reconstruction['image']
      raw_recon_loss = jax.vmap(squared_l2_norm)(prediction, images)
      # sum over feature-axis, mean over space/time
      raw_recon_loss *= images.shape[-1]
      raw_recon_loss = masked_mean(raw_recon_loss, in_episode)
      recon_loss = self._recon_coeff*raw_recon_loss

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
        online_state=self.get_state(online_outputs),
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


    if not learn_model:
      loss_metrics.update({
        "0.0.total_loss": total_loss,
        '0.1.policy_root_loss': raw_root_policy_loss,
        '0.3.value_root_loss': raw_root_value_loss,
        '0.4.recon_root_loss': raw_recon_loss,
      })
      total_loss = root_value_loss + root_policy_loss + recon_loss
      return total_loss, loss_metrics, visualize_metrics, returns, value_root_prediction

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
    mask_includes_terminal = jnp.concatenate((in_episode[1:], jnp.zeros(self._simulation_steps)))
    if self._mask_model:
      mask_no_terminal = jnp.concatenate((in_episode[1:dim_return], jnp.zeros(extra_v)))
      reward_mask = jnp.concatenate((in_episode[:-1], jnp.zeros(self._simulation_steps)))
    else:
      reward_mask = mask_no_terminal = jnp.ones_like(mask_includes_terminal)
    reward_model_mask = rolling_window(reward_mask, self._simulation_steps)
    mask_no_terminal_roll = rolling_window(mask_no_terminal, self._simulation_steps)
    mask_includes_terminal_roll = rolling_window(mask_includes_terminal, self._simulation_steps)


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
    online_state = self.get_state(online_outputs)
    model_outputs = model_unroll(
        model_keys, online_state, simulation_actions,
    )

    #------------
    # compute contrastive pieces
    #------------
    # [T, N, D]
    factors = self.get_factors(online_state)
    ntime, nfactor, dim = factors.shape

    # [T', sim_steps, factors, dim]
    vmap_roll_factor = jax.vmap(vmap_roll, 1, 2)
    def get_next_factors():
      # end should be masked out
      next_factors = jnp.concatenate((factors[1:], factors[-self._simulation_steps:]))
      return vmap_roll_factor(next_factors)
    def get_model_factors():
      # end should be masked out
      model_factors = model_outputs.new_state.factors
      model_factors = jnp.concatenate((model_factors[:-1],
                                      factors[-self._simulation_steps:]))
      return vmap_roll_factor(model_factors)

    def get_single_negatives():
      num_extra = (ntime_sim)*nfactor
      negative_idx = np.random.randint(all_factors.shape[0], size=num_extra)
      negatives = all_factors[negative_idx]

      # [T', N, D]
      negatives = negatives.reshape(ntime_sim, nfactor, dim)

      # [T', sim_steps, factors, dim]
      return vmap_roll_factor(negatives)

    # [B*T*N, D]
    all_factors = self.get_factors(self.get_state(all_online_outputs)).reshape(-1, dim)

    ntime_sim = ntime+self._simulation_steps - 1
    if self.state_loss == 'dot_contrast':
      # anchor = model_state_1
      # positive = state_1
      # negative = other factors + K random factors

      state_positive = get_next_factors()

      num_extra = (ntime_sim)*nfactor*self.extra_contrast
      negative_idx = np.random.randint(all_factors.shape[0], size=num_extra)
      negatives = all_factors[negative_idx]

      # [T', N, Extra, D]
      negatives = negatives.reshape(ntime_sim, nfactor, self.extra_contrast, dim)

      # [T', sim_steps, factors, extras, dim]
      state_negative = jax.vmap(vmap_roll_factor, 2, 3)(negatives)

    elif self.state_loss == 'cswm':
      # anchor = model_state_1
      # positive = state_1
      # negative = 1 random negative
      state_positive = get_next_factors()
      state_negative = get_single_negatives()

    elif 'laplacian-state' == self.state_loss:
      state_positive = get_next_factors()
      state_negative = get_single_negatives()

    elif 'laplacian-model' == self.state_loss:
      raise NotImplementedError
      # anchor = state_0
      # positive = model_state_1
      # negative = 1 random negative
      state_positive = get_model_factors()
      state_negative = get_single_negatives()
      raise NotImplementedError

    else:
      raise NotImplementedError

    #------------
    # compute loss
    #------------
    def compute_losses(
            model_outputs_,
            targets,
            masks):

      categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)

      reward_ce = categorical_cross_entropy(
          targets['reward'], model_outputs_.reward_logits)

      value_ce = categorical_cross_entropy(
          targets['value'], model_outputs_.value_logits)

      policy_ce = self._policy_loss_fn(
          targets['policy'], model_outputs_.policy_logits)

      if self.state_loss == 'dot_contrast':
        # [T, N, D]
        prediction = unit(model_outputs_.new_state.factors)
        state_positive = unit(targets['state_positive'])

        # [T, N, E, D]
        state_negative = unit(targets['state_negative'])

        def contrast(y_hat, y, neg):
          y_logits = dot(y_hat, y)  # [N, N]
          neg_logits = (y_hat[:, None]*neg).sum(-1)  # [N, E]

          #[N, N+E]
          logits = jnp.concatenate((y_logits, neg_logits), axis=-1)
          logits = logits/self.contrast_temp
          
          # Compute the softmax probabilities
          log_probs = jax.nn.log_softmax(logits)

          num_classes = y_hat.shape[0]   # N
          nlogits = log_probs.shape[-1]  # N+E
          labels = jax.nn.one_hot(jnp.arange(num_classes), num_classes=nlogits)
          return rlax.categorical_cross_entropy(labels, logits)

        state_ce = jax.vmap(contrast)(prediction, state_positive, state_negative)
        state_ce = state_ce.mean(-1)  # factors


      elif self.state_loss == 'cswm':
        # [T, N, D]
        prediction = model_outputs_.new_state.factors
        state_positive = targets['state_positive']
        state_negative = targets['state_negative']

        # [T]
        cons = self.cswm_spread/(self.cswm_spread*self.cswm_spread)
        positive_loss = cons*rlax.l2_loss(prediction, state_positive).mean((1, 2))
        negative_loss = cons*rlax.l2_loss(prediction, state_negative).mean((1, 2))

        state_ce = positive_loss + jax.nn.relu(self.contrast_gamma - negative_loss)

      else:
        state_ce = jnp.zeros_like(policy_ce)

      ce = {
          'reward': reward_ce,
          'value': value_ce,
          'policy': policy_ce,
          'state': state_ce,
      }
      loss = {
          'reward': masked_mean(reward_ce, masks['reward']),
          'value': masked_mean(value_ce, masks['value']),
          'policy': masked_mean(policy_ce, masks['policy']),
          'state': masked_mean(state_ce, masks['state']),
      }
      return ce, loss

    model_targets = {
      'reward': reward_model_target,
      'value': value_model_target,
      'policy': policy_model_target,
      'state_positive': state_positive,
      'state_negative': state_negative,
    }
    model_masks = {
      'reward': reward_model_mask,
      'value': mask_no_terminal_roll,
      'policy': mask_includes_terminal_roll,
      'state': mask_no_terminal_roll,
    }
    # vmap over time, sim-steps
    model_ces, model_losses = jax.vmap(compute_losses)(
        model_outputs,
        model_targets,
        model_masks)

    reward_model_ce = model_ces['reward']
    model_reward_loss = model_losses['reward']
    value_model_ce = model_ces['value']
    model_value_loss = model_losses['value']
    policy_model_ce = model_ces['policy']
    model_policy_loss = model_losses['policy']

    # state_model_ce = model_ces['state']
    state_loss = model_losses['state']

    if 'laplacian-state' == self.state_loss:
      anchor = factors[:-1]  # [T, N, D]
      positive = factors[1:]

      # [T, N]
      graph_loss = rlax.l2_loss(anchor, positive).mean(-1).mean(-1)
      penalty = jax.vmap(graph_penalty)(anchor, positive).mean(-1).mean(-1)
      state_loss = graph_loss + self.contrast_gamma*penalty

      state_loss = state_loss.mean(-1)

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
      policy_model_mask=mask_includes_terminal_roll[0],
      policy_model_target=policy_model_target[0],
      policy_model_prediction=jax.nn.softmax(model_outputs.policy_logits[0]),
      # model value
      value_model_ce=value_model_ce[0],
      value_model_mask=mask_no_terminal_roll[0],
      value_model_target=jnp.concatenate((returns[1:], jnp.zeros(nz)))[:self._simulation_steps],
      value_model_prediction=self._discretizer.logits_to_scalar(model_outputs.value_logits[0]),
    )

    visualize_metrics['visualize_model_data_t_all'] = dict(
      # model reward
      reward_model_ce=model_reward_loss,
      reward_model_mask=reward_mask[:nsteps],
      # model policy
      policy_model_ce=model_policy_loss,
      policy_model_mask=mask_includes_terminal[:nsteps],
      # model value
      value_model_ce=model_value_loss,
      value_model_mask=mask_no_terminal[:nsteps],
    )

    # all are []
    raw_model_policy_loss = masked_mean(model_policy_loss,
      mask_includes_terminal[:nsteps])

    raw_model_value_loss = masked_mean(model_value_loss,
      mask_no_terminal[:nsteps])

    raw_model_reward_loss = masked_mean(model_reward_loss,
      reward_mask[:nsteps])

    raw_state_loss = masked_mean(state_loss,
      mask_no_terminal[:nsteps])

    raw_total_loss = (
        raw_model_reward_loss +
        raw_root_value_loss +
        raw_model_value_loss + 
        raw_root_policy_loss +
        raw_model_policy_loss + 
        raw_recon_loss
        )

    total_loss = (
        self._model_reward_coef * raw_model_reward_loss +
        self._root_value_coef * raw_root_value_loss +
        self._model_value_coef * raw_model_value_loss +
        self._root_policy_coef * raw_root_policy_loss +
        self._model_policy_coef * raw_model_policy_loss + 
        recon_loss
    )
    loss_metrics.update({
        "0.0.total_loss": total_loss,
        "0.0.raw_total_loss": raw_total_loss,
        '0.1.policy_root_loss': raw_root_policy_loss,
        '0.1.policy_model_loss': raw_model_policy_loss,
        '0.2.model_reward_loss': raw_model_reward_loss,
        '0.3.value_root_loss': raw_root_value_loss,
        '0.3.value_model_loss': raw_model_value_loss,
        '0.4.recon_root_loss': raw_recon_loss,
    })

    if self._state_model_coef > 0.0:
      raw_total_loss = raw_total_loss + raw_state_loss
      total_loss = total_loss + self._state_model_coef * raw_state_loss
      loss_metrics['0.4.state_model_loss'] = raw_state_loss


    if self._attention_penalty > 0.0:
      # [T, n=num_slots, p=spatial_positions]
      slot_attn = online_outputs.state.rep.attn
      # [T, p]
      total_per_spatial_attn = slot_attn.sum(-2)

      # [T]
      l1_attn = total_per_spatial_attn.sum(-1)

      # []
      l1_attn_loss = masked_mean(l1_attn, in_episode)

      raw_total_loss = raw_total_loss + l1_attn_loss
      total_loss = total_loss + self._attention_penalty*l1_attn_loss
      loss_metrics['0.5.attn_l1'] = l1_attn_loss

    return total_loss, loss_metrics, visualize_metrics, returns, value_root_prediction

  def evaluate_model(self,
               data: acme_types.NestedArray,
               online_outputs: muzero_types.RootOutput,
               target_outputs: muzero_types.RootOutput,
               rng_key: networks_lib.PRNGKey,
               ):
    #---------------
    # policy
    #---------------
    num_actions = target_outputs.policy_logits.shape[-1]
    policy_target = jax.nn.one_hot(data.action, num_classes=num_actions)

    #---------------
    # value
    #---------------
    def compute_episode_return(rewards, gamma):
        zeros = jnp.zeros_like(rewards[:-1])
        episode_return = jnp.concatenate((zeros, rewards[-1][None]))

        for t in range(len(rewards)-2, -1, -1):
            episode_return.at[t].set(rewards[t] + gamma * episode_return[t+1])

        return episode_return
    value_target = compute_episode_return(data.reward, gamma=self._discount)

    #---------------
    # model outputs
    #---------------
    _, model_key = jax.random.split(rng_key)
    state = self.get_state(online_outputs)
    state_0 = jax.tree_map(lambda x: x[0], state)

    model_outputs, _ = self._networks.unroll_model(
        self._params, model_key, state_0, data.action)
    # ignore last step
    model_outputs = jax.tree_map(lambda x:x[:-1], model_outputs)

    reward_predictions = self._discretizer.logits_to_scalar(
        model_outputs.reward_logits)
    value_predictions = self._discretizer.logits_to_scalar(
        model_outputs.value_logits)

    #---------------
    # model outputs
    #---------------
    reward_error = rlax.l2_loss(
      reward_predictions,
      data.reward[:-1])
    value_error = rlax.l2_loss(
      value_predictions,
      value_target[:len(value_predictions)])
    policy_error = self._policy_loss_fn(
        policy_target[1:],
        model_outputs.policy_logits)

    in_episode = jnp.concatenate(
      (jnp.ones((2)), data.discount[:-2]), axis=0)[1:]

    ntimesteps = len(in_episode)
    timesteps = jnp.arange(ntimesteps)
    recency_weights = jnp.power(0.95, timesteps)


    return dict(
      model_l2_reward=masked_mean(reward_error, in_episode),
      model_l2_value=masked_mean(value_error, in_episode),
      model_ce_policy=masked_mean(policy_error, in_episode),
      model_weighted_l2_reward=masked_mean(
        reward_error*recency_weights, in_episode),
      model_weighted_l2_value=masked_mean(
        value_error*recency_weights, in_episode),
      model_weighted_ce_policy=masked_mean(
        policy_error*recency_weights, in_episode),
    )



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
