
from typing import Callable, Optional, Tuple

from acme import types

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from muzero import types as muzero_types
from muzero.utils import Discretizer


class MuZeroArch(hk.RNNCore):
  """MuZero Network Architecture.
  """

  def __init__(self,
               action_encoder,
               observation_fn: hk.Module,
               state_fn: hk.Module,
               transition_fn: hk.Module,
               root_vpi_base: hk.Module,
               root_value_fn: hk.Module,
               root_policy_fn: hk.Module,
               model_vpi_base: hk.Module,
               model_reward_fn: hk.Module,
               model_value_fn: hk.Module,
               model_policy_fn: hk.Module,
               prep_state_input: Callable[[types.NestedArray], types.NestedArray],
               model_combine_state_task: str = 'none',
               model_compute_r_v: bool = False,
               discount: float = 1.0,
               discretizer: Optional[Discretizer] = None,
              #  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
              #  num_bins: Optional[int] = None,
               num_actions: Optional[int] = None,
               ):
    super().__init__(name='muzero_network')
    self._action_encoder = action_encoder
    self._discretizer = discretizer
    self._observation_fn = observation_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._root_vpi_base = root_vpi_base
    self._root_value_fn = root_value_fn
    self._root_policy_fn = root_policy_fn
    self._model_vpi_base = model_vpi_base
    self._model_reward_fn = model_reward_fn
    self._model_value_fn = model_value_fn
    self._model_policy_fn = model_policy_fn
    self._prep_state_input = prep_state_input
    # for computing Q-values with model
    self._model_compute_r_v = model_compute_r_v
    self._discount = discount
    if model_compute_r_v:
      assert discretizer is not None, "need this for computing r, v, q"
      assert num_actions is not None, "need this for computing r, v, q"
    self._num_actions = num_actions
    self._model_combine_state_task = model_combine_state_task.lower()

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> muzero_types.MuZeroState:
    return self._state_fn.initial_state(batch_size)

  def root_predictions(self, state):
    state = self._root_vpi_base(state)
    policy_logits = self._root_policy_fn(state)
    value_logits = self._root_value_fn(state)
    return policy_logits, value_logits

  def model_compute_r_v(self, state):
    """Compute Q(s,a) = r(s,a,s') + gamma*Value(s').

    Unroll 1-step model, compute value at next-time, and combine
    with current reward.

    Args:
        state (jnp.ndarray): [D]
        value_logits (jnp.ndarray): [N]
        policy_logits (jnp.ndarray): [A]

    Returns:
        jnp.ndarray: Q(s,a) [A]
    """
    actions = jnp.arange(self._num_actions, dtype=jnp.int32)
    # vmap over action dimension, to get 1 estimate per action
    apply_model = jax.vmap(self.apply_model, in_axes=(None, 0))

    # [A, ...]
    next_logits, _ = apply_model(state, actions)

    value = self._discretizer.logits_to_scalar(next_logits.value_logits)
    reward = self._discretizer.logits_to_scalar(next_logits.reward_logits)

    q_value = reward + self._discount*value
    return q_value, reward, value

  def __call__(
      self,
      inputs: types.NestedArray,  # [B, ...]
      state: muzero_types.MuZeroState  # [B, ...]
  ) -> Tuple[muzero_types.RootOutput, muzero_types.MuZeroState]:
    """Predict value and policy for time-step.

    Args:
        inputs (types.NestedArray): _description_

    Returns:
        Tuple[muzero_types.RootOutput, muzero_types.MuZeroState]: _description_
    """
    embeddings = self._observation_fn(inputs)  # [B, D+A+1]
    state_input = self._prep_state_input(embeddings)
    core_outputs, new_state = self._state_fn(state_input, state)
    policy_logits, value_logits = self.root_predictions(core_outputs)

    muzero_state = muzero_types.MuZeroState(
        state=core_outputs,
        task=embeddings.task)
    q_value = next_reward = next_value = None
    if self._model_compute_r_v:
      q_value, next_reward, next_value = self.model_compute_r_v(muzero_state)

    return muzero_types.RootOutput(
      state=muzero_state,
      value_logits=value_logits,
      policy_logits=policy_logits,
      next_reward=next_reward,
      next_value=next_value,
      q_value=q_value,
    ), new_state

  def unroll(
      self,
      inputs: types.NestedArray,  # [T, B, ...]
      state: muzero_types.MuZeroState  # [T, ...]
  ) -> Tuple[muzero_types.RootOutput, muzero_types.MuZeroState]:
    """Apply model

    Args:
        inputs (types.NestedArray): _description_

    Returns:
        Tuple[muzero_types.RootOutput, muzero_types.MuZeroState]: _description_
    """
    embeddings = hk.BatchApply(self._observation_fn)(inputs)  # [T, B, D+A+1]
    state_input = self._prep_state_input(embeddings)

    core_outputs, new_states = hk.static_unroll(
        self._state_fn, state_input, state)

    muzero_state = muzero_types.MuZeroState(
        state=core_outputs,
        task=embeddings.task)

    policy_logits, value_logits = hk.BatchApply(self.root_predictions)(core_outputs)
    q_value = next_reward = next_value = None
    if self._model_compute_r_v:
      q_value, next_reward, next_value = jax.vmap(jax.vmap(self.model_compute_r_v))(
        muzero_state)

    return muzero_types.RootOutput(
        state=muzero_state,
        value_logits=value_logits,
        policy_logits=policy_logits,
        next_reward=next_reward,
        next_value=next_value,
        q_value=q_value,
    ), new_states

  def apply_model(
      self,
      state: muzero_types.MuZeroState,
      action: jnp.ndarray,
  ) -> Tuple[muzero_types.ModelOutput, muzero_types.MuZeroState]:
    """_summary_

    Args:
        state (muzero_types.MuZeroState): _description_

    Returns:
        Tuple[muzero_types.ModelOutput, muzero_types.MuZeroState]: _description_
    """
    state_dim = state.state.shape[-1]
    action_onehot = self._action_encoder(action)
    

    assert self._model_combine_state_task in [
      'add_state', 'add_state_bias',
      'add_head', 'add_head_bias',
      'none',
      ]
    if 'bias' in self._model_combine_state_task:
      linear = lambda x: hk.Linear(state_dim, with_bias=True)(x)
    else:
      linear = lambda x: hk.Linear(state_dim, with_bias=False)(x)

    prev_state = state.state
    if 'add_state' in self._model_combine_state_task:
      prev_state = linear(prev_state) + linear(state.task)
    # else:
    #   raise RuntimeError(self._model_combine_state_task)

    new_state = self._transition_fn(
        action_onehot=action_onehot,
        prev_state=prev_state)
    
    pred_input = new_state
    if 'add_head' in self._model_combine_state_task:
      pred_input = linear(pred_input) + linear(state.task)

    reward_out = self._model_reward_fn(pred_input)

    x = self._model_vpi_base(pred_input)
    value_logits = self._model_value_fn(x)
    policy_logits = self._model_policy_fn(x)

    model_output = muzero_types.ModelOutput(
      new_state=new_state,
      value_logits=value_logits,
      policy_logits=policy_logits,
      reward_logits=reward_out,
    )
    new_state = muzero_types.MuZeroState(
      state=new_state,
      task=state.task,
    )
    return model_output, new_state

