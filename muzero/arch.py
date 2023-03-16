
from typing import Callable, Optional, Tuple

from acme import types

import haiku as hk
import jax
import jax.numpy as jnp

from muzero import types as muzero_types
from muzero.utils import Discretizer

MuZeroState = muzero_types.MuZeroState
RewardLogits = jnp.ndarray
PolicyLogits = jnp.ndarray
ValueLogits = jnp.ndarray

RootFn = Callable[[MuZeroState], Tuple[PolicyLogits, ValueLogits]]
ModelFn = Callable[[MuZeroState], Tuple[RewardLogits, PolicyLogits, ValueLogits]]

class MuZeroArch(hk.RNNCore):
  """MuZero Network Architecture.
  """

  def __init__(self,
               action_encoder,
               observation_fn: hk.Module,
               state_fn: hk.RNNCore,
               transition_fn: hk.Module,
               root_pred_fn: RootFn,
               model_pred_fn: ModelFn,
               prep_state_input: Callable[[types.NestedArray], types.NestedArray] = lambda x: x,
               model_compute_r_v: bool = True,
               discount: float = 1.0,
               discretizer: Optional[Discretizer] = None,
               num_actions: Optional[int] = None,
               ):
    super().__init__(name='muzero_network')
    self._action_encoder = action_encoder
    self._discretizer = discretizer
    self._observation_fn = observation_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._root_pred_fn = root_pred_fn
    self._model_pred_fn = model_pred_fn
    self._prep_state_input = prep_state_input

    # for computing Q-values with model
    self._model_compute_r_v = model_compute_r_v
    self._discount = discount
    self._num_actions = num_actions

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> muzero_types.MuZeroState:
    return self._state_fn.initial_state(batch_size)

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
      inputs: types.NestedArray,  # [...]
      state: muzero_types.MuZeroState  # [...]
  ) -> Tuple[muzero_types.RootOutput, muzero_types.MuZeroState]:
    """Predict value and policy for time-step.

    Args:
        inputs (types.NestedArray): _description_

    Returns:
        Tuple[muzero_types.RootOutput, muzero_types.MuZeroState]: _description_
    """
    embeddings = self._observation_fn(inputs)  # [D+A+1]
    state_input = self._prep_state_input(embeddings)

    # [N, D]
    core_outputs, new_state = self._state_fn(state_input, state)
    muzero_state = muzero_types.MuZeroState(
        state=core_outputs,
        task=embeddings.task)

    policy_logits, value_logits = self._root_pred_fn(muzero_state)

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

    policy_logits, value_logits = hk.BatchApply(self._root_pred_fn)(muzero_state)
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
    action_onehot = self._action_encoder(action)
    new_state = self._transition_fn(
        action_onehot=action_onehot,
        prev_state=state.state)

    new_state = muzero_types.MuZeroState(
      state=new_state,
      task=state.task,
    )

    reward_logits, policy_logits, value_logits = self._model_pred_fn(new_state)

    model_output = muzero_types.ModelOutput(
      new_state=new_state,
      value_logits=value_logits,
      policy_logits=policy_logits,
      reward_logits=reward_logits,
    )
    return model_output, new_state

