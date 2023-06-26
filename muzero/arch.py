
from typing import Callable, Optional, Tuple

from acme import types

import haiku as hk
import jax
import jax.numpy as jnp

from muzero import types as muzero_types
from muzero.utils import TaskAwareRecurrentFn

State = types.NestedArray
RewardLogits = jnp.ndarray
PolicyLogits = jnp.ndarray
ValueLogits = jnp.ndarray

RootFn = Callable[[State], Tuple[PolicyLogits, ValueLogits]]
ModelFn = Callable[[State], Tuple[RewardLogits, PolicyLogits, ValueLogits]]

def mask_policy_logits(output, invalid_actions = None):
  logits = output.policy_logits
  if invalid_actions is None:
    return output
  
  if invalid_actions.ndim < logits.ndim:
    diff = logits.ndim - invalid_actions.ndim
    tile = logits.shape[:diff] + (1,)*(logits.ndim - diff)
    invalid_actions = jnp.tile(invalid_actions, tile)

  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  min_logit = jnp.finfo(logits.dtype).min
  logits = jnp.where(invalid_actions, min_logit, logits)
  return output.replace(policy_logits=logits)

class MuZeroArch(hk.RNNCore):
  """MuZero Network Architecture.
  """

  def __init__(self,
               action_encoder,
               observation_fn: hk.Module,
               state_fn: hk.RNNCore,
               transition_fn: TaskAwareRecurrentFn,
               root_pred_fn: RootFn,
               model_pred_fn: ModelFn,
               invalid_actions: Optional[jnp.ndarray] = None,
               prep_state_input: Callable[
                  [types.NestedArray], types.NestedArray] = lambda x: x,
               prep_model_state_input: Callable[
                  [types.NestedArray], types.NestedArray] = lambda x: x,
               combine_hidden_obs: Callable[
                   [types.NestedArray, types.NestedArray], types.NestedArray] = lambda h, s: h,
               name='muzero_network'):
    super().__init__(name=name)
    self._action_encoder = action_encoder
    self._observation_fn = observation_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._root_pred_fn = root_pred_fn
    self._model_pred_fn = model_pred_fn
    self._prep_state_input = prep_state_input
    self._prep_model_state_input = prep_model_state_input
    self._combine_hidden_obs = combine_hidden_obs
    self._invalid_actions = invalid_actions

  def initial_state(self,
                    batch_size: Optional[int] = None,
                    **unused_kwargs) -> State:
    return self._state_fn.initial_state(batch_size)

  def __call__(
      self,
      inputs: types.NestedArray,  # [...]
      state: State  # [...]
  ) -> Tuple[muzero_types.RootOutput, State]:
    """Apply state function over input.

    In theory, this function can be applied to a batch but this has not been tested.

    Args:
        inputs (types.NestedArray): typically observation. [D]
        state (State): state to apply function to. [D]

    Returns:
        Tuple[muzero_types.RootOutput, State]: single muzero output and single new state.
    """

    embeddings = self._observation_fn(inputs)  # [D+A+1]
    state_input = self._prep_state_input(embeddings)

    # [D], [D]
    hidden, new_state = self._state_fn(state_input, state)

    hidden = self._combine_hidden_obs(hidden, embeddings)
    root_outputs = self._root_pred_fn(hidden)

    return root_outputs, new_state

  def unroll(
      self,
      inputs: types.NestedArray,  # [T, B, ...]
      state: State  # [B, ...]
  ) -> Tuple[muzero_types.RootOutput, State]:
    """Unroll state function over inputs.

    Args:
        inputs (types.NestedArray): typically observations. [T, B, ...]
        state (State): state to begin unroll at. [T, ...]

    Returns:
        Tuple[muzero_types.RootOutput, State]: muzero outputs and single new state.
    """
    embeddings = hk.BatchApply(self._observation_fn)(inputs)  # [T, B, D+A+1]
    state_input = self._prep_state_input(embeddings)

    all_hidden, new_state = hk.static_unroll(
        self._state_fn, state_input, state)

    all_hidden = self._combine_hidden_obs(all_hidden, embeddings)
    root_output = hk.BatchApply(self._root_pred_fn)(all_hidden)
    root_output = mask_policy_logits(root_output, self._invalid_actions)

    return root_output, new_state

  def apply_model(
      self,
      state: State, # [B, D]
      action: jnp.ndarray, # [B]
  ) -> Tuple[muzero_types.ModelOutput, State]:
    """This applies the model to each element in the state, action vectors.

    Args:
        state (State): states. [B, D]
        action (jnp.ndarray): actions to take on states. [B]

    Returns:
        Tuple[muzero_types.ModelOutput, State]: muzero outputs and new states for 
          each state state action pair.
    """
    # [B, A]
    action_onehot = self._action_encoder(action)
    state = self._prep_model_state_input(state)

    # [B, D], [B, D]
    hidden, new_state = self._transition_fn(action_onehot, state)

    model_output = self._model_pred_fn(hidden)
    model_output = mask_policy_logits(model_output, self._invalid_actions)

    # [B, D], [B, D]
    return model_output, new_state

  def unroll_model(
      self,
      state: State,  # [D]
      action_sequence: jnp.ndarray,  # [T]
  ) -> Tuple[muzero_types.ModelOutput, State]:
    """This unrolls the model starting from the state and applying the 
      action sequence.

    Args:
        state (State): starting state. [D]
        action_sequence (jnp.ndarray): actions to unroll. [T]

    Returns:
        Tuple[muzero_types.ModelOutput, State]: muzero outputs and single new state.
    """
    # [T, A]
    action_onehot = self._action_encoder(action_sequence)
    state = self._prep_model_state_input(state)

    # [T, D], [D]
    all_hidden, new_state = hk.static_unroll(
        self._transition_fn, action_onehot, state)

    # [T, D]
    model_output = self._model_pred_fn(all_hidden)
    model_output = mask_policy_logits(model_output, self._invalid_actions)

    # [T, D], [D]
    return model_output, new_state
