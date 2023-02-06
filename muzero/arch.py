
from typing import Optional, Tuple

from acme import types

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from muzero import types as muzero_types

class MuZeroArch(hk.RNNCore):
  """MuZero Network Architecture.
  """

  def __init__(self,
               action_encoder,
               observation_fn: hk.Module,
               state_fn: hk.Module,
               transition_fn: hk.Module,
               root_value_fn: hk.Module,
               root_policy_fn: hk.Module,
               model_reward_fn: hk.Module,
               model_value_fn: hk.Module,
               model_policy_fn: hk.Module,
               ):
    super().__init__(name='muzero_network')
    self._action_encoder = action_encoder
    self._observation_fn = observation_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._root_value_fn = root_value_fn
    self._root_policy_fn = root_policy_fn
    self._model_reward_fn = model_reward_fn
    self._model_value_fn = model_value_fn
    self._model_policy_fn = model_policy_fn

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> muzero_types.MuZeroState:
    return self._state_fn.initial_state(batch_size)

  def root_predictions(self, state):
    policy_out = self._root_policy_fn(state)
    value_out = self._root_value_fn(state)
    return policy_out, value_out

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
    core_outputs, new_state = self._state_fn(embeddings, state)
    policy_out, value_out = self.root_predictions(core_outputs)
    return muzero_types.RootOutput(
      state=core_outputs,
      value_logits=value_out,
      policy_logits=policy_out,
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
    core_outputs, new_states = hk.static_unroll(self._state_fn, embeddings, state)
    policy_out, value_out = hk.BatchApply(self.root_predictions)(core_outputs)
    return muzero_types.RootOutput(
        state=core_outputs,
        value_logits=value_out,
        policy_logits=policy_out,
    ), new_states

  def apply_model(
      self,
      state: jnp.ndarray,  # [[B], ...]
      action: jnp.ndarray,  # [[B], ...]
  ) -> Tuple[muzero_types.ModelOutput, muzero_types.MuZeroState]:
    """_summary_

    Args:
        state (muzero_types.MuZeroState): _description_

    Returns:
        Tuple[muzero_types.ModelOutput, muzero_types.MuZeroState]: _description_
    """
    new_state = self._transition_fn(
        encoded_action=self._action_encoder(action),
        prev_state=state)
    reward_out = self._model_reward_fn(new_state)
    value_out = self._model_value_fn(new_state)
    policy_out = self._model_policy_fn(new_state)

    return muzero_types.ModelOutput(
      new_state=new_state,
      value_logits=value_out,
      policy_logits=policy_out,
      reward_logits=reward_out,
    ), new_state

