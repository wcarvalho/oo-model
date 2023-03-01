"""MuZero types.
"""

from typing import Optional, Callable, Tuple, NamedTuple

import dataclasses

from acme import specs
from acme import types
from acme.jax import networks as networks_lib

import chex
import jax.numpy as jnp

BatchSize = int
PRNGKey = networks_lib.PRNGKey
Params = networks_lib.Params
RecurrentState = networks_lib.RecurrentState
NetworkOutput = networks_lib.NetworkOutput
Observation = networks_lib.Observation
Action = networks_lib.Action


@chex.dataclass(frozen=True)
class MuZeroState:
  state: types.NestedArray
  task: jnp.ndarray


@chex.dataclass(frozen=True)
class RootOutput:
  state: types.NestedArray
  value_logits: jnp.ndarray
  policy_logits: jnp.ndarray
  next_reward: Optional[jnp.ndarray] = None
  next_value: Optional[jnp.ndarray] = None


@chex.dataclass(frozen=True)
class ModelOutput:
  new_state: types.NestedArray
  reward_logits: types.NestedArray
  value_logits: types.NestedArray
  policy_logits: types.NestedArray


@dataclasses.dataclass
class MuZeroNetworks:
  """Network that can unroll state-fn and apply model over an input sequence."""
  unroll_init: Callable[[PRNGKey], Params]
  model_init: Callable[[PRNGKey], Params]
  apply: Callable[[Params, PRNGKey, Observation, RecurrentState],
                  Tuple[NetworkOutput, RecurrentState]]
  unroll: Callable[[Params, PRNGKey, Observation, RecurrentState],
                   Tuple[NetworkOutput, RecurrentState]]
  init_recurrent_state: Callable[[PRNGKey, Optional[BatchSize]], RecurrentState]
  apply_model: Callable[[Params, PRNGKey, RecurrentState, Action],
                  Tuple[NetworkOutput, RecurrentState]]


class MuZeroParams(NamedTuple):
    """Agent parameters."""

    unroll: networks_lib.Params
    model: networks_lib.Params


