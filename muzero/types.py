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


class TaskAwareState(NamedTuple):
  state: chex.Array
  task: chex.Array


@chex.dataclass(frozen=True)
class RootOutput:
  state: types.NestedArray
  value_logits: jnp.ndarray
  policy_logits: jnp.ndarray


@chex.dataclass(frozen=True)
class ModelOutput:
  new_state: types.NestedArray
  reward_logits: types.NestedArray
  value_logits: types.NestedArray
  policy_logits: types.NestedArray

InitFn = Callable[[PRNGKey], Params]
StateFn = Callable[[Params, PRNGKey, Observation, RecurrentState],
                  Tuple[NetworkOutput, RecurrentState]]
ModelFn = Callable[[Params, PRNGKey, RecurrentState, Action],
                   Tuple[NetworkOutput, RecurrentState]]
QFn = Callable[[Params, PRNGKey, RecurrentState], chex.Array]

@dataclasses.dataclass
class MuZeroNetworks:
  """Network that can unroll state-fn and apply model over an input sequence."""
  unroll_init: InitFn
  apply: StateFn
  unroll: StateFn
  init_recurrent_state: Callable[[PRNGKey, Optional[BatchSize]], RecurrentState]
  apply_model: ModelFn
  unroll_model: ModelFn
  model_init: Optional[InitFn] = None
  compute_q_values: Optional[QFn] = None


class MuZeroParams(NamedTuple):
    """Agent parameters."""

    unroll: networks_lib.Params
    model: networks_lib.Params


