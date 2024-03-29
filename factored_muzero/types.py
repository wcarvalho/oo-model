"""MuZero types.
"""

import chex

from typing import Tuple, Optional

import jax.numpy as jnp
from acme import types
from muzero import types as muzero_types
from factored_muzero import attention


Task = types.NestedArray
TaskAwareSaviState = Tuple[attention.SaviState, Task]

@chex.dataclass(frozen=True)
class RootOutput(muzero_types.RootOutput):
  pred_attn_outputs: Optional[types.NestedArray] = None
  reconstruction: Optional[types.NestedArray] = None


@chex.dataclass(frozen=True)
class ModelOutput(muzero_types.ModelOutput):
  pred_attn_outputs: types.NestedArray
  # model_attn_outputs: types.NestedArray

