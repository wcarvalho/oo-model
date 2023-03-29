"""Factored Muzer config."""
from typing import Callable, Tuple, Optional

from acme.adders import reverb as adders_reverb
from acme.agents.jax import r2d2
from acme import types as acme_types

import dataclasses
import rlax

from muzero.config import MuZeroConfig
Array = acme_types.NestedArray

@dataclasses.dataclass
class FactoredMuZeroConfig(MuZeroConfig):

  # postion embedding
  embedding_type: str = 'linear'
  update_type: str = 'project_add'

  # state function
  savi_iterations: int = 1
  num_slots: int = 4
  state_qkv_size: int = 128

  # transition function
  slot_tran_heads: int = 4
  slot_tran_qkv_size: int = 128
  slot_tran_mlp_size: int = 256
  transition_blocks: int = 2  # number of transformer blocks
  tran_use_task: bool = False

  # prediction functions
  prediction_blocks: int = 2  # number of transformer blocks
  slot_pred_heads: int = 4
  slot_pred_qkv_size: int = 128
  slot_pred_mlp_size: int = 256
