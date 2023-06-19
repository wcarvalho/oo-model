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

  # replay
  batch_size: Optional[int] = 64
  trace_length: Optional[int] = 20
  max_replay_size: int = 40_000

  # general arch (across all parts)
  gru_init: str = 'default'
  gating: str = 'gru'
  b_init_attn: float = 1.0
  w_init_attn: float = 1.0
  pre_norm: bool = False
  share_w_init_out: bool = False
  slots_use_task: bool = False
  pos_layernorm: str = 'pre'

  # postion embedding
  embedding_type: str = 'linear'
  pos_mlp_layers: Tuple[int] = (64,)
  update_type: str = 'project_add'

  # state function
  savi_iterations: int = 1
  savi_temp: float = 1.0
  savi_rnn: str = 'gru'
  num_slots: int = 4
  slot_size: int = 64

  # transition function
  slot_tran_heads: int = 4
  slot_tran_qkv_size: Optional[int] = None
  slot_tran_mlp_size: Optional[int] = None
  transition_blocks: int = 2  # number of transformer blocks

  # prediction functions
  prediction_blocks: int = 2  # number of transformer blocks
  slot_pred_heads: Optional[int] = None
  slot_pred_qkv_size: Optional[int] = None
  slot_pred_mlp_size: Optional[int] = None
  # w_attn_head: bool = True
  pred_input_selection: str = 'attention'
  action_as_factor: bool = False
  pred_head: str = 'muzero'
  pred_gate: str = 'gru'
  reward_mlps: Tuple[int] = (32,)
  vpi_mlps: Tuple[int] = (128, 32)

