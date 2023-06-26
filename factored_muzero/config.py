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
  num_sgd_steps_per_step: int = 2

  # general arch (across all parts)
  gru_init: str = 'default'
  b_init_attn: float = 1.0
  w_init_attn: float = 1.0
  pre_norm: bool = False
  share_w_init_out: bool = False
  share_pred_base: bool = True
  slots_use_task: bool = False

  # postion embedding
  embedding_type: str = 'linear'
  pos_mlp_layers: Optional[Tuple[int]] = None
  pos_layernorm: str = 'pre'
  update_type: str = 'project_add'

  # observation function
  w_init_obs: float = 1.0
  vision_torso: str = 'babyai_patches'

  # state function
  project_slot_values: bool = True
  slot_value_combination: str = 'avg'
  savi_iterations: int = 4
  savi_temp: float = 1.0
  savi_epsilon: float = 1e-5
  savi_rnn: str = 'gru'
  num_slots: int = 4
  slot_size: int = 64
  savi_combo_update: str = 'concat'
  relation_iterations: str = 'once'
  savi_init: str = 'gauss'
  savi_gumbel_temp: float = 1.0

  # transition function
  model_gate: str = 'sum'
  slot_tran_heads: int = 1
  tran_mlp_blocks: int = 2
  slot_tran_mlp_size: Optional[int] = None
  transition_blocks: int = 4  # number of transformer blocks
  tran_out_mlp: bool = True

  # prediction functions
  pred_gate: Optional[str] = 'gru'
  pred_task_combine: str = 'cross'
  prediction_blocks: int = 1 # number of transformer blocks
  pred_out_mlp: bool = False
  slot_pred_heads: Optional[int] = None
  slot_pred_mlp_size: Optional[int] = None
  slot_pred_qkv_size: Optional[int] = 64
  # w_attn_head: bool = True
  pred_input_selection: str = 'attention_gate'  # IMPORTANT
  action_as_factor: bool = True
  pred_head: str = 'muzero'
  reward_mlps: Tuple[int] = (32,)
  vpi_mlps: Tuple[int] = (128, 32)
  query: str = 'task_rep'

