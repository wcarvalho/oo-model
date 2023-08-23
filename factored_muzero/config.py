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
  batch_size: Optional[int] = 32
  trace_length: Optional[int] = 10
  max_replay_size: int = 40_000
  num_sgd_steps_per_step: int = 4

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
  w_init_obs: Optional[float] = None
  vision_torso: str = 'babyai_patches'

  # state function
  context_as_slot: bool = False
  project_slot_values: bool = True
  slot_value_combination: str = 'avg'
  savi_iterations: int = 4
  savi_temp: float = 1.0
  savi_mlp_size: Optional[int] = None
  savi_epsilon: float = 1e-5
  savi_rnn: str = 'gru'
  num_slots: int = 4
  slot_size: int = 64
  savi_combo_update: str = 'concat'
  relation_iterations: str = 'once'
  savi_init: str = 'gauss'
  savi_gumbel_temp: float = 1.0
  mask_context: str = 'softmax'

  # transition function
  model_gate: str = 'sum'
  slot_tran_heads: int = 4
  tran_mlp_blocks: int = 2
  slot_tran_mlp_size: Optional[int] = None
  transition_blocks: int = 4  # number of transformer blocks
  tran_out_mlp: bool = True

  # prediction functions
  pred_gate: Optional[str] = 'gru'
  pred_task_combine: str = 'gather'
  prediction_blocks: int = 2 # number of transformer blocks
  pred_out_mlp: bool = True
  slot_pred_heads: Optional[int] = None
  slot_pred_mlp_size: Optional[int] = None
  slot_pred_qkv_size: Optional[int] = None

  # w_attn_head: bool = True
  pred_input_selection: str = 'attention_gate'  # IMPORTANT
  action_as_factor: bool = True
  pred_head: str = 'muzero'
  reward_mlps: Tuple[int] = (32,)
  vpi_mlps: Tuple[int] = (128, 32)
  query: str = 'task_rep'


  # loss
  state_model_loss: str = 'dot_contrast'
  contrast_gamma: float = 1.0  # only for cswm and laplacian
  contrast_temp: float = 0.01  # only for dot_contrast
  state_model_coef: float = 0.00
  weight_decay_fn: str = "default"
  weight_decay: float = 1e-4  # very few params
  attention_penalty: float = 0.0
  extra_contrast: int = 5

  recon_coeff: float = 1.0
