"""Factored Muzer config."""
from typing import Callable, Tuple, Optional, List

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
  num_sgd_steps_per_step: int = 1

  # general arch (across all parts)
  gru_init: str = 'default'
  b_init_attn: float = 1.0
  w_init_attn: float = 1.0
  pre_norm: bool = True
  share_w_init_out: bool = False
  share_pred_base: bool = False
  slots_use_task: bool = True

  # postion embedding
  embedding_type: str = 'linear'
  pos_mlp_layers: Optional[Tuple[int]] = (128,)
  pos_layernorm: str = 'pre'
  update_type: str = 'concat'

  # observation function
  w_init_obs: Optional[float] = None
  vision_torso: str = 'babyai_patches'

  # state function
  inverted_attn: bool = True
  context_slot_dim: int = 32
  pos_embed_attn: bool = False
  slot_value_combination: str = 'avg'
  clip_attn_probs: bool = True
  savi_iterations: int = 4
  relation_dim: int = 64
  savi_mlp_size: Optional[int] = None
  savi_epsilon: float = 1e-5
  savi_rnn: str = 'gru'
  num_slots: int = 4
  slot_size: int = 64
  transform_pos_embed: bool = True
  fixed_point: bool = True
  savi_combo_update: str = 'sum'
  savi_model_update: str = 'sum'
  forward_context: bool = False
  relation_iterations: str = 'once'
  savi_init: str = 'gauss'
  savi_gumbel_temp: float = 1.0
  mask_context: str = 'softmax'
  attention_in_updates: bool = False
  mix_when: str = 'conv'

  # transition function
  model_gate: str = 'sum'
  slot_tran_heads: int = 2
  tran_mlp_blocks: int = 2
  slot_tran_mlp_size: Optional[int] = None
  transition_blocks: int = 4  # number of transformer blocks
  tran_out_mlp: bool = True

  # prediction functions
  seperate_model_nets: bool = True
  pred_gate: Optional[str] = 'gru'
  pred_task_combine: str = 'gather'
  prediction_blocks: int = 2 # number of transformer blocks
  pred_out_mlp: bool = True
  slot_pred_heads: Optional[int] = None
  slot_pred_mlp_size: Optional[int] = None
  slot_pred_qkv_size: Optional[int] = None
  learned_weights: str = 'softmax'

  # w_attn_head: bool = True
  pred_input_selection: str = 'attention_gate'  # IMPORTANT
  action_as_factor: bool = True
  pred_head: str = 'muzero'
  reward_mlps: Tuple[int] = (32,)
  vpi_mlps: Tuple[int] = (128, 32)
  query: str = 'task_rep'
  combine_context: str = 'sum'
  combine_factors: str = 'logits'


  # loss
  new_factored_learner: bool = True
  reanalyze_ratio: float = 0.25 # percent of time to use mcts vs. observed return
  state_model_loss: str = 'dot_contrast'
  contrast_gamma: float = 1e-2  # only for cswm and laplacian
  contrast_temp: float = 0.01  # only for dot_contrast
  state_model_coef: float = 1.0
  weight_decay_fn: str = "default"
  weight_decay: float = 1e-4  # very few params
  attention_penalty: float = 0.0
  extra_contrast: int = 10
  lr_transition_steps: int = 1_000_000
  root_target: str = 'model_target'

  recon_coeff: float = 0.0

  savi_grad_norm: Optional[float] = None
  muzero_grad_model: bool = False
  grad_fn: str = 'shared'