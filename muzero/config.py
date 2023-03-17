# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MuZero config."""
from typing import Callable, Tuple, Optional

from acme.adders import reverb as adders_reverb
from acme.agents.jax import r2d2
from acme import types as acme_types

import dataclasses
import rlax

Array = acme_types.NestedArray

@dataclasses.dataclass
class MuZeroConfig(r2d2.R2D2Config):
  """Configuration options for MuZero agent."""
  discount: float = 0.997**4  # from paper
  # target_update_period: int = 2500
  # evaluation_epsilon: float = 0.
  # variable_update_period: int = 400
  seed: int = 1234
  num_steps: int = 3e6

  # value-based action-selection options
  num_epsilons: int = 256
  epsilon_min: float = 1e-2
  epsilon_max: float = 1

  # Learner options
  burn_in_length: int = 0
  batch_size: Optional[int] = 64
  trace_length: Optional[int] = 20
  sequence_period: Optional[int] = None
  learning_rate: float = 1e-3
  learning_rate_decay: float = .1
  lr_transition_steps: int = 100_000
  weight_decay: float = 1e-4
  max_grad_norm: float = 80.0
  warmup_steps: int = 0
  ema_update: float = 0.0
  metrics: str = 'dense'
  # bootstrap_n: int = 5
  # clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  adam_eps: float = 1e-3

  # Replay options
  # samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 50.0
  min_replay_size: int = 1_000
  max_replay_size: int = 80_000
  target_batch_size: int = 1024
  prefetch_size: int = 0
  num_parallel_calls: int = 1
  # replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.0
  priority_exponent: float = 0.0
  # max_priority_weight: float = 0.9

  #Loss hps
  num_bins: Optional[int] = None  # number of bins for two-hot rep
  scalar_step_size: Optional[float] = .25  # number of bins for two-hot rep
  max_scalar_value: float = 10.0  # number of bins for two-hot rep
  td_steps: int = 5
  v_target_source: str = 'return'

  # MCTS general hps
  simulation_steps: int = 5
  num_simulations: int = 50
  q_normalize_epsilon: float = 0.01  # copied from `jax_muzero`
  muzero_policy: str = 'gumbel_muzero'

  # MCTS muzero hps
  dirichlet_fraction: float = 0.25
  dirichlet_alpha: float = 0.3
  pb_c_init: float = 1.25
  pb_c_base: float = 19652
  temperature: float = 1.0

  # MCTS gumble_muzero hps
  maxvisit_init: int = 50
  gumbel_scale: int = 1.0


  # Architecture
  ln: bool = True
  output_init: Optional[float] = None
  vision_torso: str = 'babyai'
  network_fn: str = 'babyai'
  model_combine_state_task: str = 'none'
  state_dim: int = 256
  vocab_size: int = 50  # vocab size for env
  word_dim: int = 128  # dimensionality of word embeddings
  sentence_dim: int = 128  # dimensionality of sentence embeddings
  task_dim: int = 128  # projection of task to lower dimension
  resnet_transition_dim: Optional[int] = None  # dim of resnet for transition function
  transition_blocks: int = 6  # number of resnet blocks
  prediction_blocks: int = 2  # number of resnet blocks
  seperate_model_nets: bool = False
  model_state_extract_fn: Callable[[Array], Array] = lambda state: state.hidden
  reward_mlps: Tuple[int] = (32,)
  vpi_mlps: Tuple[int] = (128, 32)
  conv_out_dim: int = 0

  # actor hps
  action_source: str = 'policy'  # 'policy', 'value', 'mcts'

  model_coef: float = 1.0
  policy_coef: float = 1.0
  value_coef: float = 0.25
  reward_coef: float = 1.0

  show_gradients: int = 0