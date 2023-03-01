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
  discount: float = 0.997
  # target_update_period: int = 2500
  # evaluation_epsilon: float = 0.
  # num_epsilons: int = 256
  # variable_update_period: int = 400
  seed: int = 1234
  num_steps: int = 3e6

  # # Learner options
  burn_in_length: int = 4
  # trace_length: int = 80
  # sequence_period: int = 40
  learning_rate: float = 1e-3
  learning_rate_decay: float = .1
  lr_transition_steps: int = 100_000
  weight_decay: float = 0.0
  max_grad_norm: float = 5.0
  warmup_steps: int = 1_000
  # bootstrap_n: int = 5
  # clip_rewards: bool = False
  # tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # # Replay options
  # samples_per_insert_tolerance_rate: float = 0.1
  # samples_per_insert: float = 4.0
  min_replay_size: int = 10_000
  max_replay_size: int = 80_000
  # batch_size: int = 64
  prefetch_size: int = 0
  num_parallel_calls: int = 1
  # replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # # Priority options
  # importance_sampling_exponent: float = 0.6
  # priority_exponent: float = 0.9
  # max_priority_weight: float = 0.9

  #Loss hps
  num_bins: int = 301  # number of bins for two-hot rep
  td_steps: int = 5

  # MCTS hps
  simulation_steps: int = 7
  num_simulations: int = 50
  maxvisit_init: int = 50
  gumbel_scale: int = 1.0
  q_normalize_epsilon: float = 0.01  # copied from `jax_muzero`

  # Architecture
  state_dim: int = 256
  vocab_size: int = 50  # vocab size for env
  word_dim: int = 64  # dimensionality of word embeddings
  sentence_dim: int = 64  # dimensionality of sentence embeddings
  resnet_transition_dim: Optional[int] = None  # dim of resnet for transition function
  transition_blocks: int = 2  # number of resnet blocks
  prediction_blocks: int = 2  # number of resnet blocks
  seperate_model_nets: bool = True
  model_state_extract_fn: Callable[[Array], Array] = lambda state: state.hidden
  reward_mlps: Tuple[int] = (32,)
  vpi_mlps: Tuple[int] = (128, 32)
  action_dim : int = 32

  # actor hps
  action_source: str = 'policy'  # 'policy', 'value', 'mcts'


  policy_coef: float = 1.0
  value_coef: float = 0.25
  reward_coef: float = 1.0