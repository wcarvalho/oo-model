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
from typing import Callable

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
  # burn_in_length: int = 40
  # trace_length: int = 80
  # sequence_period: int = 40
  # learning_rate: float = 1e-3
  # bootstrap_n: int = 5
  # clip_rewards: bool = False
  # tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # # Replay options
  # samples_per_insert_tolerance_rate: float = 0.1
  # samples_per_insert: float = 4.0
  min_replay_size: int = 100
  # max_replay_size: int = 100_000
  # batch_size: int = 64
  prefetch_size: int = 0
  num_parallel_calls: int = 1
  # replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # # Priority options
  # importance_sampling_exponent: float = 0.6
  # priority_exponent: float = 0.9
  # max_priority_weight: float = 0.9

  vocab_size: int = 50  # vocab size for env
  word_dim: int = 32  # dimensionality of word embeddings
  sentence_dim: int = 32  # dimensionality of sentence embeddings
  resnet_transition_dim: int = 256  # dim of resnet for transition function
  num_blocks: int = 8  # number of resnet blocks
  num_bins: int = 301  # number of bins for two-hot rep

  simulation_steps: int = 4
  model_state_extract_fn: Callable[[Array], Array] = lambda state: state.hidden
  num_simulations: int = 50
  maxvisit_init: int = 50
  gumbel_scale: int = 1.0
  td_steps: int = 4