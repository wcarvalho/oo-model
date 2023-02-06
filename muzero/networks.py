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

"""MuZero Networks."""

import dataclasses
from typing import Callable, Optional, Tuple

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils


from acme.wrappers import observation_action_reward

import haiku as hk
import jax
import jax.numpy as jnp


from modules import vision
from modules import language
from modules import embedding
from modules.mlp_muzero import BasicMlp, Transition
from muzero.arch import MuZeroArch
from muzero.types import MuZeroNetworks
from muzero.config import MuZeroConfig


# MuZeroNetworks = networks_lib.UnrollableNetwork
NetworkOutput = networks_lib.NetworkOutput
RecurrentState = networks_lib.RecurrentState


def make_babyai_networks(
  env_spec: specs.EnvironmentSpec,
  config: MuZeroConfig) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values


  def batch_observation_fn(inputs: observation_action_reward.OAR) -> jnp.ndarray:
    """encode observation + language instruction + concatenate.

    Args:
        inputs (types.NestedArray): [B, ...]

    Returns:
        jnp.ndarray: concatenated embedding.
    """
    batched = len(inputs.observation.image.shape) == 4

    def observation_fn(
        inputs: observation_action_reward.OAR) -> jnp.ndarray:
        """Same as above but assumes data has _no_ batch [B] dimension."""
        # compute task encoding
        task_encoder = language.LanguageEncoder(
            vocab_size=config.vocab_size,
            word_dim=config.word_dim,
            sentence_dim=config.sentence_dim,
        )
        task = task_encoder(inputs.observation.mission)

        # compute image encoding
        inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)
        image = vision.BabyAIVisionTorso()(
          inputs.observation.image/255.0)

        # combine task, image, reward, action reps
        embedder = embedding.OARTEmbedding(num_actions)

        return embedder(inputs, obs=image, task=task)

    if batched:
      observation_fn = jax.vmap(observation_fn)
    return observation_fn(inputs)

  def make_core_module() -> MuZeroNetworks:
    
    res_dim = config.resnet_transition_dim
    return MuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      observation_fn=hk.to_module(
        batch_observation_fn)(name="ObservationFn"),
      state_fn=hk.LSTM(512),
      transition_fn=Transition(
        channels=res_dim,
        num_blocks=config.num_blocks,
      ),
      root_value_fn=BasicMlp([res_dim, 128], config.num_bins),
      root_policy_fn=BasicMlp([res_dim, 128], num_actions),
      model_reward_fn=BasicMlp([res_dim, 128], config.num_bins),
      model_value_fn=BasicMlp([res_dim, 128], config.num_bins),
      model_policy_fn=BasicMlp([res_dim, 128], num_actions))

  return make_unrollable_model_network(env_spec, make_core_module)


def make_unrollable_model_network(
        environment_spec: specs.EnvironmentSpec,
        make_core_module: Callable[[], hk.RNNCore]) -> MuZeroNetworks:
  """Builds a MuZeroNetworks from a hk.Module factory."""

  dummy_observation = jax_utils.zeros_like(environment_spec.observations)
  dummy_action = jnp.array(0)

  def make_unrollable_network_functions():
    network = make_core_module()

    def init() -> Tuple[NetworkOutput, RecurrentState]:
      return network(dummy_observation, network.initial_state(None))

    apply = network.__call__
    return init, (apply,
                  network.unroll,
                  network.initial_state)

  # Transform and unpack pure functions
  f = hk.multi_transform(make_unrollable_network_functions)
  apply, unroll, initial_state_fn = f.apply

  def make_transition_model_functions():
    network = make_core_module()

    def init() -> Tuple[NetworkOutput, RecurrentState]:
      return network.apply_model(network.initial_state(None).hidden, dummy_action)

    return init, network.apply_model

  # Transform and unpack pure functions
  g = hk.multi_transform(make_transition_model_functions)
  apply_model = g.apply

  def init_recurrent_state(key: jax_types.PRNGKey,
                           batch_size: Optional[int]) -> RecurrentState:
    # TODO(b/244311990): Consider supporting parameterized and learnable initial
    # state functions.
    no_params = None
    return initial_state_fn(no_params, key, batch_size)

  return MuZeroNetworks(
    unroll_init=f.init,
    model_init=g.init,
    apply=apply,
    unroll=unroll,
    apply_model=apply_model,
    init_recurrent_state=init_recurrent_state)
