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
from modules import vision_language
from modules import simple_mlp_muzero
from modules.mlp_muzero import PredictionMlp, Transition, ResMlp
from modules.conv_muzero import VisionTorso as MuZeroVisionTorso
from muzero.arch import MuZeroArch
from muzero.types import MuZeroNetworks
from muzero.config import MuZeroConfig
from muzero.utils import Discretizer


# MuZeroNetworks = networks_lib.UnrollableNetwork
NetworkOutput = networks_lib.NetworkOutput
RecurrentState = networks_lib.RecurrentState

def concat_embeddings(embeddings):
  return jnp.concatenate((
    embeddings.image,
    embeddings.action,
    embeddings.reward,
    embeddings.task), axis=-1)


def make_babyai_networks(
  env_spec: specs.EnvironmentSpec,
  config: MuZeroConfig,
  discretizer: Discretizer) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> MuZeroNetworks:
    state_dim = config.state_dim
    res_dim = config.resnet_transition_dim or state_dim

    assert config.vision_torso in ('babyai', 'atari', 'muzero')
    if config.vision_torso == 'babyai':
      vision_torso = vision.BabyAIVisionTorso(conv_dim=0, flatten=False)
    elif config.vision_torso == 'atari':
      vision_torso = vision.AtariVisionTorso(conv_dim=0, flatten=False)
    elif config.vision_torso == 'muzero':
      vision_torso = MuZeroVisionTorso(channels=64, num_blocks=6)

    observation_fn = vision_language.Torso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder=language.LanguageEncoder(
          vocab_size=config.vocab_size,
          word_dim=config.word_dim,
          sentence_dim=config.sentence_dim,
      ),
      image_dim=state_dim,
      output_fn=vision_language.struct_output,
    )

    if config.output_init is not None:
      output_init = hk.initializers.VarianceScaling(scale=config.output_init)
    else:
      output_init = None
    root_vpi_base = ResMlp(config.prediction_blocks, ln=config.ln, name='root_base')
    root_value_fn = PredictionMlp(config.vpi_mlps,
                                  discretizer._num_bins,
                                  ln=config.ln,
                                  output_init=output_init,
                                  name='root_value')
    root_policy_fn = PredictionMlp(config.vpi_mlps,
                                   num_actions, ln=config.ln, output_init=output_init, name='root_policy')
    model_reward_fn = PredictionMlp(config.reward_mlps,
                                    discretizer._num_bins,
                                    ln=config.ln,
                                    output_init=output_init,
                                    name='model_reward')
    if config.seperate_model_nets:
      model_vpi_base = ResMlp(config.prediction_blocks, name='root_model')
      model_value_fn = PredictionMlp(config.vpi_mlps,
                                     discretizer._num_bins,
                                     ln=config.ln,
                                     output_init=output_init,
                                     name='model_value')
      model_policy_fn = PredictionMlp(config.vpi_mlps,
                                       num_actions, ln=config.ln, output_init=output_init, name='model_policy')
    else:
      model_value_fn=root_value_fn
      model_policy_fn=root_policy_fn
      model_vpi_base = root_vpi_base

    # model_compute_r_v = config.action_source in ['value']
    return MuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      discretizer=discretizer,
      observation_fn=observation_fn,
      prep_state_input=concat_embeddings,
      state_fn=hk.LSTM(state_dim),
      transition_fn=Transition(
        channels=res_dim,
        num_blocks=config.transition_blocks,
        action_dim=config.action_dim,
        ln=config.ln),
      root_vpi_base=root_vpi_base,
      root_value_fn=root_value_fn,
      root_policy_fn=root_policy_fn,
      model_vpi_base=model_vpi_base,
      model_reward_fn=model_reward_fn,
      model_value_fn=model_value_fn,
      model_policy_fn=model_policy_fn,
      model_compute_r_v=True,
      discount=config.discount,
      model_combine_state_task=config.model_combine_state_task,
      num_actions=num_actions)

  return make_unrollable_model_network(env_spec, make_core_module)

def make_simple_babyai_networks(
  env_spec: specs.EnvironmentSpec,
  config: MuZeroConfig,
  discretizer: Discretizer) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> MuZeroNetworks:
    state_dim = config.state_dim

    assert config.vision_torso in ('babyai', 'atari', 'muzero')
    if config.vision_torso == 'babyai':
      vision_torso = vision.BabyAIVisionTorso(conv_dim=0, flatten=False)
    elif config.vision_torso == 'atari':
      vision_torso = vision.AtariVisionTorso(conv_dim=0, flatten=False)
    elif config.vision_torso == 'muzero':
      vision_torso = MuZeroVisionTorso(channels=64, num_blocks=6)

    observation_fn = vision_language.Torso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder = language.LanguageEncoder(
          vocab_size=config.vocab_size,
          word_dim=config.word_dim,
          sentence_dim=config.sentence_dim,
      ),
      image_dim=state_dim,
      output_fn=vision_language.struct_output,
    )
    
    if config.output_init is not None:
      output_init = hk.initializers.VarianceScaling(scale=config.output_init)
    else:
      output_init = None
    root_vpi_base = hk.Linear(state_dim, name='root')
    root_value_fn = simple_mlp_muzero.PredictionMlp(
      [state_dim], config.num_bins, output_init=output_init,
      name='root_value')
    root_policy_fn = simple_mlp_muzero.PredictionMlp(
      [state_dim], num_actions, output_init=output_init,
      name='root_policy')
    model_reward_fn = simple_mlp_muzero.PredictionMlp(
      [state_dim], config.num_bins, output_init=output_init,
      name='model_reward')

    if config.seperate_model_nets:
      raise NotImplementedError
      # model_vpi_base = ResMlp(config.prediction_blocks)
      # model_value_fn = simple_mlp_muzero.PredictionMlp(
      #   config.vpi_mlps, config.num_bins, output_init=output_init)
      # model_policy_fn = simple_mlp_muzero.PredictionMlp(
      #   config.vpi_mlps, num_actions, output_init=output_init)
    else:
      model_value_fn=root_value_fn
      model_policy_fn=root_policy_fn
      model_vpi_base = root_vpi_base

    # model_compute_r_v = config.action_source in ['value']
    return MuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      discretizer=discretizer,
      observation_fn=observation_fn,
      prep_state_input=concat_embeddings,
      state_fn=hk.LSTM(state_dim),
      transition_fn=simple_mlp_muzero.Transition(
        num_blocks=2),
      root_vpi_base=root_vpi_base,
      root_value_fn=root_value_fn,
      root_policy_fn=root_policy_fn,
      model_vpi_base=model_vpi_base,
      model_reward_fn=model_reward_fn,
      model_value_fn=model_value_fn,
      model_policy_fn=model_policy_fn,
      model_compute_r_v=True,
      discount=config.discount,
      num_actions=num_actions)

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
