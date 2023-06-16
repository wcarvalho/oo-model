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

import functools
from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils


import haiku as hk
import jax
import jax.numpy as jnp


from modules import vision
from modules import language
from modules import vision_language
from modules import simple_mlp_muzero
from modules.mlp_muzero import PredictionMlp, Transition, ResMlp
from muzero.arch import MuZeroArch
from muzero.types import MuZeroNetworks, TaskAwareRep, RootOutput, ModelOutput
from muzero.config import MuZeroConfig
from muzero.utils import Discretizer, TaskAwareRecurrentFn, scale_gradient, compute_q_values


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
  invalid_actions=None,
  **kwargs) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> MuZeroNetworks:
    """Create MuZero.

    It works as follows
    1. observation_fn encodes (a) image (b) task description.
      output: image embedding, task embedding, prev reward, prev action are output


    Returns:
        MuZeroNetworks: _description_
    """
    state_dim = config.state_dim

    ###########################
    # Setup observation encoders (image + language)
    ###########################
    assert config.vision_torso in ('babyai')
    vision_torso = vision.BabyAIVisionTorso(conv_dim=config.conv_out_dim)

    observation_fn = vision_language.Torso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder=language.LanguageEncoder(
          vocab_size=config.vocab_size,
          word_dim=config.word_dim,
          sentence_dim=config.sentence_dim,
      ),
      image_dim=state_dim,
      task_dim=config.task_dim,
      output_fn=vision_language.struct_output,
    )

    ###########################
    # Setup state function: LSTM 
    ###########################
    state_fn = hk.LSTM(state_dim)
    def combine_hidden_obs(hidden: jnp.ndarray, emb: vision_language.TorsoOutput):
      """After get hidden from LSTM, combine with task from embedding."""
      return TaskAwareRep(rep=hidden, task=emb.task)

    ###########################
    # Setup transition function: ResNet
    ###########################
    def resnet_model(action_onehot: jnp.ndarray, state: TaskAwareRep):
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"
      def _resnet_model(action_onehot, state):
        """ResNet transition model that scales gradient."""
        # action: [A]
        # state: [D]
        out = Transition(
          channels=res_dim,
          num_blocks=config.transition_blocks,
          ln=config.ln)(action_onehot, state)
        if config.scale_grad:
          out = scale_gradient(out, config.scale_grad)
        return out, out
      if action_onehot.ndim == 2:
        _resnet_model = jax.vmap(_resnet_model)
      return _resnet_model(action_onehot, state)

    res_dim = config.resnet_transition_dim or state_dim
    # transition gets task from state and stores in state
    transition_fn = TaskAwareRecurrentFn(
      get_task=lambda inputs, state: state.task,
      prep_state=lambda state: state.state,  # get state-vector from TaskAwareRep
      couple_state_task=True,
      core=resnet_model,
    )

    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    if config.output_init is not None:
      output_init = hk.initializers.VarianceScaling(scale=config.output_init)
    else:
      output_init = None
    root_vpi_base = ResMlp(config.prediction_blocks, ln=config.ln, name='root_base')
    root_value_fn = PredictionMlp(config.vpi_mlps,
                                  config.num_bins,
                                  ln=config.ln,
                                  output_init=output_init,
                                  name='root_value')
    root_policy_fn = PredictionMlp(config.vpi_mlps,
                                   num_actions,
                                   ln=config.ln,
                                   output_init=output_init,
                                   name='root_policy')
    model_reward_fn = PredictionMlp(config.reward_mlps,
                                    config.num_bins,
                                    ln=config.ln,
                                    output_init=output_init,
                                    name='model_reward')

    if config.seperate_model_nets:
      model_vpi_base = ResMlp(config.prediction_blocks, name='root_model')
      model_value_fn = PredictionMlp(config.vpi_mlps,
                                     config.num_bins,
                                     ln=config.ln,
                                     output_init=output_init,
                                     name='model_value')
      model_policy_fn = PredictionMlp(config.vpi_mlps,
                                       num_actions, ln=config.ln, output_init=output_init, name='model_policy')
    else:
      model_value_fn = root_value_fn
      model_policy_fn = root_policy_fn
      model_vpi_base = root_vpi_base

    def root_predictor(state: TaskAwareRep):
      assert state.task.ndim in (1, 2), "should be [D] or [B, D]"
      def _root_predictor(state: TaskAwareRep):
        state_ = state.state
        state_ = root_vpi_base(state_)

        policy_logits = root_policy_fn(state_)
        value_logits = root_value_fn(state_)

        return RootOutput(
            state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
        )
      if state.task.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)

    def model_predictor(state: TaskAwareRep):
      assert state.task.ndim in (1,2), "should be [D] or [B, D]"
      def _model_predictor(state: TaskAwareRep):
        state_ = state.state
        reward_logits = model_reward_fn(state_)

        state_ = model_vpi_base(state_)
        policy_logits = model_policy_fn(state_)
        value_logits = model_value_fn(state_)

        return ModelOutput(
          new_state=state,
          value_logits=value_logits,
          policy_logits=policy_logits,
          reward_logits=reward_logits,
        )
      if state.task.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)



    return MuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      observation_fn=observation_fn,
      prep_state_input=concat_embeddings,
      state_fn=state_fn,
      combine_hidden_obs=combine_hidden_obs,
      transition_fn=transition_fn,
      root_pred_fn=root_predictor,
      model_pred_fn=model_predictor,
      invalid_actions=invalid_actions)

  return make_network(config=config,
                      environment_spec=env_spec,
                      make_core_module=make_core_module,
                      invalid_actions=invalid_actions,
                      **kwargs)

def make_network(
        environment_spec: specs.EnvironmentSpec,
        make_core_module: Callable[[], hk.RNNCore],
        config,
        invalid_actions = None,
        discretizer: Discretizer = None,
        ) -> MuZeroNetworks:
  """Builds a MuZeroNetworks from a hk.Module factory."""

  dummy_observation = jax_utils.zeros_like(environment_spec.observations)
  dummy_action = jnp.array(0)

  def make_unrollable_network_functions():
    network = make_core_module()

    def init() -> Tuple[NetworkOutput, RecurrentState]:
      out, _ = network(dummy_observation, network.initial_state(None))
      return network.apply_model(out.state, dummy_action)

    apply = network.__call__

    q_values_fn = None
    if config.action_source == 'value':
      assert discretizer is not None, 'give discretizer when computing q-values'
      q_values_fn = functools.partial(
        compute_q_values,
        discretizer=discretizer,
        invalid_actions=invalid_actions,
        num_actions=environment_spec.actions.num_values,
        apply_model=network.apply_model,
        discount=config.discount)

    return init, (apply,
                  network.unroll,
                  network.initial_state,
                  network.apply_model,
                  network.unroll_model,
                  q_values_fn)

  # Transform and unpack pure functions
  f = hk.multi_transform(make_unrollable_network_functions)
  apply, unroll, initial_state_fn, apply_model, unroll_model, q_values_fn = f.apply

  def init_recurrent_state(key: jax_types.PRNGKey,
                           batch_size: Optional[int]) -> RecurrentState:
    # TODO(b/244311990): Consider supporting parameterized and learnable initial
    # state functions.
    no_params = None
    return initial_state_fn(no_params, key, batch_size)

  return MuZeroNetworks(
      unroll_init=f.init,
      apply=apply,
      unroll=unroll,
      apply_model=apply_model,
      unroll_model=unroll_model,
      init_recurrent_state=init_recurrent_state,
      compute_q_values=q_values_fn)
