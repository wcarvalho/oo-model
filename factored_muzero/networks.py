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

"""Factored MuZero Networks."""

from typing import Callable, Optional, Tuple

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils


import haiku as hk
import jax
import jax.numpy as jnp


from modules import vision
from modules import language
from modules import vision_language
from modules.mlp_muzero import PredictionMlp, Transition, ResMlp
from muzero.arch import MuZeroArch
from muzero.types import MuZeroNetworks, TaskAwareState
from muzero.config import MuZeroConfig
from muzero.utils import Discretizer, TaskAwareRecurrentFn, scale_gradient
from muzero.networks import make_network


from factored_muzero import attention
from factored_muzero import encoder

# MuZeroNetworks = networks_lib.UnrollableNetwork
NetworkOutput = networks_lib.NetworkOutput
RecurrentState = networks_lib.RecurrentState

def make_save_input(embedding: vision_language.TorsoOutput):
  return attention.SaviInputs(
    image=embedding.image,
    task=embedding.task
  )


def make_babyai_networks(
  env_spec: specs.EnvironmentSpec,
  config: MuZeroConfig) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> MuZeroNetworks:
    state_dim = config.state_dim
    res_dim = config.resnet_transition_dim or state_dim

    ###########################
    # Setup observation encoders (image + language)
    ###########################
    vision_torso = encoder.PositionEncodingTorso(
      img_embed=vision.BabyAIVisionTorso(conv_dim=0, flatten=False),
      pos_embed=encoder.PositionEmbedding(
        embedding_type=config.embedding_type,
        update_type=config.update_type,
        output_transform=encoder.Mlp(
          # default settings from paper
          mlp_layers=[64],
          layernorm='pre',
        )
      )
    )

    observation_fn = vision_language.Torso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      flatten_image=False,
      task_encoder=language.LanguageEncoder(
          vocab_size=config.vocab_size,
          word_dim=config.word_dim,
          sentence_dim=config.sentence_dim,
      ),
      task_dim=config.task_dim,
      output_fn=vision_language.struct_output,
    )
    ###########################
    # Setup state function: SaVi 
    ###########################
    # during state unroll, rnn gets task from inputs and stores in state
    state_fn = attention.SlotAttention(
        num_iterations=config.savi_iterations,
        qkv_size=config.state_qkv_size,
        num_slots=config.num_slots,
    )
    def combine_hidden_obs(hidden: jnp.ndarray, emb: vision_language.TorsoOutput):
      """After get hidden from LSTM, combine with task from embedding."""
      return TaskAwareState(state=hidden, task=emb.task)

    ###########################
    # Setup transition function: transformer
    ###########################
    def transformer_model(action_onehot, state):
      def _transformer_model(action_onehot, state):
        """Just make long sequence of state slots + action."""
        # action: [A]
        # state (slots): [N, D]
        action_w_init = hk.initializers.TruncatedNormal()
        encoded_action = hk.Linear(state.shape[-1],
                                  w_init=action_w_init,
                                  with_bias=False)(action_onehot)
        encoded_action = jnp.expand_dims(encoded_action, 0)
        queries = jnp.concatenate((state, encoded_action))

        out = attention.Transformer(
            num_heads=config.slot_tran_heads,
            qkv_size=config.slot_tran_qkv_size,
            mlp_size=config.slot_tran_mlp_size,
            num_layers=config.transition_blocks,
            name='transition')(queries)
        out = out[:-1]  # remove action representation
        if config.scale_grad:
          out = scale_gradient(out, config.scale_grad)
        return out, out
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"
      if action_onehot.ndim == 2:
        _transformer_model = jax.vmap(_transformer_model)
      return _transformer_model(action_onehot, state)

    # transition gets task from state and stores in state
    transition_fn = TaskAwareRecurrentFn(
        get_task=lambda _, state: state.task,
        prep_state=lambda state: state.state,  # get state-vector from TaskAwareState
        couple_state_task=True,
        core=transformer_model
    )

    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    def make_transformer(name, num_layers):
      return attention.Transformer(
          num_heads=config.slot_pred_heads,
          qkv_size=config.slot_pred_qkv_size,
          mlp_size=config.slot_pred_mlp_size,
          num_layers=num_layers,
          name=name)

    def make_pred_fn(name, num_preds):
      return PredictionMlp(
        config.vpi_mlps,
        num_preds,
        ln=config.ln,
        name=name)

    assert config.seperate_model_nets == False, 'need to redo function for this'
    # root
    pre_blocks_base = max(1, config.prediction_blocks//2)
    base1 = make_transformer('base1', num_layers=pre_blocks_base)
    base2 = make_transformer('base2', num_layers=pre_blocks_base)
    task_attn_value = attention.GeneralMultiHeadAttention(
        num_heads=config.slot_pred_heads,
        key_size=config.slot_pred_qkv_size,
        w_init=hk.initializers.VarianceScaling(2.0),
    )
    policy_fn = make_pred_fn(name='policy', num_preds=num_actions)
    value_fn = make_pred_fn(name='value', num_preds=config.num_bins)

    # model
    task_attn_reward = attention.GeneralMultiHeadAttention(
        num_heads=config.slot_pred_heads,
        key_size=config.slot_pred_qkv_size,
        w_init=hk.initializers.VarianceScaling(2.0),
    )
    reward_fn = make_pred_fn(name='reward', num_preds=config.num_bins)
    task_projection = hk.Linear(config.slot_tran_qkv_size, with_bias=False)

    def state_task_to_queries(state: TaskAwareState):
      # state: [D] or [B, D]
      state_hidden = state.state
      task = task_projection(state.task)  # [D]
      task = jnp.expand_dims(task, 0)
      return jnp.concatenate((task, state_hidden))
    def root_predictor(state: TaskAwareState):
      def _root_predictor(state: TaskAwareState):
        queries = state_task_to_queries(state)
        queries = base1(queries)  # [N, D]
        queries = base2(queries)  # [N, D]

        # [1, D]
        # use None to make query [1, D]. Needed for attention on [N, D] queries.
        task_attn = task_attn_value(query=queries[0][None], key=queries)[0]
        task_attn = task_attn + queries[0]
        policy_logits = policy_fn(task_attn)
        value_logits = value_fn(task_attn)
        return policy_logits, value_logits
      assert state.task.ndim in (1,2), "should be [D] or [B, D]"
      if state.task.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)

    def model_predictor(state: TaskAwareState):
      def _model_predictor(state: TaskAwareState):
        queries = state_task_to_queries(state)
        queries = base1(queries)

        # use None to make query [1, D]. Needed for attention on [N, D] queries.
        task_attn = task_attn_reward(query=queries[0][None], key=queries)[0]  # [N, D]
        task_attn = task_attn + queries[0]
        reward_logits = reward_fn(task_attn)

        queries = base2(queries)
        # use None to make query [1, D]. Needed for attention on [N, D] queries.
        task_attn = task_attn_value(query=queries[0][None], key=queries)[0]  # [N, D]
        task_attn = task_attn + queries[0]
        policy_logits = policy_fn(task_attn)
        value_logits = value_fn(task_attn)
        return reward_logits, policy_logits, value_logits
      assert state.task.ndim in (1,2), "should be [D] or [B, D]"
      if state.task.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)

    return MuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      observation_fn=observation_fn,
      prep_state_input=make_save_input,
      state_fn=state_fn,
      combine_hidden_obs=combine_hidden_obs,
      transition_fn=transition_fn,
      root_pred_fn=root_predictor,
      model_pred_fn=model_predictor)

  return make_network(env_spec, make_core_module)
