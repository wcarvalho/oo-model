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
import functools

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
from muzero.utils import Discretizer, TaskAwareRecurrentFn, scale_gradient
from muzero.networks import make_network


from factored_muzero.config import FactoredMuZeroConfig as MuZeroConfig
from factored_muzero import attention
from factored_muzero import encoder
from factored_muzero import gates

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
  num_spatial_vectors: dict,
  config: MuZeroConfig) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values
  def make_core_module() -> MuZeroNetworks:
    ###########################
    # Setup observation encoders (image + language)
    ###########################
    w_init_attn = hk.initializers.VarianceScaling(config.w_init_attn)
    vision_torso = encoder.PositionEncodingTorso(
      img_embed=vision.BabyAIVisionTorso(conv_dim=config.conv_out_dim,
                                         flatten=False),
      pos_embed=encoder.PositionEmbedding(
        embedding_type=config.embedding_type,
        update_type=config.update_type,
        w_init=w_init_attn,
        output_transform=encoder.Mlp(
          # default settings from paper
          mlp_layers=[64],
          w_init=w_init_attn,
          layernorm=config.pos_layernorm,
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
      w_init=w_init_attn,
      output_fn=vision_language.struct_output,
    )
    ###########################
    # Setup state function: SaVi 
    ###########################
    # set default transition sizes based on state function
    slot_tran_heads = config.slot_tran_heads
    slot_tran_qkv_size = config.slot_tran_qkv_size or config.slot_size
    slot_tran_mlp_size = config.slot_tran_mlp_size or slot_tran_qkv_size

    # set default prediction sizes based on transition function
    slot_pred_heads = config.slot_pred_heads or config.slot_tran_heads
    slot_pred_qkv_size = config.slot_pred_qkv_size or slot_tran_qkv_size
    slot_pred_mlp_size = config.slot_pred_mlp_size or slot_tran_mlp_size

    # during state unroll, rnn gets task from inputs and stores in state
    assert config.gru_init in ('orthogonal', 'default')
    if config.gru_init == 'orthogonal':
      # used in Slot Attention paper
      gru_w_i_init = hk.initializers.Orthogonal()
    elif config.gru_init == 'default':
      # from haiku
      gru_w_i_init = None
    state_fn = attention.SlotAttention(
        num_iterations=config.savi_iterations,
        qkv_size=config.slot_size,
        num_slots=config.num_slots,
        num_spatial=num_spatial_vectors,
        w_init=w_init_attn,
        gru_w_i_init=gru_w_i_init,
        use_task=config.slots_use_task,
    )
    def combine_hidden_obs(hidden: jnp.ndarray, emb: vision_language.TorsoOutput):
      """After get hidden from SlotAttention, combine with task from embedding."""
      return TaskAwareState(state=hidden, task=emb.task)

    ###########################
    # Setup gating functions
    ###########################
    assert config.gating in ('sum', 'gru', 'sigtanh')
    # following options from GTRxL: https://arxiv.org/pdf/1910.06764.pdf
    get_gate_factory = functools.partial(
      gates.get_gate_factory,
      b_init=hk.initializers.Constant(config.b_init_attn),
      w_init=w_init_attn)

    ###########################
    # Setup transition function: transformer
    ###########################

    def transformer_model(action_onehot: jnp.ndarray, 
                          state: TaskAwareState):
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"
      def _transformer_model(action_onehot, state):
        """Just make long sequence of state slots + action."""
        # action: [A]
        # state (slots): [N, D]
        action_w_init = hk.initializers.TruncatedNormal()
        encoded_action = hk.Linear(state.shape[-1],
                                  w_init=action_w_init,
                                  with_bias=False)(action_onehot)
        if config.action_as_factor:
          # [N+1, D]
          queries = jnp.concatenate((state, encoded_action[None]))
        else:
          join = lambda a,b: a+b
          # [N, D]
          queries = jax.vmap(join, (0, None))(state, encoded_action)

        out = attention.Transformer(
            num_heads=slot_tran_heads,
            qkv_size=slot_tran_qkv_size,
            mlp_size=slot_tran_mlp_size,
            num_layers=config.transition_blocks,
            pre_norm=config.pre_norm,
            gate_factory=get_gate_factory(config.gating),
            w_init=w_init_attn,
            name='transition')(queries)
        if config.action_as_factor:
          out = out[:-1]  # remove action representation
        if config.scale_grad:
          out = scale_gradient(out, config.scale_grad)
        return out, out
      if action_onehot.ndim == 2:
        _transformer_model = jax.vmap(_transformer_model)
      return _transformer_model(action_onehot, state)

    # transition gets task from state and stores in state
    transition_fn = TaskAwareRecurrentFn(
        get_task=lambda inputs, state: state.task,
        prep_state=lambda state: state.state,  # get state-vector from TaskAwareState
        couple_state_task=True,
        core=hk.to_module(transformer_model)(name='transformer_model'),
    )

    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    def make_transformer(name: str, num_layers: int):
      return attention.Transformer(
          num_heads=slot_pred_heads,
          qkv_size=slot_pred_qkv_size,
          mlp_size=slot_pred_mlp_size,
          num_layers=num_layers,
          w_init=w_init_attn,
          pre_norm=config.pre_norm,
          gate_factory=get_gate_factory(config.gating),
          name=name)

    w_init_out = w_init_attn if config.share_w_init_out else None
    def make_pred_fn(name: str, sizes: Tuple[int], num_preds: int):
      assert config.pred_head in ('muzero', 'haiku_mlp')
      if config.pred_head == 'muzero':
        return PredictionMlp(
          sizes, num_preds,
          w_init=w_init_out,
          output_init=w_init_out,
          ln=config.ln,
          name=name)
      elif config.pred_head == 'haiku_mlp':
        return hk.nets.MLP(tuple(sizes) + (num_preds,),
                           w_init=w_init_out,
                           name=name)
    def make_pred_input_module(selection, attn_head, gate, name='pred_input'):
      """This creates a function which aggregates slot information tto single vector.

      attention: use task-vector to select slot information
      mean: mean of slot information
      """
      def pred_input_module(slots, task_query):
        # slots: [N, D]
        # task_query: [D]
        if selection == 'attention':
          # [D]
          out = attn_head(query=task_query[None],  # convert to [1, D] for attention
                          key=slots)[0]
          out = gate(task_query, out)
          return hk.LayerNorm(
            axis=(-1), create_scale=True, create_offset=True)(out)
        elif selection == 'slot_mean':
          return slots.mean(0)
        elif selection == 'task_query':
          return task_query
      return hk.to_module(pred_input_module)(name=name)

    assert config.seperate_model_nets == False, 'need to redo function for this'
    # root
    pre_blocks_base = max(1, config.prediction_blocks//2)
    transformer1 = make_transformer(
      'transformer1', num_layers=pre_blocks_base)
    transformer2 = make_transformer(
      'transformer2', num_layers=pre_blocks_base)

    policy_input_module = make_pred_input_module(
      name='policy_input_module',
      selection=config.pred_input_selection,
      gate=get_gate_factory(config.pred_gate)(),
      attn_head=attention.GeneralMultiHeadAttention(
        num_heads=slot_pred_heads,
        key_size=slot_pred_qkv_size,
        model_size=slot_pred_qkv_size,
        w_init=w_init_attn),
    )
    policy_fn = make_pred_fn(
      name='policy', sizes=config.vpi_mlps, num_preds=num_actions)
    value_fn = make_pred_fn(
      name='value', sizes=config.vpi_mlps, num_preds=config.num_bins)

    # model
    reward_input_module = make_pred_input_module(
      name='reward_input_module',
      selection=config.pred_input_selection,
      gate=get_gate_factory(config.pred_gate)(),
      attn_head=attention.GeneralMultiHeadAttention(
          num_heads=slot_pred_heads,
          key_size=slot_pred_qkv_size,
          model_size=slot_pred_qkv_size,
          w_init=w_init_attn),
    )
    reward_fn = make_pred_fn(
      name='reward', sizes=config.reward_mlps, num_preds=config.num_bins)
    task_projection = hk.Linear(slot_tran_qkv_size,
                                w_init=w_init_attn,
                                with_bias=False)

    def state_task_to_queries(state: TaskAwareState):
      # state (slot): [N, D]
      state_hidden = state.state
      task = task_projection(state.task)  # [D]
      task = jnp.expand_dims(task, 0)  # [1, D]

      return jnp.concatenate((task, state_hidden))  #[N+1, D]

    def root_predictor(state: TaskAwareState):
      def _root_predictor(state: TaskAwareState):
        queries = state.state
        if config.pred_input_selection == 'attention':
          queries = state_task_to_queries(state)
        queries = transformer1(queries)  # [N, D]
        queries = transformer2(queries)  # [N, D]

        # [1, D]
        # use None to make query [1, D]. Needed for attention on [N, D] queries.
        policy_input = policy_input_module(slots=queries[1:],
                                           task_query=queries[0])
        policy_logits = policy_fn(policy_input)
        value_logits = value_fn(policy_input)
        return policy_logits, value_logits
      assert state.task.ndim in (1,2), "should be [D] or [B, D]"
      if state.task.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)

    def model_predictor(state: TaskAwareState):
      def _model_predictor(state: TaskAwareState):
        queries = state.state
        if config.pred_input_selection == 'attention':
          queries = state_task_to_queries(state)
        queries = transformer1(queries)

        # use None to make query [1, D]. Needed for attention on [N, D] queries.
        reward_input = reward_input_module(slots=queries[1:],
                                           task_query=queries[0])

        reward_logits = reward_fn(reward_input)

        queries = transformer2(queries)
        # use None to make query [1, D]. Needed for attention on [N, D] queries.
        policy_input = policy_input_module(slots=queries[1:],
                                           task_query=queries[0])
        policy_logits = policy_fn(policy_input)
        value_logits = value_fn(policy_input)
        return reward_logits, policy_logits, value_logits
      assert state.task.ndim in (1,2), "should be [D] or [B, D]"
      if state.task.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)

    root_pred_fn = hk.to_module(root_predictor)(name='root_predictor')
    model_pred_fn = hk.to_module(model_predictor)(name='model_predictor')
    arch = MuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      observation_fn=observation_fn,
      prep_state_input=make_save_input,
      state_fn=state_fn,
      combine_hidden_obs=combine_hidden_obs,
      transition_fn=transition_fn,
      root_pred_fn=root_pred_fn,
      model_pred_fn=model_pred_fn)

    return arch

  return make_network(env_spec, make_core_module)
