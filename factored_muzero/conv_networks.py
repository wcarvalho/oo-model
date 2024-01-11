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

from typing import Callable, Optional, NamedTuple, Tuple, Union
import functools

from acme import specs
from acme import types as acme_types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils

import chex
import haiku as hk
import jax
import jax.numpy as jnp


from modules import vision
from modules import language
from modules import vision_language
from modules.mlp_muzero import PredictionMlp, ResMlp
from modules.conv_muzero import Transition as ConvTransition
from muzero.arch import MuZeroArch
from muzero.types import MuZeroNetworks
from muzero.utils import scale_gradient
from muzero.networks import make_network


from factored_muzero.config import FactoredMuZeroConfig as MuZeroConfig
from factored_muzero import attention
from factored_muzero import encoder
from factored_muzero import gates
from factored_muzero.types import RootOutput, ModelOutput, TaskAwareSaviState

# MuZeroNetworks = networks_lib.UnrollableNetwork
Array = acme_types.NestedArray

class ConvSaviState(NamedTuple):
  conv: jax.Array
  factors: jnp.ndarray
  factor_states: jnp.ndarray
  attn: Optional[jnp.ndarray] = None


class FactoredMuZeroArch(MuZeroArch):
  """Factored MuZero Architecture.

  Observation function: CNN.
  State function: SaVi.
  World Model: TaskAwareRecurrentFn[Transformer]
  """
  def initial_state(self, batch_size: Optional[int] = None,
                    **unused_kwargs) -> attention.SaviState:
    return super().initial_state(batch_size, **unused_kwargs)

  def __call__(
      self,
      inputs: attention.SaviInputs,
      state: attention.SaviState
  ) -> Tuple[RootOutput, attention.SaviState]:
    """Update slots with observation information. No batch dim."""
    return super().__call__(inputs, state)
  
  def unroll(
      self,
      input_sequence: attention.SaviInputs,
      state: attention.SaviState,
  ) -> Tuple[RootOutput, attention.SaviState]:
    """Unroll savi over input_sequence. State has 1 time-step."""
    return super().unroll(input_sequence, state)

  def apply_model(
      self,
      state: TaskAwareSaviState,
      action: Array,
  ) -> Tuple[ModelOutput, TaskAwareSaviState]:
    """Apply transformer model over slots. Can have batch_dim."""
    return super().apply_model(state, action)

  def unroll_model(
      self,
      state: TaskAwareSaviState,
      action_sequence: Array,
  ) -> Tuple[ModelOutput, TaskAwareSaviState]:
    """Unrolls transformer model over action_sequence. No batch dim."""
    return super().unroll_model(state, action_sequence)

  def decode_observation(self, state: attention.SaviState):
    """Apply decoder to state."""
    return super().decode_observation(state)

class ConvSlotMemory(hk.RNNCore):

  def __init__(
      self,
      conv_state_fn: hk.Conv2DLSTM,
      factors_state_fn: attention.SlotAttention,
      pos_embedder: encoder.PositionEmbedding,
      name: Optional[str] = 'ConvSlotMemory'):
    """Constructs an LSTM.

    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.conv_state_fn = conv_state_fn
    self.factors_state_fn = factors_state_fn
    self.pos_embedder = pos_embedder

  def __call__(
      self,
      inputs: vision_language.TorsoOutput,
      prev_state: ConvSaviState,
  ) -> Tuple[ConvSaviState, ConvSaviState]:

    channels = inputs.image.shape[-1]

    #-------------------
    # add (action, reward, task) info to obs tensor
    #-------------------
    encoded_inputs = hk.Linear(
      channels, 
      w_init=hk.initializers.TruncatedNormal(),
      with_bias=False)(jnp.concatenate(
        (inputs.action, inputs.reward, inputs.task), axis=-1))
    # [..., D] --> [..., 1, 1, D]
    encoded_inputs = jnp.expand_dims(encoded_inputs, axis=-2)
    encoded_inputs = jnp.expand_dims(encoded_inputs, axis=-2)

    # [H, W, C] + [1, 1, D], i.e. broadcast across H,W
    conv_input = inputs.image + encoded_inputs

    #-------------------
    # apply ConvLTSM
    #-------------------
    conv_hidden, conv_state = self.conv_state_fn(
      conv_input, prev_state.conv)

    #-------------------
    # embed positions and then apply slot attention
    #-------------------
    factors_state, _ = self.factors_state_fn(
      image=self.pos_embedder(conv_hidden),
      state=prev_state)

    new_hidden = ConvSaviState(
      conv=conv_hidden,
      **factors_state._asdict())

    new_state = ConvSaviState(
      conv=conv_state,
      **factors_state._asdict())

    return new_hidden, new_state

  def initial_state(self, batch_size: Optional[int]) -> ConvSaviState:
    return ConvSaviState(
      conv=self.conv_state_fn.initial_state(batch_size),
      **self.factors_state_fn.initial_state(batch_size)._asdict()
    )

def make_observation_fn(
    config, env_spec):
  num_actions = env_spec.actions.num_values

  def observation_fn(inputs):
      if config.vision_torso == 'babyai':
        img_embed = vision.BabyAIVisionTorso(
              conv_dim=config.conv_out_dim, flatten=False)
        img_embed = vision.SaviVisionTorso(
          features=[128, 128, 128],
          kernel_size=[(8, 8), (3, 3), (1,1)],
          strides=[(8, 8), 1, 1],
          activation_fns=[lambda x:x, jax.nn.relu, jax.nn.relu],
          )
      elif config.vision_torso == 'motts2019':
        img_embed = vision.SaviVisionTorso(
          features=[32, 64, 128],
          kernel_size=[(4, 4), (4, 4), (3, 3)],
          strides=[(4, 4), (2, 2), (1, 1)],
          )
      elif config.vision_torso == 'babyai_patches':
        img_embed = vision.SaviVisionTorso(
          features=[128, 128],
          kernel_size=[(8, 8), (1, 1)],
          strides=[(8, 8), (1, 1)],
          )
      elif config.vision_torso == 'savi':
        img_embed = vision.SaviVisionTorso(
          features=[32, 32, 32, 32],
          kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
          strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
          )
      elif config.vision_torso == 'savi_small':
        img_embed = vision.SaviVisionTorso(
          features=[32, 32, 32],
          kernel_size=[(5, 5), (5, 5), (5, 5)],
          strides=[(1, 1), (1, 1), (1, 1)],
          )
      else:
        raise NotImplementedError(config.vision_torso)

      task_encoder = language.LanguageEncoder(
              vocab_size=config.vocab_size,
              word_dim=config.word_dim,
              sentence_dim=config.sentence_dim,
          )
      return vision_language.Torso(
          num_actions=num_actions,
          vision_torso=img_embed,
          flatten_image=False,
          task_encoder=task_encoder,
          task_dim=config.task_dim,
          output_fn=vision_language.struct_output,
      )(inputs)
  return observation_fn

def make_transition_fn(
    config,
    factors_state_fn: attention.SlotAttention,
    pos_embedder: encoder.PositionEmbedding,
    ):

    def transition_fn(action_onehot: Array,
                          state: TaskAwareSaviState):
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transition_fn(
          action_onehot: Array,
          state: ConvSaviState,
      ) -> Tuple[ConvSaviState,
                 ConvSaviState]:
        """Just make long sequence of state slots + action."""

        #-------------------
        # apply ResNet transition model
        #-------------------
        new_conv_state = ConvTransition(
          channels=state.conv.shape[-1],
          num_blocks=config.transition_blocks)(
            action_onehot, state.conv)

        #-------------------
        # apply slot attention on top
        #-------------------
        new_factors, _ = factors_state_fn(
          image=pos_embedder(new_conv_state),
          state=state)

        new_state = ConvSaviState(
          conv=new_conv_state,
          **new_factors._asdict())

        if config.scale_grad:
          scale = functools.partial(scale_gradient,
                                    scale=config.scale_grad)
          new_state = jax.tree_map(lambda a: scale(a), new_state)

        return new_state, new_state

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)

    # transition gets task from state and stores in state
    return hk.to_module(transition_fn)(name='transition_fn')

def make_prediction_function(
        config, env_spec):

  num_actions = env_spec.actions.num_values

  w_init_attn = hk.initializers.VarianceScaling(config.w_init_attn)

  get_gate_factory = functools.partial(
      gates.get_gate_factory,
      b_init=hk.initializers.Constant(config.b_init_attn),
      w_init=w_init_attn)


  pred_gate = config.pred_gate or config.model_gate

  def make_transformer(name: str, num_layers: int):
    return attention.Transformer(
        num_heads=config.slot_tran_heads,
        qkv_size=config.slot_size,
        mlp_size=config.slot_size,
        num_layers=num_layers,
        w_init=w_init_attn,
        pre_norm=config.pre_norm,
        out_mlp=config.pred_out_mlp,
        gate_factory=get_gate_factory(pred_gate),
        name=name)

  def make_pred_fn(name: str, sizes: Tuple[int], num_preds: int):
    assert config.pred_head in ('muzero', 'haiku_mlp')
    if config.pred_head == 'muzero':
      return PredictionMlp(
          sizes, num_preds,
          name=name)
    elif config.pred_head == 'haiku_mlp':
      return hk.nets.MLP(tuple(sizes) + (num_preds,),
                          name=name)

  # root
  vpi_base = ResMlp(config.prediction_blocks,
                    ln=config.ln, name='pred_root_base')
  pred_transformer = make_transformer(
      'pred_root_base',
      num_layers=config.prediction_blocks)

  policy_fn = make_pred_fn(
      name='pred_root_policy', sizes=config.vpi_mlps, num_preds=num_actions)
  value_fn = make_pred_fn(
      name='pred_root_value', sizes=config.vpi_mlps, num_preds=config.num_bins)
  reward_fn = make_pred_fn(
      name='pred_model_reward', sizes=config.reward_mlps, num_preds=config.num_bins)
  weight_fn = make_pred_fn(
      name='pred_weights', sizes=config.vpi_mlps, num_preds=1)

  context_projection = hk.Linear(
      config.slot_size,
      w_init=w_init_attn,
      with_bias=False,
      name='context_projection')

  if config.seperate_model_nets:
    model_pred_transformer = make_transformer(
      'pred_model_base',
      num_layers=config.prediction_blocks)
    model_vpi_base = ResMlp(config.prediction_blocks, ln=config.ln, name='pred_model_base')
    model_weight_fn = make_pred_fn(
      name='pred_weights', sizes=config.vpi_mlps, num_preds=1)
    model_value_fn = make_pred_fn(
      name='pred_model_value', sizes=config.vpi_mlps, num_preds=config.num_bins)
    model_policy_fn = make_pred_fn(
      name='pred_model_policy', sizes=config.vpi_mlps, num_preds=num_actions)
    model_context_projection = hk.Linear(
      config.slot_size,
      w_init=w_init_attn,
      with_bias=False,
      name='model_context_projection')

  else:
    model_pred_transformer = pred_transformer
    model_value_fn = value_fn
    model_policy_fn = policy_fn
    model_vpi_base = vpi_base
    model_weight_fn = weight_fn
    model_context_projection = context_projection

  def factors_attn(state_rep,
                   pred_transformer_,
                   context_projection_,
                   weight_fn_):
    """
    factors: [N or N+1,D]. N+1 if have a factor for full image.
    task: [D]
    Concatenate (task + full image factor) with every factor
    """
    factors = state_rep.factors  # [N, D]
    concat = lambda a, b: jnp.concatenate((a, b))
    if config.context_slot_dim:
      context = context_projection_(
        hk.Flatten(-3)(state_rep.conv))  # [D]
      # concatenate context with every factor, [N, 2D]
      pred_input = jax.vmap(concat, (None, 0), (0))(context, factors)
      pred_input = pred_transformer_(queries=pred_input)
      attn_output = None
    else:
      pred_input = pred_transformer_(queries=factors)
      attn_output = pred_input
    pred_input = pred_input.factors

    if config.learned_weights == 'softmax':
      weights = weight_fn_(pred_input)  # [N, 1]
      weights = jax.nn.softmax(weights, axis=-2)
    elif config.learned_weights == 'none':
      nfactors = len(factors)
      weights = jnp.ones(nfactors)/nfactors
      weights = weights[:, None]
    else:
      raise NotImplementedError

    return pred_input, weights, attn_output

  def weighted_average(x: jnp.ndarray,
                        weights: jnp.ndarray):
    """x: [..., num_factors, num_predictions]"""
    return (x*weights).sum(axis=-2)

  def root_predictor(state_rep: TaskAwareSaviState):
    def _root_predictor(state_rep: TaskAwareSaviState):
      if config.num_slots > 1:
        pred_input, weights, attn_outputs = factors_attn(
          state_rep,
          pred_transformer_=pred_transformer,
          context_projection_=context_projection,
          weight_fn_=weight_fn)

        if config.share_pred_base:
          pred_input = vpi_base(pred_input)
        factor_policy_logits = policy_fn(pred_input)
        factor_value_logits = value_fn(pred_input)

        policy_logits = weighted_average(
            factor_policy_logits, weights)
        value_logits = weighted_average(
            factor_value_logits, weights)
      else:
        attn_outputs = None
        pred_input = state_rep.factors
        factor_policy_logits = policy_fn(pred_input)
        factor_value_logits = value_fn(pred_input)

        policy_logits = factor_policy_logits[0]
        value_logits = factor_value_logits[0]

      return RootOutput(
          pred_attn_outputs=attn_outputs,
          state=state_rep,
          factor_policy_logits=factor_policy_logits,
          factor_value_logits=factor_value_logits,
          value_logits=value_logits,
          policy_logits=policy_logits,
      )
    assert state_rep.conv.ndim in (3,4), "with or without batch dim"
    if state_rep.conv.ndim == 4:
      _root_predictor = jax.vmap(_root_predictor)
    return _root_predictor(state_rep)

  def model_predictor(state_rep: TaskAwareSaviState):
    def _model_predictor(state_rep: TaskAwareSaviState):

      if config.num_slots > 1:
        pred_input, weights, attn_outputs = factors_attn(
          state_rep,
          pred_transformer_=model_pred_transformer,
          context_projection_=model_context_projection,
          weight_fn_=model_weight_fn)

        factor_reward_logits = reward_fn(pred_input)
        if config.share_pred_base:
          pred_input = model_vpi_base(pred_input)
        factor_policy_logits = model_policy_fn(pred_input)
        factor_value_logits = model_value_fn(pred_input)

        reward_logits = weighted_average(
            factor_reward_logits, weights)
        policy_logits = weighted_average(
            factor_policy_logits, weights)
        value_logits = weighted_average(
            factor_value_logits, weights)
      else:
        attn_outputs = None
        pred_input = state_rep.factors
        factor_reward_logits = reward_fn(pred_input)
        if config.share_pred_base:
          pred_input = model_vpi_base(pred_input)
        factor_policy_logits = model_policy_fn(pred_input)
        factor_value_logits = model_value_fn(pred_input)

        reward_logits = factor_reward_logits[0]
        policy_logits = factor_policy_logits[0]
        value_logits = factor_value_logits[0]

      return ModelOutput(
          pred_attn_outputs=attn_outputs,
          new_state=state_rep,
          factor_reward_logits=factor_reward_logits,
          factor_policy_logits=factor_policy_logits,
          factor_value_logits=factor_value_logits,
          value_logits=value_logits,
          policy_logits=policy_logits,
          reward_logits=reward_logits,
      )
    assert state_rep.conv.ndim in (3, 4), "with or without batch dim"
    if state_rep.conv.ndim == 4:
      _model_predictor = jax.vmap(_model_predictor)
    return _model_predictor(state_rep)

  return root_predictor, model_predictor


def make_babyai_networks(
  env_spec: specs.EnvironmentSpec,
  config: MuZeroConfig,
  invalid_actions=None,
  agent_name: str = 'factored',
  **kwargs) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""
  num_actions = env_spec.actions.num_values
  def make_core_module() -> MuZeroNetworks:

    ###########################
    # Setup observation encoders (image + language)
    ###########################
    observation_fn = make_observation_fn(config, env_spec=env_spec)
    observation_fn = hk.to_module(observation_fn)("observation_fn")

    ###########################
    # Setup state function:
    #   conv-lstm --> embed --> SaVi 
    ###########################
    w_init_attn = hk.initializers.VarianceScaling(config.w_init_attn)

    # automatically calculate how many spatial vectors there are
    dummy_observation = jax_utils.zeros_like(env_spec.observations)
    sample = jax.lax.stop_gradient(observation_fn(dummy_observation))
    height = sample.image.shape[-3]
    width = sample.image.shape[-2]
    num_spatial_vectors = height*width
    del dummy_observation

    conv_state_fn = hk.Conv2DLSTM(
      input_shape=(height, width),
      output_channels=config.conv_lstm_dim,
      kernel_shape=3,
      name='state_lstm')

    def pos_embedder_factory():
      def output_transform(image):
        image = encoder.Mlp(
          # default settings from paper
          mlp_layers=config.pos_mlp_layers,
          layernorm=config.pos_layernorm)(image)

        height, width, n_features = image.shape[-3:]
        image = jnp.reshape(
          image,(*image.shape[:-3], height * width, n_features))  # [H*W, D]
        return image

      return encoder.PositionEmbedding(
          embedding_type=config.embedding_type,
          update_type=config.update_type,
          output_transform=output_transform
      )

    def factors_state_fn_factory():
      return attention.SlotAttentionV2(
        num_iterations=config.savi_iterations,
        qkv_size=config.slot_size,
        inverted_attn=config.inverted_attn,
        num_slots=config.num_slots,
        num_spatial=num_spatial_vectors,
        value_combination=config.slot_value_combination,
        epsilon=config.savi_epsilon,
        init=config.savi_init,
        gumbel_temp=config.savi_gumbel_temp,
        combo_update=config.savi_combo_update,
        relation_iterations=config.relation_iterations,
        clip_attn_probs=config.clip_attn_probs,
        rnn_class=hk.GRU,
        fixed_point=config.fixed_point,
        mlp_size=config.savi_mlp_size,
        relation_dim=config.relation_dim,
        w_init=w_init_attn,
        name='state_slot_attention'
      )

    pos_embedder = pos_embedder_factory()
    factors_state_fn = factors_state_fn_factory()

    state_fn = ConvSlotMemory(
      conv_state_fn=conv_state_fn,
      factors_state_fn=factors_state_fn,
      pos_embedder=pos_embedder,
    )

    ###########################
    # Setup ResNet + transformer transition_fn
    ###########################
    transition_pos_embedder = pos_embedder
    transition_factors_state_fn = factors_state_fn

    if config.seperate_model_nets:
      transition_pos_embedder = pos_embedder_factory()
      transition_factors_state_fn = factors_state_fn_factory()
    transition_fn = make_transition_fn(
        config=config,
        pos_embedder=transition_pos_embedder,
        factors_state_fn=transition_factors_state_fn,
    )


    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    root_predictor, model_predictor = make_prediction_function(
      config, env_spec)
    root_pred_fn = hk.to_module(root_predictor)(name='root_predictor')
    model_pred_fn = hk.to_module(model_predictor)(name='model_predictor')

    arch = FactoredMuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      observation_fn=observation_fn,
      state_fn=state_fn,
      transition_fn=transition_fn,
      root_pred_fn=root_pred_fn,
      model_pred_fn=model_pred_fn,
      invalid_actions=invalid_actions)

    return arch

  return make_network(config=config,
                      environment_spec=env_spec,
                      make_core_module=make_core_module,
                      invalid_actions=invalid_actions,
                      **kwargs)
