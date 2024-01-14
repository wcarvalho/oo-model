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


import haiku as hk
import jax
import jax.numpy as jnp


from modules import vision
from modules import language
from modules import vision_language
from modules.mlp_muzero import PredictionMlp, ResMlp, ResMlpBlock
from muzero.arch import MuZeroArch
from muzero.types import MuZeroNetworks, TaskAwareRep
from muzero.utils import TaskAwareRecurrentFn, scale_gradient
from muzero.networks import make_network


from factored_muzero.config import FactoredMuZeroConfig as MuZeroConfig
from factored_muzero import attention
from factored_muzero import encoder
from factored_muzero import gates
from factored_muzero.types import RootOutput, ModelOutput, TaskAwareSaviState

# MuZeroNetworks = networks_lib.UnrollableNetwork
NetworkOutput = networks_lib.NetworkOutput
RecurrentState = networks_lib.RecurrentState
Array = acme_types.NestedArray
GateFn = Callable[[Array, Array], Array]

class DualSaviState(NamedTuple):
  context: jnp.ndarray
  factors: jnp.ndarray
  factor_states: jnp.ndarray
  attn: Optional[jnp.ndarray] = None

def swap_if_none(a, b):
  if a is None: return b
  return a

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

class DualSlotMemory(hk.RNNCore):

  def __init__(self,
               context_memory: hk.GRU,
               factors_memory: attention.SlotAttention,
               mask_context: str = 'softmax',
               name: str = "slot_attention_dual_memory"):
    super().__init__(name=name)
    self.context_memory = context_memory
    self.factors_memory = factors_memory
    self.mask_context = mask_context
    assert isinstance(context_memory, hk.GRU)
    assert isinstance(factors_memory.rnn, hk.GRU)

  def combine_states(self, context_state, context_attn, factors_state):
    return DualSaviState(
        context=context_state,
        factors=factors_state.factors,
        factor_states=factors_state.factor_states,
        attn=factors_state.attn,
    )

  def __call__(
      self,
      inputs: attention.SaviInputs,
      state: attention.SaviState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    image = inputs.image
    has_batch = len(image.shape) == 3

    # [..., N, D]
    new_factors_state, _ = self.factors_memory(inputs, state)

    # [..., N, spatial_positions]
    slot_attn = new_factors_state.attn

    # [..., spatial_positions]
    # min_spatial_attn = slot_attn.min(-2)
    total_spatial_attn = slot_attn.sum(-2)
    leftover = (1 - total_spatial_attn)/.01
    if self.mask_context == "softmax":
      context_attn = jax.nn.softmax(leftover)
      context_attn = jax.lax.stop_gradient(context_attn)
    elif self.mask_context == "sum":
      context_attn = leftover/jnp.sum(leftover, axis=-1, keepdims=True)
      context_attn = jax.lax.stop_gradient(context_attn)
    elif self.mask_context == "none":
      context_attn = jnp.ones_like(total_spatial_attn)
    else:
      raise NotImplementedError

    image = image*jnp.expand_dims(context_attn, -1)
    if has_batch:
      image = image.reshape(image.shape[0], -1)
    else:
      image = image.reshape(-1)

    # [..., D]
    new_context_state, _ = self.context_memory(
        hk.Linear(128)(image), state.context)

    new_state = self.combine_states(
        new_context_state, context_attn, new_factors_state)
    return new_state, new_state

  def initial_state(self, batch_size: Optional[int] = None,
                    **unused_kwargs) -> attention.SaviState:

    context_state = self.context_memory.initial_state(batch_size)
    factors_state = self.factors_memory.initial_state(batch_size)

    total_spatial_attn = factors_state.attn.sum(-2)
    context_attn = jnp.ones_like(total_spatial_attn)

    return self.combine_states(context_state, context_attn, factors_state)

def make_savi_input(embedding: vision_language.TorsoOutput,
                    use_task: bool = False):
  other_obs_info = (embedding.reward, embedding.action)
  if use_task:
    other_obs_info = other_obs_info + (embedding.task, )

  return attention.SaviInputs(
    image=embedding.image,
    other_obs_info=jnp.concatenate(other_obs_info, axis=-1)
  )

def make_slot_selection_fn(selection: str,
                           attn_factory,
                           gate_factory):
  """This creates a function which aggregates slot information to single vector.

  attention: use task-vector to select slot information
  mean: mean of slot information
  """
  def selection_module(slot_reps: Array,
                       task_query: Array,
                       attn_outputs: Array):
    # slot_reps: [N, D]
    # task_query: [D]
    # attn_outputs: [layers, heads, factors, factors]

    def combine_attn(old, new_attn):
      old_attn = old.attn
      _, heads, rows, columns = old_attn.shape
      assert heads == new_attn.shape[0]

      # [H, 1, C-1] --> [H, 1, C]
      new_columns = columns - new_attn.shape[-1]
      zeros = jnp.zeros((*new_attn.shape[:-1], new_columns))
      new_attn = jnp.concatenate((zeros, new_attn), axis=-1)

      # [H, 1, C] --> [H, R, C]
      new_rows = rows - new_attn.shape[-2]
      zeros = jnp.zeros((*new_attn.shape[:-2], new_rows, columns))
      new_attn = jnp.concatenate((zeros, new_attn), axis=-2)

      # [L, H, R, C] + [H, R, C] --> [L+1, H, R, C]
      attn = jnp.concatenate((old_attn, new_attn[None]))

      return old._replace(attn=attn)

    if selection == 'attention_gate':
      # [D], [heads, factors_in, factors_out]
      pred_inputs, attn = attn_factory()(
        query=task_query[None],  # convert to [1, D] for attention
        key=slot_reps)
      pred_inputs = pred_inputs[0]  # query only
      pred_inputs = gate_factory()(task_query, pred_inputs)
      pred_inputs = hk.LayerNorm(
        axis=(-1),
        create_scale=True,
        create_offset=True)(pred_inputs)

      attn_outputs = combine_attn(attn_outputs, attn)

    elif selection == 'attention':
      # [D]
      pred_inputs, attn = attn_factory()(
        query=task_query[None],  # convert to [1, D] for attention
        key=slot_reps)
      pred_inputs = pred_inputs[0] # query only

      attn_outputs = combine_attn(attn_outputs, attn)

    elif selection == 'slot_mean':
      pred_inputs = slot_reps.mean(0)
    elif selection == 'task_query':
      pred_inputs = task_query

    return pred_inputs, attn_outputs
  return selection_module

def make_observation_fn(config, w_init_obs, env_spec, pos_embedder=None):
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
          w_init=w_init_obs,
          )
      elif config.vision_torso == 'motts2019':
        img_embed = vision.SaviVisionTorso(
          features=[32, 64, 128],
          kernel_size=[(4, 4), (4, 4), (3, 3)],
          strides=[(4, 4), (2, 2), (1, 1)],
          w_init=w_init_obs,
          )
      elif config.vision_torso == 'babyai_patches':
        img_embed = vision.SaviVisionTorso(
          features=[128, 128],
          kernel_size=[(8, 8), (1, 1)],
          strides=[(8, 8), (1, 1)],
          w_init=w_init_obs,
          )
      elif config.vision_torso == 'savi':
        img_embed = vision.SaviVisionTorso(
          features=[32, 32, 32, 32],
          kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
          strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
          w_init=w_init_obs,
          )
      elif config.vision_torso == 'savi_small':
        img_embed = vision.SaviVisionTorso(
          features=[32, 32, 32],
          kernel_size=[(5, 5), (5, 5), (5, 5)],
          strides=[(1, 1), (1, 1), (1, 1)],
          w_init=w_init_obs,
          )
      if not config.pos_embed_attn:
        vision_torso = encoder.PositionEncodingTorso(
            img_embed=img_embed,
            pos_embed=pos_embedder
        )
      else:
        vision_torso = img_embed

      return vision_language.Torso(
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
      )(inputs)
  return observation_fn

def make_savi_state_fn(config, num_spatial_vectors, w_init, pos_embedder=None):

  # during state unroll, rnn gets task from inputs and stores in state
  assert config.gru_init in ('orthogonal', 'default')
  if config.gru_init == 'orthogonal':
    # used in Slot Attention paper
    rnn_w_i_init = hk.initializers.Orthogonal()
  elif config.gru_init == 'default':
    # from haiku
    rnn_w_i_init = None

  assert config.savi_rnn in ('gru', 'lstm')
  if config.savi_rnn == 'gru':
    rnn_class = functools.partial(hk.GRU, w_i_init=rnn_w_i_init)
  elif config.savi_rnn == 'lstm':
    rnn_class = hk.LSTM

  savi_state_fn = attention.SlotAttention(
      num_iterations=config.savi_iterations,
      qkv_size=config.slot_size,
      inverted_attn=config.inverted_attn,
      num_slots=config.num_slots,
      num_spatial=num_spatial_vectors,
      project_values=config.project_slot_values,
      value_combination=config.slot_value_combination,
      epsilon=config.savi_epsilon,
      init=config.savi_init,
      gumbel_temp=config.savi_gumbel_temp,
      combo_update=config.savi_combo_update,
      relation_iterations=config.relation_iterations,
      clip_attn_probs=config.clip_attn_probs,
      rnn_class=rnn_class,
      fixed_point=config.fixed_point,
      mlp_size=config.savi_mlp_size,
      w_init=w_init,
      pos_embed=pos_embedder,
      name='state_slot_attention'
  )
  return savi_state_fn

def make_transformer_model(
    config,
    num_heads,
    qkv_size,
    mlp_size,
    w_init_attn
    ):
    get_gate_factory = functools.partial(
          gates.get_gate_factory,
          b_init=hk.initializers.Constant(config.b_init_attn),
          w_init=w_init_attn)

    def transformer_model(action_onehot: Array,
                          state: TaskAwareSaviState):
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transformer_model(
          action_onehot: Array,
          state: Union[attention.SaviState,
                       attention.TransformerOutput]
      ) -> Tuple[attention.TransformerOutput,
                 attention.TransformerOutput]:
        """Just make long sequence of state slots + action."""
        # action: [A]
        # state (slots): [N, D]
        slots = state.factors
        slot_dim = slots.shape[-1]
        action_w_init = hk.initializers.TruncatedNormal()
        encoded_action = hk.Linear(slot_dim,
                                   w_init=action_w_init,
                                   with_bias=False)(action_onehot)
        if config.action_as_factor:
          # [N+1, D]
          queries = jnp.concatenate((slots, encoded_action[None]))
        else:
          def join(a, b): return a+b
          shortcut = slots
          slots = hk.LayerNorm(
              axis=(-1), create_scale=True, create_offset=True)(slots)
          slots = jax.nn.relu(slots)
          # [N, D]
          queries = jax.vmap(join, (0, None))(slots, encoded_action)
          queries = hk.Linear(slot_dim, with_bias=False)(queries)
          queries += shortcut  # Residual link to maintain recurrent info flow.

        gate_factory = get_gate_factory(config.model_gate)

        outputs: attention.TransformerOutput = attention.Transformer(
            num_heads=num_heads,
            qkv_size=qkv_size,
            mlp_size=mlp_size,
            num_layers=config.transition_blocks,
            pre_norm=config.pre_norm,
            gate_factory=gate_factory,
            out_mlp=config.tran_out_mlp,
            w_init=w_init_attn,
            name='model_transition')(
              queries=queries)
        factors = outputs.factors

        if config.action_as_factor:
          # remove action representation
          factors = factors[:-1]

        res_layers = [
            ResMlpBlock(
              slot_dim,
              gate=lambda query, out: out,
              use_projection=False)
            for _ in range(config.tran_mlp_blocks)
        ]
        factors = hk.Sequential(res_layers)(factors)

        if config.scale_grad:
          factors=scale_gradient(factors, config.scale_grad)

        # outputs = outputs._replace(factors=factors)
        outputs = state._replace(
          factors=factors,
          attn=outputs.attn,
        )

        return outputs, outputs
      if action_onehot.ndim == 2:
        _transformer_model = jax.vmap(_transformer_model)
      return _transformer_model(action_onehot, state)

    # transition gets task from state and stores in state
    return TaskAwareRecurrentFn(
        get_task=lambda inputs, state: state.task,
        prep_state=lambda state: state.rep,  # get state-vector from TaskAwareRep
        couple_state_task=True,
        core=hk.to_module(transformer_model)(name='model_transformer'),
    )

def make_single_head_prediction_function(
    config, env_spec, img_decoder):

  num_actions = env_spec.actions.num_values

  slot_tran_mlp_size = swap_if_none(
      config.slot_tran_mlp_size, config.slot_size)
  w_init_attn = hk.initializers.VarianceScaling(config.w_init_attn)
  get_gate_factory = functools.partial(
    gates.get_gate_factory,
    b_init=hk.initializers.Constant(config.b_init_attn),
    w_init=w_init_attn)

  # set default prediction sizes based on transition function
  slot_pred_heads = config.slot_pred_heads or config.slot_tran_heads
  slot_pred_mlp_size = swap_if_none(
      config.slot_pred_mlp_size, slot_tran_mlp_size)
  slot_pred_qkv_size = config.slot_pred_qkv_size or config.slot_size

  pred_gate = config.pred_gate or config.model_gate

  def make_transformer(name: str, num_layers: int):
    return attention.Transformer(
        num_heads=slot_pred_heads,
        qkv_size=slot_pred_qkv_size,
        mlp_size=slot_pred_mlp_size,
        num_layers=num_layers,
        w_init=w_init_attn,
        pre_norm=config.pre_norm,
        out_mlp=config.pred_out_mlp,
        gate_factory=get_gate_factory(pred_gate),
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

  # root
  vpi_base = ResMlp(config.prediction_blocks,
                    ln=config.ln, name='pred_root_base')
  # pre_blocks_base = max(1, config.prediction_blocks//2)
  pred_transformer = make_transformer(
      'pred_root_base',
      num_layers=config.prediction_blocks)

  policy_fn = make_pred_fn(
      name='pred_root_policy', sizes=config.vpi_mlps, num_preds=num_actions)
  value_fn = make_pred_fn(
      name='pred_root_value', sizes=config.vpi_mlps, num_preds=config.num_bins)

  reward_fn = make_pred_fn(
      name='pred_model_reward', sizes=config.reward_mlps, num_preds=config.num_bins)

  task_projection = hk.Linear(
      config.slot_size,
      w_init=w_init_attn,
      with_bias=False,
      name='pred_task_projection')

  def task_gathers_factors(factors, task):
      keys = factors
      queries = task_projection(task)[None]
      outputs: attention.TransformerOutput = pred_transformer(
          queries=queries, inputs=keys)  # [N, D]
      return outputs, outputs.factors[0]

  def root_predictor(state_rep: TaskAwareSaviState):
    def _root_predictor(state_rep: TaskAwareSaviState):
      factors = state_rep.rep.factors
      outputs, pred_input = task_gathers_factors(
          factors=factors,
          task=state_rep.task)

      if config.share_pred_base:
        pred_input = vpi_base(pred_input)
      policy_logits = policy_fn(pred_input)
      value_logits = value_fn(pred_input)

      reconstruction = None
      if config.recon_coeff > 0:
        reconstruction = img_decoder(factors)

      return RootOutput(
          reconstruction=reconstruction,
          pred_attn_outputs=outputs,
          state=state_rep,
          value_logits=value_logits,
          policy_logits=policy_logits,
      )

    assert state_rep.task.ndim in (1, 2), "should be [D] or [B, D]"
    if state_rep.task.ndim == 2:
      _root_predictor = jax.vmap(_root_predictor)
    return _root_predictor(state_rep)

  def model_predictor(state_rep: TaskAwareSaviState):
    def _model_predictor(state_rep: TaskAwareSaviState):
      # output of transformer model
      model_state: attention.TransformerOutput = state_rep.rep
      outputs, pred_input = task_gathers_factors(
          factors=model_state.factors,
          task=state_rep.task)

      reward_logits = reward_fn(pred_input)

      if config.share_pred_base:
        pred_input = vpi_base(pred_input)
      policy_logits = policy_fn(pred_input)
      value_logits = value_fn(pred_input)

      return ModelOutput(
          pred_attn_outputs=outputs,
          new_state=model_state,
          value_logits=value_logits,
          policy_logits=policy_logits,
          reward_logits=reward_logits,
      )
    assert state_rep.task.ndim in (1, 2), "should be [D] or [B, D]"
    if state_rep.task.ndim == 2:
      _model_predictor = jax.vmap(_model_predictor)
    return _model_predictor(state_rep)

  return root_predictor, model_predictor

def make_multi_head_prediction_function(
        config, env_spec, img_decoder):
  num_actions = env_spec.actions.num_values

  slot_tran_mlp_size = swap_if_none(
      config.slot_tran_mlp_size, config.slot_size)
  w_init_attn = hk.initializers.VarianceScaling(config.w_init_attn)
  get_gate_factory = functools.partial(
      gates.get_gate_factory,
      b_init=hk.initializers.Constant(config.b_init_attn),
      w_init=w_init_attn)

  # set default prediction sizes based on transition function
  slot_pred_heads = config.slot_pred_heads or config.slot_tran_heads
  slot_pred_mlp_size = swap_if_none(
      config.slot_pred_mlp_size, slot_tran_mlp_size)
  slot_pred_qkv_size = config.slot_pred_qkv_size or config.slot_size

  pred_gate = config.pred_gate or config.model_gate

  def make_transformer(name: str, num_layers: int):
    return attention.Transformer(
        num_heads=slot_pred_heads,
        qkv_size=slot_pred_qkv_size,
        mlp_size=slot_pred_mlp_size,
        num_layers=num_layers,
        w_init=w_init_attn,
        pre_norm=config.pre_norm,
        out_mlp=config.pred_out_mlp,
        gate_factory=get_gate_factory(pred_gate),
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

  # root
  vpi_base = ResMlp(config.prediction_blocks,
                    ln=config.ln, name='pred_root_base')
  # pre_blocks_base = max(1, config.prediction_blocks//2)
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

  if config.seperate_model_nets:
    model_pred_transformer = make_transformer(
      'pred_model_base',
      num_layers=config.prediction_blocks)
    model_vpi_base = ResMlp(config.prediction_blocks, ln=config.ln, name='pred_model_base')
    model_value_fn = make_pred_fn(
      name='pred_model_value', sizes=config.vpi_mlps, num_preds=config.num_bins)
    model_policy_fn = make_pred_fn(
      name='pred_model_policy', sizes=config.vpi_mlps, num_preds=num_actions)
  else:
    model_pred_transformer = pred_transformer
    model_value_fn = value_fn
    model_policy_fn = policy_fn
    model_vpi_base = vpi_base

  task_projection = hk.Linear(
      config.slot_size,
      w_init=w_init_attn,
      with_bias=False,
      name='pred_task_projection')

  def task_concat_factors(state_rep, pred_transformer_):
    """
    factors: [N or N+1,D]. N+1 if have a factor for full image.
    task: [D]
    Concatenate (task + full image factor) with every factor
    """
    factors = state_rep.rep.factors  # [N, D]
    task = state_rep.task            # [D]

    if config.slots_use_task:
      # if slots already have task as input, no need to use here 
      task = jnp.zeros_like(task)

    context = task_projection(task)  # [D]
    concat = lambda a, b: jnp.concatenate((a, b))
    #import ipdb; ipdb.set_trace()
    if config.context_slot_dim:
      # task + image
      context = concat(context, state_rep.rep.context)
      # concatenate context with every factor, [N, 2D]
      pred_input = jax.vmap(concat, (None, 0), (0))(context, factors)
      pred_input = pred_transformer_(queries=pred_input)
      attn_output = None
    else:
      queries = jax.vmap(concat, (None, 0), (0))(context, factors)
      pred_input = pred_transformer_(queries=queries)  # [N, 2D]
      attn_output = pred_input
    pred_input = pred_input.factors

    if config.learned_weights:
      weights = weight_fn(pred_input)  # [N, 1]
      weights = jax.nn.softmax(weights, axis=-2)
    else:
      nfactors = len(factors)
      weights = jnp.ones(nfactors)/nfactors
      weights = weights[:, None]

    return pred_input, weights, attn_output

  def weighted_average(x: jnp.ndarray,
                        weights: jnp.ndarray):
    """x: [..., num_factors, num_predictions]"""
    return (x*weights).sum(axis=-2)

  def root_predictor(state_rep: TaskAwareSaviState):
    def _root_predictor(state_rep: TaskAwareSaviState):

      pred_input, weights, outputs = task_concat_factors(
        state_rep, pred_transformer_=pred_transformer)

      if config.share_pred_base:
        pred_input = vpi_base(pred_input).factors
      policy_logits = policy_fn(pred_input)
      value_logits = value_fn(pred_input)

      policy_logits = weighted_average(
          policy_logits, weights)
      value_logits = weighted_average(
          value_logits, weights)

      reconstruction = None

      return RootOutput(
          reconstruction=reconstruction,
          pred_attn_outputs=outputs,
          state=state_rep,
          value_logits=value_logits,
          policy_logits=policy_logits,
      )
    assert state_rep.task.ndim in (1, 2), "should be [D] or [B, D]"
    if state_rep.task.ndim == 2:
      _root_predictor = jax.vmap(_root_predictor)
    return _root_predictor(state_rep)

  def model_predictor(state_rep: TaskAwareSaviState):
    def _model_predictor(state_rep: TaskAwareSaviState):

      pred_input, weights, outputs = task_concat_factors(
        state_rep, pred_transformer_=model_pred_transformer)

      reward_logits = reward_fn(pred_input)
      if config.share_pred_base:
        pred_input = model_vpi_base(pred_input)
      policy_logits = model_policy_fn(pred_input)
      value_logits = model_value_fn(pred_input)

      reward_logits = weighted_average(
          reward_logits, weights)
      policy_logits = weighted_average(
          policy_logits, weights)
      value_logits = weighted_average(
          value_logits, weights)

      return ModelOutput(
          pred_attn_outputs=outputs,
          new_state=state_rep.rep,
          value_logits=value_logits,
          policy_logits=policy_logits,
          reward_logits=reward_logits,
      )
    assert state_rep.task.ndim in (1, 2), "should be [D] or [B, D]"
    if state_rep.task.ndim == 2:
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
    w_init_obs = config.w_init_obs
    if config.w_init_obs is not None:
      w_init_obs = hk.initializers.VarianceScaling(config.w_init_obs)

    output_transform = lambda x:x
    if config.transform_pos_embed:
      output_transform = encoder.Mlp(
            # default settings from paper
            mlp_layers=config.pos_mlp_layers,
            w_init=w_init_obs,
            layernorm=config.pos_layernorm,
        )
    pos_embedder = encoder.PositionEmbedding(
        embedding_type=config.embedding_type,
        update_type=config.update_type,
        w_init=w_init_obs,
        output_transform=output_transform
    )
    if config.pos_embed_attn:
      pos_embedder_attn = pos_embedder
      pos_embedder_obs = None
    else:
      pos_embedder_attn = None
      pos_embedder_obs = pos_embedder

    observation_fn = make_observation_fn(
        config,
        w_init_obs=w_init_obs,
        pos_embedder=pos_embedder_obs,
        env_spec=env_spec)
    observation_fn = hk.to_module(observation_fn)("observation_fn")

    # automatically calculate how many spatial vectors there are
    dummy_observation = jax_utils.zeros_like(env_spec.observations)
    sample = jax.lax.stop_gradient(observation_fn(dummy_observation))
    if config.pos_embed_attn:
      num_spatial_vectors = sample.image.shape[-3]*sample.image.shape[-2]
    else:
      num_spatial_vectors = sample.image.shape[-2]

    ###########################
    # Setup observation image decoder
    ###########################
    img_decoder = None
    if config.recon_coeff > 0.0:
      if config.vision_torso == 'babyai_patches':
        import math
        latent_height = math.sqrt(num_spatial_vectors)
        latent_height = int(latent_height)
        resolution = (latent_height, latent_height)
        img_decode = vision.SaviVisionTorso(
            features=[32, 32, 32],
            kernel_size=[(1, 1), (1, 1), (8, 8)],
            strides=[(1, 1), (1, 1), (8, 8)],
            layer_transpose=[True, True, True],
            w_init=w_init_obs,
        )
      elif config.vision_torso == 'savi':
        resolution = (8, 8)
        img_decode = vision.SaviVisionTorso(
            features=[64, 64, 64, 64],
            kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
            strides=[(2, 2), (2, 2), (2, 2), (2, 2)],
            layer_transpose=[True, True, True, False],
            w_init=w_init_obs,
        )
      else:
        raise NotImplementedError

      img_decoder = vision.SpatialBroadcastDecoder(
          resolution=resolution,
          backbone=img_decode,
          pos_emb=encoder.PositionEmbedding(
              embedding_type=config.embedding_type,
              update_type=config.update_type),
      )



    ###########################
    # Setup state function: SaVi 
    ###########################
    w_init_attn = hk.initializers.VarianceScaling(config.w_init_attn)
    state_fn = make_savi_state_fn(
      config, num_spatial_vectors, w_init_attn,
      pos_embedder=pos_embedder_attn)
    if config.context_slot_dim:
      state_fn = DualSlotMemory(
        context_memory=hk.GRU(
          config.context_slot_dim,
          name='slot_attention_context'),
        mask_context=config.mask_context,
        factors_memory=state_fn)

    def combine_hidden_obs(
        hidden: attention.SaviState,
        emb: vision_language.TorsoOutput):
      """After get hidden from SlotAttention, combine with task from embedding."""
      return TaskAwareRep(rep=hidden, task=emb.task)

    ###########################
    # Setup transformer model transition_fn
    ###########################
    # set default transition sizes based on state function
    slot_tran_heads = config.slot_tran_heads
    slot_tran_mlp_size = swap_if_none(
      config.slot_tran_mlp_size, config.slot_size)

    transition_fn = make_transformer_model(
        config=config,
        num_heads=slot_tran_heads,
        qkv_size=config.slot_size,
        mlp_size=slot_tran_mlp_size,
        w_init_attn=w_init_attn,
    )


    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    if agent_name == 'factored':
      make_prediction_fn = make_single_head_prediction_function
    elif agent_name == 'branched':
      make_prediction_fn = make_multi_head_prediction_function
    else:
      raise NotImplementedError

    root_predictor, model_predictor = make_prediction_fn(
      config, env_spec, img_decoder)
    root_pred_fn = hk.to_module(root_predictor)(name='pred_root_predictor')
    model_pred_fn = hk.to_module(model_predictor)(name='pred_model_predictor')

    arch = FactoredMuZeroArch(
      action_encoder=lambda a: jax.nn.one_hot(
        a, num_classes=num_actions),
      observation_fn=observation_fn,
      prep_state_input=functools.partial(make_savi_input,
                                         use_task=config.slots_use_task),
      state_fn=state_fn,
      combine_hidden_obs=combine_hidden_obs,
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
