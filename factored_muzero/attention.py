from typing import Tuple, Optional, Callable, NamedTuple, Union

import functools

from acme import types
import chex
import haiku as hk
import jax
import math
import jax.numpy as jnp
import numpy as np
from factored_muzero import encoder

Array = types.NestedArray
GateFn = Callable[[Array, Array], Array]

@chex.dataclass(frozen=True)
class SaviInputs:
  image: jnp.ndarray
  other_obs_info: Optional[jnp.ndarray] = None

class TransformerOutput(NamedTuple):
  factors: jnp.ndarray
  attn: Optional[jnp.ndarray] = None

class SaviState(NamedTuple):
  factors: jnp.ndarray
  factor_states: jnp.ndarray
  attn: Optional[jnp.ndarray] = None

class MultiLinear(hk.Linear):
  def __init__(self, *args, heads, **kwargs):
    super().__init__(*args, **kwargs)
    self._heads = heads

  def __call__(self, x: jnp.ndarray):
    # [N, D]
    out = super().__call__(x)
    dim = out.shape[-1]

    # [N, H, D/H]
    return out.reshape(*out.shape[:-1], self._heads, dim//self._heads)


def gumbel_logits(rng, logits: jnp.ndarray, temp: float=1.):
  """Sample from the Gumbel softmax / concrete distribution."""
  gumbel_noise = jax.random.gumbel(rng, logits.shape)
  return (logits + gumbel_noise) / temp


def straight_through(x: Array, transform: Callable[[Array], Array]):
    no_gradient = jax.lax.stop_gradient(transform(x))
    # Subtracting out and then adding back
    return no_gradient + x - jax.lax.stop_gradient(x)

def straight_through_estimator(soft_samples: jnp.ndarray):
    hard_samples = jnp.argmax(soft_samples, axis=-1)
    hard_samples_one_hot = jax.nn.one_hot(hard_samples, soft_samples.shape[-1])

    # Subtracting out the continuous relaxation of the argmax function,
    # and then adding back in the hard version.
    return hard_samples_one_hot + soft_samples - jax.lax.stop_gradient(soft_samples)

def sample_multi_categorical(
    slots_rep: jnp.ndarray,
    slot_shape: Tuple[int],
    ncategories: int,
    temp: float):
  logits = hk.Linear(slot_shape[-1])(slots_rep)

  # split into subsets, each with M categories
  logits = logits.reshape(slot_shape[:-1] + (-1, ncategories))
  key = hk.next_rng_key()

  logits = gumbel_logits(key, logits, temp)
  soft_sample = jax.nn.softmax(logits)
  samples = straight_through_estimator(soft_sample)
  return samples.reshape(slot_shape)


class SlotAttention(hk.RNNCore):
  def __init__(self,
               qkv_size: int,
               num_spatial: int,
               num_iterations: int = 1,
               mlp_size: Optional[int] = None,
               epsilon: float = 1e-8,
               num_heads: int = 1,
               num_slots: int = 4,
               inverted_attn: bool = True,
               value_combination: str = 'avg',
               init: str = 'noise',
               relation_dim: int = 64,
               slot_categories: int = 4,
               gumbel_temp: float = 1.0,
               inter_slot_heads: int = 2,
               relation_iterations: str = 'once',
               combo_update: str = 'concat',
               clip_attn_probs: bool = True,
               fixed_point: bool = False,
               w_init: Optional[hk.initializers.Initializer] = None,
               rnn_class: hk.RNNCore = hk.GRU,
               pos_embed: Optional[hk.Module] = None,
               name: Optional[str] = None,
               **kwargs):
    super().__init__(name=name)
    self.num_iterations = num_iterations
    self.qkv_size = qkv_size
    self.mlp_size = mlp_size
    self.epsilon = epsilon
    self.num_heads = num_heads
    self.num_slots = num_slots
    self.num_spatial = num_spatial
    self.clip_attn_probs = clip_attn_probs
    self.slot_categories = slot_categories
    self.gumbel_temp = gumbel_temp
    self.init = init
    self.w_init = w_init
    self.attention_in_updates = False
    self.combo_update = combo_update
    self.value_combination = value_combination
    self.head_dim = self.qkv_size // self.num_heads
    self.rnn = rnn_class(self.head_dim)
    assert relation_iterations in ('once', 'every', 'none', 'last')
    self.relation_iterations = relation_iterations
    self.fixed_point = fixed_point

    self.inverted_attention = InvertedDotProductAttention(
        norm_type="mean",
        inverted=inverted_attn,
        multi_head=num_heads > 1,
        value_combination=value_combination,
        epsilon=self.epsilon,
        clip_attn_probs=clip_attn_probs,
        name='obs_attn')


    self.inter_slot_attn = GeneralMultiHeadAttention(
        num_heads=inter_slot_heads,
        key_size=qkv_size,
        model_size=relation_dim,
        w_init=w_init,
        attn_weights=False,
        epsilon=self.epsilon,
        clip_attn_probs=clip_attn_probs,
        name='relation_attn')

    self.pos_embedder = pos_embed

  def __call__(
      self,
      inputs: SaviInputs,
      state: SaviState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    ####################################
    # Sizes
    ####################################
    image = inputs.image
    slots = state.factors
    qkv_size = self.qkv_size
    head_dim = qkv_size // self.num_heads

    if self.pos_embedder is not None:
      has_batch = len(image.shape) == 4
      assert len(image.shape) in (3,4), "should either be [H, W, C] or [B, H, W, C]"
      nspatial = image.shape[-3]*image.shape[-2]
      assert nspatial == self.num_spatial, f"{nspatial} != {self.num_spatial}"
      raise NotImplementedError("don't follow this path")
    else:
      has_batch = len(image.shape) == 3
      assert len(image.shape) in (2,3), "should either be [N, C] or [B, N, C]"
      nspatial = image.shape[-2]
      assert nspatial == self.num_spatial, f"{nspatial} != {self.num_spatial}"



    ####################################
    # Shared modules
    ####################################
    dense = functools.partial(MultiLinear,
                              output_size=qkv_size,
                              heads=self.num_heads,
                              w_init=self.w_init,
                              with_bias=False)

    # Shared modules.
    dense_q = dense(name="general_dense_q_0")
    layernorm_q = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)

    if self.mlp_size is not None and self.mlp_size > 0:
      mlp = encoder.Mlp(
        mlp_layers=[self.mlp_size],
        layernorm="pre", residual=True,
        w_init=self.w_init)

    if self.pos_embedder is not None:
      def pos_embed(x):
        shape = (-1, self.num_heads, qkv_size)
        if has_batch:
          x = hk.BatchApply(self.pos_embedder)(x)
          return x.reshape((x.shape[0],)+shape)
        else:
          x = self.pos_embedder(x)
          return x.reshape(shape)

    ####################################
    # Compute query, key, value representations for attention
    ####################################
    # add other obs info (e.g. prev_action, prev_reward)
    other_obs_info = hk.Linear(head_dim, name='other_obs_info')(inputs.other_obs_info)

    #---------
    # For attention to image
    #---------
    # inputs.shape = (..., n_inputs, inputs_size).
    image = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(image)

    # k.shape = (..., n_inputs, slot_size).
    k = dense(name="general_dense_k_0")(image)

    # v.shape = (..., n_inputs, slot_size).
    v = dense(name="general_dense_v_0")(image)

    #---------
    # For attention across slots
    #---------
    def make_relation_inputs(slots):
      """add other observation informaton to slot info."""
      concat = lambda a, b: jnp.concatenate((a, b), axis=-1)
      # repeat over # of slots
      concat = jax.vmap(concat, (-2, None), -2)
      return concat(slots, other_obs_info)

    # if attention heads are allowed to attention to nothing, add 1 slot of zeros
    def add_zeros_vector(x):
      zeros_shape = (1, x.shape[-1])
      if has_batch:
        zeros_shape = (x.shape[0],) + zeros_shape
      zeros = jnp.zeros(zeros_shape)
      return jnp.concatenate((x, zeros), axis=-2) 

    def combine_attention_outputs(image_updates, inter_slot_updates):
      if self.combo_update == 'concat':
        # concat over slot dim
        concat = lambda a, b, c: jnp.concatenate(
          (a, b, c), axis=-1)
        # repeat over # of slots
        concat = jax.vmap(concat, (-2, -2, None), -2)
        return concat(
          image_updates,
          inter_slot_updates,
          other_obs_info)
      elif self.combo_update == 'sum':
        dim = image_updates.shape[-1]
        linear = lambda x: hk.Linear(dim, with_bias=False)(jax.nn.relu(x))
        # [N, D] + [N, D] + [1, D]
        return (linear(image_updates) + 
                linear(inter_slot_updates) +
                linear(jnp.expand_dims(other_obs_info, axis=-2)))
      else: 
        raise NotImplementedError(self.combo_update)

    ####################################
    # Multiple rounds of attention.
    ####################################
    factor_states = state.factor_states

    inter_slot_updates = jnp.zeros(
      (*slots.shape[:-1], self.inter_slot_attn.model_size))
    if self.relation_iterations == "none":
      pass
    elif self.relation_iterations == "once":
      attn_inputs = make_relation_inputs(slots)
      inter_slot_updates = self.inter_slot_attn(
        query=attn_inputs,
        key=add_zeros_vector(attn_inputs))

    for idx in range(self.num_iterations):
      if idx == self.num_iterations - 1:
        if self.fixed_point:
          slots = jax.lax.stop_gradient(slots)
        if self.relation_iterations == "last":
          attn_inputs = make_relation_inputs(slots)
          inter_slot_updates = self.inter_slot_attn(
            query=attn_inputs,
            key=add_zeros_vector(attn_inputs))


      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_inputs, slot_size).
      if self.pos_embedder is not None:
        k = pos_embed(k)
        v = pos_embed(v)

      updates, attn_weights = self.inverted_attention(
        query=q, key=k, value=v)

      if self.relation_iterations == "every":
        attn_inputs = make_relation_inputs(slots)
        inter_slot_updates = self.inter_slot_attn(
          query=attn_inputs,
          key=add_zeros_vector(attn_inputs))

      updates = combine_attention_outputs(
        image_updates=updates,
        inter_slot_updates=inter_slot_updates)

      # Recurrent update.
      if has_batch:
        slots, factor_states = hk.BatchApply(self.rnn)(updates, factor_states)
      else:
        slots, factor_states = self.rnn(updates, factor_states)

      # Feedforward block with pre-normalization.
      if self.mlp_size is not None and self.mlp_size > 0:
        slots = mlp(slots)

    # cat_slots = None
    # if self.init == 'categorical':
    #   cat_slots = sample_multi_categorical(
    #     slots_rep=slots,
    #     slot_shape=slots.shape,
    #     ncategories=self.slot_categories,
    #     temp=self.gumbel_temp)

    #   slots = jnp.concatenate((cat_slots, slots), axis=-1)

    state = SaviState(
      factors=slots,
      factor_states=factor_states,
      attn=attn_weights,
    )
    return state, state

  def initial_slots(self, batch_size: Optional[int] = None):
    shape = (self.num_slots, self.qkv_size)
    if batch_size is not None:
      shape = (batch_size,) + shape

    def get_slot_embeddings(dim=None):
      dim = dim or self.qkv_size
      embeddings = hk.Embed(
        vocab_size=self.num_slots,
        embed_dim=dim,
        name='init_embed')(
          jnp.arange(self.num_slots))
      if batch_size is not None:
        embeddings = embeddings[None]
        embeddings = jnp.tile(embeddings, (batch_size, 1, 1))
      return embeddings

    if self.init == 'zeros':
      return jnp.zeros(shape, dtype=jnp.float32)
    elif self.init == 'noise':
      return jax.random.normal(
        hk.next_rng_key(), shape, dtype=jnp.float32)
    elif self.init == 'embed':
      slot_inits = get_slot_embeddings()
      slot_inits = hk.Linear(self.qkv_size, name='init_embed_proj')(slot_inits)
      return slot_inits
    elif self.init == 'gauss':
      mean_std = get_slot_embeddings()
      embeds = jax.nn.relu(mean_std)
      mean_std = hk.Linear(2*self.qkv_size, name='init_embed_proj')(embeds)
      mean, log_std = jnp.split(mean_std, 2, axis=-1)
      std = jnp.exp(jax.nn.softplus(log_std)) + self.epsilon
      eps = jax.random.normal(
        hk.next_rng_key(), std.shape, dtype=jnp.float32)

      slot_inits = mean + std*eps

      return slot_inits

    elif self.init == 'categorical':
      embeds = get_slot_embeddings()
      embeds = jax.nn.relu(embeds)
      slot_inits = sample_multi_categorical(
        slots_rep=embeds,
        slot_shape=shape,
        ncategories=self.slot_categories,
        temp=self.gumbel_temp)
      return slot_inits
    else:
      raise NotImplementedError(self.init)

  def initial_state(self, batch_size: Optional[int] = None):
    attn_shape = (self.num_slots, self.num_spatial)
    if batch_size is not None:
      attn_shape = (batch_size,) + attn_shape

    factors = self.initial_slots(batch_size)
    factor_states = self.rnn.initial_state(batch_size)

    if isinstance(factor_states, hk.LSTMState):
      factor_states = hk.LSTMState(
        hidden=factors,
        cell=factors*0.)
    elif isinstance(factor_states, jnp.ndarray):
      factor_states = factors
    else:
      raise RuntimeError(type(factor_states))

    state = SaviState(
      factors=factors,
      factor_states=factor_states,
      attn=jnp.zeros(attn_shape, dtype=jnp.float32),
    )

    return state


class SlotAttentionV2(hk.RNNCore):
  def __init__(self,
               qkv_size: int,
               num_spatial: int,
               num_iterations: int = 1,
               mlp_size: Optional[int] = None,
               epsilon: float = 1e-8,
               num_heads: int = 1,
               num_slots: int = 4,
               inverted_attn: bool = True,
               value_combination: str = 'avg',
               init: str = 'noise',
               relation_dim: int = 64,
               slot_categories: int = 4,
               gumbel_temp: float = 1.0,
               inter_slot_heads: int = 2,
               relation_iterations: str = 'once',
               combo_update: str = 'concat',
               clip_attn_probs: bool = True,
               fixed_point: bool = False,
               w_init: Optional[hk.initializers.Initializer] = None,
               rnn_class: hk.RNNCore = hk.GRU,
               pos_embed: Optional[hk.Module] = None,
               name: Optional[str] = None,
               **kwargs):
    super().__init__(name=name)
    self.num_iterations = num_iterations
    self.qkv_size = qkv_size
    self.mlp_size = mlp_size
    self.epsilon = epsilon
    self.num_heads = num_heads
    self.num_slots = num_slots
    self.num_spatial = num_spatial
    self.clip_attn_probs = clip_attn_probs
    self.slot_categories = slot_categories
    self.gumbel_temp = gumbel_temp
    self.init = init
    self.w_init = w_init
    self.attention_in_updates = False
    self.combo_update = combo_update
    self.value_combination = value_combination
    self.head_dim = self.qkv_size // self.num_heads
    self.rnn = rnn_class(self.head_dim)
    assert relation_iterations in ('once', 'every', 'none', 'last')
    self.relation_iterations = relation_iterations
    self.fixed_point = fixed_point

    self.inverted_attention = InvertedDotProductAttention(
        norm_type="mean",
        inverted=inverted_attn,
        multi_head=num_heads > 1,
        value_combination=value_combination,
        epsilon=self.epsilon,
        clip_attn_probs=clip_attn_probs,
        name='obs_attn')


    self.inter_slot_attn = GeneralMultiHeadAttention(
        num_heads=inter_slot_heads,
        key_size=qkv_size,
        model_size=relation_dim,
        w_init=w_init,
        attn_weights=False,
        epsilon=self.epsilon,
        clip_attn_probs=clip_attn_probs,
        name='relation_attn')

    self.pos_embedder = pos_embed

  def __call__(
      self,
      image: jax.Array,
      state: SaviState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    ####################################
    # Sizes
    ####################################
    slots = state.factors
    has_batch = len(image.shape) == 3


    ####################################
    # Shared modules
    ####################################
    dense = functools.partial(MultiLinear,
                              output_size=self.qkv_size,
                              heads=self.num_heads,
                              w_init=self.w_init,
                              with_bias=False)

    # Shared modules.
    dense_q = dense(name="general_dense_q_0")
    layernorm_q = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)

    if self.mlp_size is not None and self.mlp_size > 0:
      mlp = encoder.Mlp(
        mlp_layers=[self.mlp_size],
        layernorm="pre", residual=True,
        w_init=self.w_init)

    ####################################
    # Compute query, key, value representations for attention
    ####################################
    # if attention heads are allowed to attention to nothing, add 1 slot of zeros
    def add_zeros_vector(x):
      zeros_shape = (1, x.shape[-1])
      if has_batch:
        zeros_shape = (x.shape[0],) + zeros_shape
      zeros = jnp.zeros(zeros_shape)
      return jnp.concatenate((x, zeros), axis=-2) 

    #---------
    # For attention to image
    #---------
    # inputs.shape = (..., n_inputs, inputs_size).
    image = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(image)

    # k.shape = (..., n_inputs, slot_size).
    k = dense(name="general_dense_k_0")(image)

    # v.shape = (..., n_inputs, slot_size).
    v = dense(name="general_dense_v_0")(image)

    ####################################
    # Multiple rounds of attention.
    ####################################
    factor_states = state.factor_states

    inter_slot_updates = jnp.zeros(
      (*slots.shape[:-1], self.inter_slot_attn.model_size))
    if self.relation_iterations == "none":
      pass
    elif self.relation_iterations == "once":
      inter_slot_updates = self.inter_slot_attn(
        query=slots,
        key=add_zeros_vector(slots))

    for idx in range(self.num_iterations):
      if idx == self.num_iterations - 1:
        if self.fixed_point:
          slots = jax.lax.stop_gradient(slots)

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_inputs, slot_size).

      updates, attn_weights = self.inverted_attention(
        query=q, key=k, value=v)
      
      # combine with inter-slot attention
      updates = jnp.concatenate((updates, inter_slot_updates), axis=-1)


      # Recurrent update.
      if has_batch:
        slots, factor_states = hk.BatchApply(self.rnn)(updates, factor_states)
      else:
        slots, factor_states = self.rnn(updates, factor_states)

      # Feedforward block with pre-normalization.
      if self.mlp_size is not None and self.mlp_size > 0:
        slots = mlp(slots)

    state = SaviState(
      factors=slots,
      factor_states=factor_states,
      attn=attn_weights,
    )
    return state, state

  def initial_slots(self, batch_size: Optional[int] = None):
    shape = (self.num_slots, self.qkv_size)
    if batch_size is not None:
      shape = (batch_size,) + shape

    def get_slot_embeddings(dim=None):
      dim = dim or self.qkv_size
      embeddings = hk.Embed(
        vocab_size=self.num_slots,
        embed_dim=dim,
        name='init_embed')(
          jnp.arange(self.num_slots))
      if batch_size is not None:
        embeddings = embeddings[None]
        embeddings = jnp.tile(embeddings, (batch_size, 1, 1))
      return embeddings

    if self.init == 'zeros':
      return jnp.zeros(shape, dtype=jnp.float32)
    elif self.init == 'noise':
      return jax.random.normal(
        hk.next_rng_key(), shape, dtype=jnp.float32)
    elif self.init == 'embed':
      slot_inits = get_slot_embeddings()
      slot_inits = hk.Linear(self.qkv_size, name='init_embed_proj')(slot_inits)
      return slot_inits
    elif self.init == 'gauss':
      mean_std = get_slot_embeddings()
      embeds = jax.nn.relu(mean_std)
      mean_std = hk.Linear(2*self.qkv_size, name='init_embed_proj')(embeds)
      mean, log_std = jnp.split(mean_std, 2, axis=-1)
      std = jnp.exp(jax.nn.softplus(log_std)) + self.epsilon
      eps = jax.random.normal(
        hk.next_rng_key(), std.shape, dtype=jnp.float32)

      slot_inits = mean + std*eps

      return slot_inits

    elif self.init == 'categorical':
      embeds = get_slot_embeddings()
      embeds = jax.nn.relu(embeds)
      slot_inits = sample_multi_categorical(
        slots_rep=embeds,
        slot_shape=shape,
        ncategories=self.slot_categories,
        temp=self.gumbel_temp)
      return slot_inits
    else:
      raise NotImplementedError(self.init)

  def initial_state(self, batch_size: Optional[int] = None):
    attn_shape = (self.num_slots, self.num_spatial)
    if batch_size is not None:
      attn_shape = (batch_size,) + attn_shape

    factors = self.initial_slots(batch_size)
    factor_states = self.rnn.initial_state(batch_size)

    if isinstance(factor_states, hk.LSTMState):
      factor_states = hk.LSTMState(
        hidden=factors,
        cell=factors*0.)
    elif isinstance(factor_states, jnp.ndarray):
      factor_states = factors
    else:
      raise RuntimeError(type(factor_states))

    state = SaviState(
      factors=factors,
      factor_states=factor_states,
      attn=jnp.zeros(attn_shape, dtype=jnp.float32),
    )

    return state


class InvertedDotProductAttention(hk.Module):
  """Inverted version of dot-product attention (softmax over query axis)."""

  def __init__(self,
               norm_type: Optional[str] = "mean",  # mean, layernorm, or None
               inverted: bool = True,
               multi_head: bool = False,
               name: Optional[str] = None,
               **kwargs,
               ):
    super().__init__(name=name)
    self.norm_type = norm_type
    self.multi_head = multi_head

    self.attn = GeneralizedDotProductAttention(
        inverted_attn=inverted,
        renormalize_keys=True if self.norm_type == "mean" else False,
        attn_weights=True,
        **kwargs)

  def __call__(self, query: Array, key: Array, value: Array,
               train: bool = False) -> Array:
    """Computes inverted dot-product attention.
    Args:
      query: Queries with shape of `[batch..., q_num, qk_features]`.
      key: Keys with shape of `[batch..., kv_num, qk_features]`.
      value: Values with shape of `[batch..., kv_num, v_features]`.
      train: Indicating whether we're training or evaluating.
    Returns:
      Output of shape `[batch_size..., n_queries, v_features]`
    """
    del train  # Unused.

    # Apply attention mechanism.
    output, attn_weights = self.attn(query=query, key=key, value=value)

    if self.multi_head:
      # Multi-head aggregation. Equivalent to concat + dense layer.
      print("need to incorporate proper w_init initialization")
      import ipdb; ipdb.set_trace()
      output = hk.Linear(output.shape[-1])(output)
      # nn.DenseGeneral(
      #     features=output.shape[-1], axis=(-2, -1))(output)
    else:
      # Remove head dimension.
      # [N, H, D] --> [N, D]
      output = jnp.squeeze(output, axis=-2)

      # [H, N, D] --> [N, D]
      attn_weights = jnp.squeeze(attn_weights, axis=-3)

    if self.norm_type == "layernorm":
      output = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(output)

    return output, attn_weights


def sum_gate_factory(*args, **kwargs) -> GateFn:
  return lambda x, y: x+y


class Transformer(hk.Module):
  """Transformer with multiple blocks."""

  def __init__(self,
               num_heads: int,
               qkv_size: int,
               mlp_size: int,
               num_layers: int,
               pre_norm: bool = False,
               w_init: Optional[hk.initializers.Initializer] = None,
               gate_factory = None,
               mlp_factory = None,
               out_mlp = True,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.qkv_size = qkv_size
    self.mlp_size = mlp_size
    self.num_layers = num_layers
    self.pre_norm = pre_norm
    self.w_init = w_init
    if gate_factory is None:
      gate_factory = sum_gate_factory
    self.gate_factory = gate_factory
    self.mlp_factory = mlp_factory
    self.out_mlp = out_mlp

  def __call__(self,
               queries: Union[Array, TransformerOutput],
               inputs: Optional[Array] = None,
               padding_mask: Optional[Array] = None,
               train: bool = False) -> Array:
    if self.num_layers == 0:
      return TransformerOutput(factors=queries)

    x = queries
    all_attn_weights = []
    if isinstance(queries, TransformerOutput):
      x = queries.factors
      prior_attn_weights = queries.attn

    for lyr in range(self.num_layers):
      x, attn_weights = TransformerBlock(
          num_heads=self.num_heads, qkv_size=self.qkv_size,
          mlp_size=self.mlp_size, pre_norm=self.pre_norm,
          w_init=self.w_init,
          gate_factory=self.gate_factory,
          mlp_factory=self.mlp_factory,
          out_mlp=self.out_mlp,
          name=f"TransformerBlock{lyr}")(  # pytype: disable=wrong-arg-types
              x, inputs, padding_mask, train)
      all_attn_weights.append(attn_weights)

    attn_weights = jnp.stack(all_attn_weights)
    if isinstance(queries, TransformerOutput):
      out = TransformerOutput(
        factors=x,
        attn=jnp.concatenate(
          (prior_attn_weights, attn_weights))
      )
    else:
      out = TransformerOutput(
        factors=x,
        attn=attn_weights)

    return out


class TransformerBlock(hk.Module):
  """Transformer decoder block.
  
  Typical RL settings:
  pre_norm=True,
  gate_factory=GRU.
  """
  
  def __init__(self,
               num_heads: int,
               qkv_size: int,
               mlp_size: int,
               pre_norm: bool = False,
               w_init: Optional[hk.initializers.Initializer] = None,
               gate_factory = None,
               mlp_factory = None,
               out_mlp: bool = True,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.qkv_size = qkv_size
    self.mlp_size = mlp_size
    self.pre_norm = pre_norm
    self.w_init = w_init
    self.out_mlp = out_mlp
    if gate_factory is None:
      gate_factory = sum_gate_factory
    self.gate_factory = gate_factory

    if mlp_factory is None:
      mlp_factory = lambda: encoder.Mlp(
        mlp_layers=[self.mlp_size],
        w_init=self.w_init)
    self.mlp_factory = mlp_factory


  def __call__(self,
               queries: Array,
               inputs: Optional[Array] = None,
               padding_mask: Optional[Array] = None,
               train: bool = False) -> Array:
    del padding_mask  # Unused.
    assert queries.ndim in (2, 3), 'must be [T, D] or [B, T, D]'

    attn = functools.partial(
        GeneralMultiHeadAttention,
        num_heads=self.num_heads,
        key_size=self.qkv_size,
        model_size=self.qkv_size,
        w_init=self.w_init,
        attn_weights=True)

    mlp = self.mlp_factory()

    if self.qkv_size != queries.shape[-1]:
      queries = hk.Linear(self.qkv_size)(queries)

    if self.pre_norm:
      # THIS IS WHAT'S USED IN REINFORCEMENT LEARNING
      # Self-attention on queries.
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(queries)
      if inputs is not None:
        key = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(inputs)
      else:
        key = x
      x, attn_weights = attn()(query=x, key=key)
      x = self.gate_factory()(x, queries)

      y = x
      if self.out_mlp:
        x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)
        # MLP
        y = mlp(x)
        y = self.gate_factory()(x, y)
    else:
      # Self-attention on queries.
      x = queries
      if inputs is not None:
        key = hk.LayerNorm(axis=(-1), create_scale=True,
                           create_offset=True)(inputs)
      else:
        key = x

      x, attn_weights = attn()(query=x, key=key)
      x = self.gate_factory()(x, queries)
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)

      y = x
      if self.out_mlp:
        y = mlp(x)
        y = self.gate_factory()(x, y)
        y = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(y)
    return y, attn_weights



class GeneralMultiHeadAttention(hk.MultiHeadAttention):
  """Attention function is input now."""

  def __init__(self, *args,
               attn_weights: bool = True, 
               attention_fn: hk.Module = None,
               project_out: bool = True,
               clip_attn_probs: bool = True,
               epsilon: float = 1e-8,
               **kwargs):
    super().__init__(*args, **kwargs)
    if attention_fn is None:
      attention_fn = GeneralizedDotProductAttention(
        attn_weights=attn_weights,
        clip_attn_probs=clip_attn_probs,
        epsilon=epsilon)
    self.attn_weights = attn_weights
    self.attention_fn = attention_fn
    self.project_out = project_out

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Computes (optionally masked) MHA with queries, keys & values.

    This module broadcasts over zero or more 'batch-like' leading dimensions.

    Args:
      query: Embeddings sequence used to compute queries; shape [..., T', D_q].
      key: Embeddings sequence used to compute keys; shape [..., T, D_k].
      value: Embeddings sequence used to compute values; shape [..., T, D_v].
      mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

    Returns:
      A new sequence of embeddings, consisting of a projection of the
        attention-weighted value projections; shape [..., T', D'].
    """
    if value is None:
      value = key

    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
    key_heads = projection(key, self.key_size, "key")  # [T, H, K]
    value_heads = projection(value, self.value_size, "value")  # [T, H, V]

    # [T, H, V]
    if self.attn_weights:
      attn, attn_weights = self.attention_fn(
        query_heads, key_heads, value_heads, mask=mask)
    else:
      attn =  self.attention_fn(query_heads, key_heads, value_heads, mask=mask)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    output = attn
    if self.project_out:
      # Apply another projection to get the final embeddings.
      final_projection = hk.Linear(self.model_size, w_init=self.w_init)
      output = final_projection(attn)

    if self.attn_weights:
      return output, attn_weights  # [T', D']

    return output


class GeneralizedDotProductAttention(hk.Module):
  """Multi-head dot-product attention with customizable normalization axis.
  This module supports logging of attention weights in a variable collection.
  """
  def __init__(self,
               epsilon: float = 1e-8,
               inverted_attn: bool = False,
               renormalize_keys: bool = False,
               attn_weights: bool = False,
               temperature: bool = 1.0,
               value_combination: str = 'avg',
               clip_attn_probs: bool = False,
               dtype = jnp.float32,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.dtype = dtype
    self.epsilon = epsilon
    self.inverted_attn = inverted_attn
    self.renormalize_keys = renormalize_keys
    self.attn_weights = attn_weights
    del temperature
    self.value_combination = value_combination
    self.clip_attn_probs = clip_attn_probs,

  def __call__(self, query: Array, key: Array, value: Array,
               train: bool = False,
               mask: Optional[jnp.ndarray] = None,
               **kwargs) -> Array:
    """Computes multi-head dot-product attention given query, key, and value.
    Args:
      query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
      key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
      value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
      train: Indicating whether we're training or evaluating.
      **kwargs: Additional keyword arguments are required when used as attention
        function in nn.MultiHeadDotProductAttention, but they will be ignored
        here.
    Returns:
      Output of shape `[..., q_num, num_heads, v_features]`.
    """
    del train

    assert query.ndim == key.ndim == value.ndim, (
        "Queries, keys, and values must have the same rank.")
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        "Query, key, and value batch dimensions must match.")
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        "Query, key, and value num_heads dimensions must match.")
    assert key.shape[-3] == value.shape[-3], (
        "Key and value cardinality dimensions must match.")
    assert query.shape[-1] == key.shape[-1], (
        "Query and key feature dimensions must match.")

    if kwargs.get("bias") is not None:
      raise NotImplementedError(
          "Support for masked attention is not yet implemented.")

    if "dropout_rate" in kwargs:
      if kwargs["dropout_rate"] > 0.:
        raise NotImplementedError(
            "Support for dropout is not yet implemented.")

    # Temperature normalization.
    qk_features = query.shape[-1]
    query = query / jnp.sqrt(qk_features).astype(self.dtype)

    # attn.shape = (batch..., num_heads, q_num, kv_num)
    attn = jnp.einsum("...qhd,...khd->...hqk", query, key)
    if mask is not None:
      if mask.ndim != attn.ndim:
        raise ValueError(
            f"Mask dimensionality {mask.ndim} must match logits dimensionality "
            f"{attn.ndim}."
        )
      print('never tested')
      import ipdb; ipdb.set_trace()
      attn = jnp.where(mask, attn, -1e30)

    if self.inverted_attn:
      attention_axis = -2  # Query axis.
    else:
      attention_axis = -1  # Key axis.

    # Softmax normalization (by default over key axis).
    attn = jax.nn.softmax(attn, axis=attention_axis).astype(self.dtype)
    if self.clip_attn_probs:
      attn = jnp.clip(attn, 0, 1)  # for numerical stability

    if self.renormalize_keys:
      # Corresponds to value aggregation via weighted mean (as opposed to sum).
      normalizer = jnp.sum(attn, axis=-1, keepdims=True) + self.epsilon
      attn = attn / normalizer

    if self.value_combination == 'avg':
      # Aggregate values using a weighted sum with weights provided by `attn`.
      # output is Q, H, D (i.e. one output per query, e.g. slot query in savi)
      outputs = jnp.einsum("...hqk,...khd->...qhd", attn, value)

    elif self.value_combination == 'product':
      # value: 
      # [..., Keys, Heads, Dim] --> [..., Heads, Keys, Dim]
      value = jnp.swapaxes(value, -3, -2)

      # attn: 
      # [..., Heads, Queries, Keys] --> [..., Heads, Queries, Keys, 1]
      attn_expanded = jnp.expand_dims(attn, -1)

      # conforms to ordering of einsum
      multiply = jax.vmap(jnp.multiply, in_axes=(-3, None), out_axes=(-4))

      # output: [..., Queries, Heads, Keys, Dim]
      outputs = multiply(attn_expanded, value)

      # output: [..., Queries, Heads, spatial, spatial, Dim]
      height = width = int(math.sqrt(outputs.shape[-2]))
      outputs = outputs.reshape(*outputs.shape[:-2], height, width, -1)

      # collapse everything except last 3 (H, W, C)
      batch_dims = outputs.ndim - 3
      def batch(f): return hk.BatchApply(f, num_dims=batch_dims)

      outputs = batch(hk.Conv2D(qk_features, 3, 3))(outputs)
      outputs = jax.nn.relu(outputs)
      outputs = batch(hk.Conv2D(qk_features, 3, 3))(outputs)

      # output: [..., Queries, Heads, Keys*Dim]
      outputs = outputs.reshape(*outputs.shape[:-3], -1)

    if self.attn_weights:
      return outputs, attn

    return outputs

