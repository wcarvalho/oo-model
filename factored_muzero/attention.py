from typing import Tuple, Optional

import functools

from acme import types
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from factored_muzero import encoder

@chex.dataclass(frozen=True)
class SaviInputs:
  image: jnp.ndarray
  task: jnp.ndarray


Array = types.NestedArray

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

class SlotAttention(hk.RNNCore):
  def __init__(self,
               qkv_size: int,
               num_iterations: int = 1,
               mlp_size: Optional[int] = None,
               epsilon: float = 1e-8,
               num_heads: int = 1,
               num_slots: int = 4,
               use_task: bool = False,
               init: str = 'noise',
               name: Optional[str] = None):
    super().__init__(name=name)
    self.num_iterations = num_iterations
    self.qkv_size = qkv_size
    self.use_task = use_task
    self.mlp_size = mlp_size
    self.epsilon = epsilon
    self.num_heads = num_heads
    self.num_slots = num_slots
    self.init = init
    assert init in ('zeros', 'noise', 'embed')

  def __call__(
      self,
      inputs: SaviInputs,
      slots: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    image = inputs.image
    assert len(image.shape) in (2,3), "should either be [N, C] or [B, N, C]"
    has_batch = len(image.shape) == 3
    qkv_size = self.qkv_size or slots.shape[-1]
    head_dim = qkv_size // self.num_heads
    dense = functools.partial(MultiLinear, output_size=qkv_size, heads=self.num_heads, with_bias=False)

    # Shared modules.
    dense_q = dense(name="general_dense_q_0")
    layernorm_q = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)
    inverted_attention = InvertedDotProductAttention(
        norm_type="mean", multi_head=self.num_heads > 1)

    gru = hk.GRU(head_dim)

    if self.mlp_size is not None:
      raise NotImplementedError
      # mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

    # inputs.shape = (..., n_inputs, inputs_size).
    image = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(image)
    # k.shape = (..., n_inputs, slot_size).
    k = dense(name="general_dense_k_0")(image)
    # v.shape = (..., n_inputs, slot_size).
    v = dense(name="general_dense_v_0")(image)

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_inputs, slot_size).
      updates = inverted_attention(query=q, key=k, value=v)

      # Recurrent update.
      if has_batch:
        slots, _ = hk.BatchApply(gru)(updates, slots)
      else:
        slots, _ = gru(updates, slots)

      # Feedforward block with pre-normalization.
      if self.mlp_size is not None:
        raise NotImplementedError
        # slots = mlp(slots)
    return slots, slots

  def initial_state(self, batch_size: Optional[int] = None):
    shape = (self.num_slots, self.qkv_size)
    if batch_size is not None:
      shape = (batch_size,) + shape
    if self.init == 'zeros':
      return jnp.zeros(shape, dtype=jnp.float32)
    elif self.init == 'noise':
      return jax.random.normal(
        hk.next_rng_key(), shape, dtype=jnp.float32)
    elif self.init == 'embed':
      raise NotImplementedError("will require re-working initialization...")



class InvertedDotProductAttention(hk.Module):
  """Inverted version of dot-product attention (softmax over query axis)."""

  def __init__(self,
               norm_type: Optional[str] = "mean",  # mean, layernorm, or None
               multi_head: bool = False,
               epsilon: float = 1e-8,
               dtype = jnp.float32,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.norm_type = norm_type
    self.multi_head = multi_head
    self.epsilon = epsilon
    self.dtype = dtype
  # precision: Optional[jax.lax.Precision] = None

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

    attn = GeneralizedDotProductAttention(
        inverted_attn=True,
        renormalize_keys=True if self.norm_type == "mean" else False,
        epsilon=self.epsilon,
        dtype=self.dtype)

    # Apply attention mechanism.
    output = attn(query=query, key=key, value=value)

    if self.multi_head:
      # Multi-head aggregation. Equivalent to concat + dense layer.
      import ipdb; ipdb.set_trace()
      output = hk.Linear(output.shape[-1])(output)
      # nn.DenseGeneral(
      #     features=output.shape[-1], axis=(-2, -1))(output)
    else:
      # Remove head dimension.
      output = jnp.squeeze(output, axis=-2)

    if self.norm_type == "layernorm":
      output = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(output)

    return output


class GeneralizedDotProductAttention(hk.Module):
  """Multi-head dot-product attention with customizable normalization axis.
  This module supports logging of attention weights in a variable collection.
  """
  def __init__(self,
               epsilon: float = 1e-8,
               inverted_attn: bool = False,
               renormalize_keys: bool = False,
               attn_weights_only: bool = False,
               dtype = jnp.float32,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.dtype = dtype
    self.epsilon = epsilon
    self.inverted_attn = inverted_attn
    self.renormalize_keys = renormalize_keys
    self.attn_weights_only = attn_weights_only

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

    # Defines intermediate for logging.
    # if not train:
    #   self.sow("intermediates", "attn", attn)

    if self.renormalize_keys:
      # Corresponds to value aggregation via weighted mean (as opposed to sum).
      normalizer = jnp.sum(attn, axis=-1, keepdims=True) + self.epsilon
      attn = attn / normalizer

    if self.attn_weights_only:
      return attn

    # Aggregate values using a weighted sum with weights provided by `attn`.
    return jnp.einsum("...hqk,...khd->...qhd", attn, value)


class Transformer(hk.Module):
  """Transformer with multiple blocks."""

  def __init__(self,
               num_heads: int,
               qkv_size: int,
               mlp_size: int,
               num_layers: int,
               pre_norm: bool = False,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.qkv_size = qkv_size
    self.mlp_size = mlp_size
    self.num_layers = num_layers
    self.pre_norm = pre_norm

  def __call__(self, queries: Array, inputs: Optional[Array] = None,
               padding_mask: Optional[Array] = None,
               train: bool = False) -> Array:
    x = queries
    for lyr in range(self.num_layers):
      x = TransformerBlock(
          num_heads=self.num_heads, qkv_size=self.qkv_size,
          mlp_size=self.mlp_size, pre_norm=self.pre_norm,
          name=f"TransformerBlock{lyr}")(  # pytype: disable=wrong-arg-types
              x, inputs, padding_mask, train)
    return x


class TransformerBlock(hk.Module):
  """Transformer decoder block."""
  
  def __init__(self,
               num_heads: int,
               qkv_size: int,
               mlp_size: int,
               pre_norm: bool = False,
               name: Optional[str] = None,
               ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.qkv_size = qkv_size
    self.mlp_size = mlp_size
    self.pre_norm = pre_norm

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
        w_init=hk.initializers.VarianceScaling(2.0),
        attention_fn=GeneralizedDotProductAttention())

    mlp = encoder.Mlp(mlp_layers=[self.mlp_size])  # type: ignore

    if self.pre_norm:
      print("never checked")
      import ipdb; ipdb.set_trace()
      # Self-attention on queries.
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(queries)
      x = attn()(query=x, key=x)
      x = x + queries

      # Cross-attention on inputs.
      if inputs is not None:
        assert inputs.ndim in (2, 3), 'must be [T, D] or [B, T, D]'
        y = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)
        y = attn()(query=y, key=inputs)
        y = y + x
      else:
        y = x

      # MLP
      z = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(y)
      z = mlp(z, train)
      z = z + y
    else:
      # Self-attention on queries.
      x = queries
      x = attn()(query=x, key=x)
      x = x + queries
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)

      # Cross-attention on inputs.
      if inputs is not None:
        assert inputs.ndim in (2, 3), 'must be [T, D] or [B, T, D]'
        y = attn()(query=x, key=inputs)
        y = y + x
        y = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(y)
      else:
        y = x

      # MLP.
      z = mlp(y)
      z = z + y
      z = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(z)
    return z


class GeneralMultiHeadAttention(hk.MultiHeadAttention):
  """Attention function is input now."""

  def __init__(self, *args, attention_fn: hk.Module = None, **kwargs):
    super().__init__(*args, **kwargs)
    if attention_fn is None:
      attention_fn = GeneralizedDotProductAttention()
    self.attention_fn = attention_fn

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
    attn = self.attention_fn(query_heads, key_heads, value_heads, mask=mask)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    # Apply another projection to get the final embeddings.
    final_projection = hk.Linear(self.model_size, w_init=self.w_init)
    return final_projection(attn)  # [T', D']
