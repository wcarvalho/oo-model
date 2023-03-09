"""Agent utilities."""
# Codes adapted from:
# https://github.com/Hwhitetooth/jax_muzero/blob/main/algorithms/utils.py

from typing import Optional
import chex
import jax
import jax.numpy as jnp
import math
import rlax
# only change is below
from acme.types import NestedArray as Array

def scale_gradient(g: Array, scale: float) -> Array:
    """Scale the gradient.

    Args:
        g (_type_): Parameters that contain gradients.
        scale (float): Scale.

    Returns:
        Array: Parameters with scaled gradients.
    """
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)

# create class

class Discretizer:
  def __init__(self,
               max_value,
               num_bins: Optional[int] = None,
               step_size: Optional[int] = None,
               min_value: Optional[int] = None,
               tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR):
    self._max_value = max_value
    self._min_value = min_value if min_value is not None else -max_value
    if step_size is None:
      assert num_bins is not None
    else:
      num_bins = math.ceil((self._max_value-self._min_value)/step_size)+1

    self._num_bins = num_bins
    self._tx_pair = tx_pair

  def logits_to_scalar(self, logits):
     return self.probs_to_scalar(jax.nn.softmax(logits))

  def probs_to_scalar(self, probs):
     scalar = rlax.transform_from_2hot(
        probs=probs,
        min_value=self._min_value,
        max_value=self._max_value,
        num_bins=self._num_bins)
     unscaled_scalar = self._tx_pair.apply_inv(scalar)
     return unscaled_scalar

  def scalar_to_probs(self, scalar):
     scaled_scalar = self._tx_pair.apply(scalar)
     probs = rlax.transform_to_2hot(
        scalar=scaled_scalar,
        min_value=self._min_value,
        max_value=self._max_value,
        num_bins=self._num_bins)
     return probs

def scalar_to_two_hot(x: Array,
                      num_bins: int,
                      max_value: float = 50.0) -> Array:
    """A categorical representation of real values.

    Ref: https://www.nature.com/articles/s41586-020-03051-4.pdf.

    Args:
        x (Array): Scalar data.
        num_bins (int): Number of bins.

    Returns:
        Array: Distributional data.
    """
    return rlax.transform_to_2hot(x, -max_value, max_value, num_bins)
    # max_val = (num_bins - 1) // 2
    # x = jnp.clip(x, -max_val, max_val)
    # x_low = jnp.floor(x).astype(jnp.int32)
    # x_high = jnp.ceil(x).astype(jnp.int32)
    # p_high = x - x_low
    # p_low = 1.0 - p_high
    # idx_low = x_low + max_val
    # idx_high = x_high + max_val
    # cat_low = jax.nn.one_hot(idx_low, num_bins) * p_low[..., None]
    # cat_high = jax.nn.one_hot(idx_high, num_bins) * p_high[..., None]
    # return cat_low + cat_high


def logits_to_scalar(logits: Array,
                     num_bins: int,
                     max_value: float = 50.0) -> Array:
    """The inverse of the scalar_to_two_hot function above.

    Args:
        logits (Array): Distributional logits.
        num_bins (int): Number of bins.

    Returns:
        Array: Scalar data.
    """
    return rlax.transform_from_2hot(jax.nn.softmax(logits), -max_value, max_value, num_bins)
    # chex.assert_equal(num_bins, logits.shape[-1])
    # max_val = (num_bins - 1) // 2
    # x = jnp.sum((jnp.arange(num_bins) - max_val) * jax.nn.softmax(logits), axis=-1)
    # return x


def value_transform(x: Array, epsilon: float = 1e-3) -> Array:
    """A non-linear value transformation for variance reduction.

    Ref: https://arxiv.org/abs/1805.11593.

    Args:
        x (Array): Data.
        epsilon (float, optional): Epsilon. Defaults to 1e-3.

    Returns:
        Array: Transformed data.
    """
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x


def inv_value_transform(x: Array, epsilon: float = 1e-3) -> Array:
    """The inverse of the non-linear value transformation above.

    Args:
        x (Array): Data.
        epsilon (float, optional): Epsilon. Defaults to 1e-3.

    Returns:
        Array: Inversely transformed data.
    """
    return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon))
        ** 2
        - 1
    )
