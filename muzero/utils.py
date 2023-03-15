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
