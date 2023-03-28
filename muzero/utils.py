"""Agent utilities."""

from typing import Optional, Callable
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import math
import rlax

from acme.types import NestedArray as Array

from muzero.types import TaskAwareState

def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_util.tree_map(broadcast, nest)

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

class TaskAwareRecurrentFn(hk.RNNCore):
  """Helper RNN which adds task to state and, optionally, hidden output. 
  
  It's useful to couple the task g and the state s_t output for functions that make
    predictions at each time-step. For example:
      - value predictions: V(s_t, g)
      - policy: pi(s_t, g)
    If you use this class with hk.static_unroll, then the hidden output will have s_t 
      and g of the same dimension.
      i.e. ((s_1, g), (s_2, g), ..., (s_k, g))
  """
  def __init__(self,
               core: hk.RNNCore,
               task_dim: Optional[int] = None,
               couple_state_task: bool = False,
               couple_hidden_task: bool = True,
               get_task: Callable[
                   [Array, Array], Array] = lambda inputs, state: state.task,
               prep_input: Callable[
                  [Array], Array] = lambda x: x,
               prep_state: Callable[[Array], Array] = lambda x: x,
               name: Optional[str] = None
               ):
    super().__init__(name=name)
    self._core = core
    self._task_dim = task_dim
    self._get_task = get_task
    self._couple_state_task = couple_state_task
    self._couple_hidden_task = couple_hidden_task
    self._prep_input = prep_input
    self._prep_state = prep_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> TaskAwareState:
    if not self._couple_state_task:
      return self._core.initial_state(batch_size)

    if self._task_dim is None:
      raise RuntimeError("Don't expect to initialize state")

    state = TaskAwareState(
      state=self._core.initial_state(None),
      task=jnp.zeros(self._task_dim, dtype=jnp.float32)
    )
    if batch_size:
      state = add_batch(state, batch_size)
    return state

  def __call__(self, inputs: Array, prev_state: TaskAwareState):
    prepped_input = self._prep_input(inputs)
    prepped_state = self._prep_state(prev_state)

    hidden, state = self._core(prepped_input, prepped_state)

    task = self._get_task(inputs, prev_state)
    if self._couple_state_task:
      state = TaskAwareState(state=state, task=task)

    if self._couple_hidden_task:
      hidden = TaskAwareState(state=hidden, task=task)

    return hidden, state