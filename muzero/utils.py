"""Agent utilities."""

from typing import Optional, Callable
from acme.jax import networks as networks_lib
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import math
import rlax
import mctx

from acme.types import NestedArray as Array
from muzero import types as muzero_types

State = Array
Action = Array

from muzero.types import TaskAwareRep, ModelOutput

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
               clip_probs: bool = False,
               tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR):
    self._max_value = max_value
    self._min_value = min_value if min_value is not None else -max_value
    self._clip_probs = clip_probs
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
     if self._clip_probs:
      probs = jnp.clip(probs, 0, 1)  # for numerical stability
    #  total = jnp.sum(probs, axis=-1, keepdims=True)
    #  probs = probs/total
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
                    **unused_kwargs) -> TaskAwareRep:
    if not self._couple_state_task:
      return self._core.initial_state(batch_size)

    if self._task_dim is None:
      raise RuntimeError("Don't expect to initialize state")

    state = TaskAwareRep(
      rep=self._core.initial_state(None),
      task=jnp.zeros(self._task_dim, dtype=jnp.float32)
    )
    if batch_size:
      state = add_batch(state, batch_size)
    return state

  def __call__(self, inputs: Array, prev_state: TaskAwareRep):
    prepped_input = self._prep_input(inputs)
    prepped_state = self._prep_state(prev_state)

    hidden, state = self._core(prepped_input, prepped_state)

    task = self._get_task(inputs, prev_state)
    if self._couple_state_task:
      state = TaskAwareRep(rep=state, task=task)

    if self._couple_hidden_task:
      hidden = TaskAwareRep(rep=hidden, task=task)

    return hidden, state


def compute_q_values(state: State,
                     discount: float,
                     num_actions: int,
                     apply_model: Callable[[State, Action], ModelOutput],
                     discretizer: Discretizer,
                     invalid_actions=None):
  """Compute Q(s,a) = r(s,a,s') + gamma*Value(s').

  Unroll 1-step model, compute value at next-time, and combine
  with current reward.

  Args:
      state (jnp.ndarray): [D]
      value_logits (jnp.ndarray): [N]
      policy_logits (jnp.ndarray): [A]

  Returns:
      jnp.ndarray: Q(s,a) [A]
  """
  actions = jnp.arange(num_actions, dtype=jnp.int32)
  # vmap over action dimension, to get 1 estimate per action
  apply_model = jax.vmap(apply_model, in_axes=(None, 0))

  # [A, ...]
  next_logits, _ = apply_model(state, actions)

  value = discretizer.logits_to_scalar(next_logits.value_logits)
  reward = discretizer.logits_to_scalar(next_logits.reward_logits)

  q_value = reward + discount*value
  if invalid_actions is not None:
    q_value = jnp.where(invalid_actions>0, -1e10, q_value)

  return q_value


def model_step(params: networks_lib.Params,
               rng_key: chex.Array,
               action: chex.Array,
               state: chex.Array,
               discount: chex.Array,
               networks: muzero_types.MuZeroNetworks,
               discretizer: Discretizer):
  """One simulation step in MCTS."""
  rng_key, model_key = jax.random.split(rng_key)
  model_output, next_state = networks.apply_model(
      params, model_key, state, action,
  )
  reward = discretizer.logits_to_scalar(model_output.reward_logits)
  value = discretizer.logits_to_scalar(model_output.value_logits)

  recurrent_fn_output = mctx.RecurrentFnOutput(
      reward=reward,
      discount=discount,
      prior_logits=model_output.policy_logits,
      value=value,
  )
  return recurrent_fn_output, next_state