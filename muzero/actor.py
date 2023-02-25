
import functools

from typing import Generic

from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.jax import networks as networks_lib
import chex
import jax
import numpy as np
import jax.numpy as jnp
import rlax
from muzero.utils import (
    logits_to_scalar,
)

from muzero import types
from muzero.config import MuZeroConfig

@chex.dataclass(frozen=True, mappable_dataclass=False)
class MuZeroActorState(Generic[actor_core_lib.RecurrentState]):
  rng: networks_lib.PRNGKey
  recurrent_state: actor_core_lib.RecurrentState
  prev_recurrent_state: actor_core_lib.RecurrentState
  epsilon: jnp.ndarray = None


R2D2Policy = actor_core_lib.ActorCore[
    MuZeroActorState[actor_core_lib.RecurrentState], actor_core_lib.Extras]

def policy_select_action(
    params: types.MuZeroParams,
    observation: networks_lib.Observation,
    state: MuZeroActorState[actor_core_lib.RecurrentState],
    networks: types.MuZeroNetworks,
    evaluation: bool = True):
  rng, policy_rng = jax.random.split(state.rng)

  logits, recurrent_state = networks.apply(params.unroll, policy_rng, observation,
                                              state.recurrent_state)
  if evaluation:
    action = jnp.argmax(logits.policy_logits, axis=-1)
  else:
    action = jax.random.categorical(policy_rng, logits.policy_logits)

  return action, MuZeroActorState(
      rng=rng,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def value_select_action(
    params: types.MuZeroParams,
    observation: networks_lib.Observation,
    state: MuZeroActorState[actor_core_lib.RecurrentState],
    networks: types.MuZeroNetworks,
    num_bins: int,
    discount: float,
    tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
    evaluation: bool = True):
  # TODO: jit this function if possible.
  rng, policy_rng = jax.random.split(state.rng)

  logits, recurrent_state = networks.apply(params.unroll, policy_rng, observation,
                                           state.recurrent_state)
  # V(s_t)
  value = logits_to_scalar(logits.value_logits, num_bins)
  value = tx_pair.apply_inv(value)

  actions = jnp.arange(logits.policy_logits.shape[-1])
  next_logits, _ = jax.vmap(
    networks.apply_model, in_axes=(None, None, None, 0))(
      params.model, policy_rng, recurrent_state.hidden, actions)

  # r(s_t, a_t, s_t+1)
  reward = logits_to_scalar(next_logits.reward_logits, num_bins)
  reward = tx_pair.apply_inv(reward)

  # Q(s_t, a_t) = V(s_t) + gamma*r(s_t, a_t, s_t+1)
  q_values = value[None] + discount*reward

  action = rlax.epsilon_greedy(state.epsilon).sample(policy_rng, q_values)

  return action, MuZeroActorState(
      rng=rng,
      epsilon=state.epsilon,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)


def get_actor_core(
    networks: types.MuZeroNetworks,
    config: MuZeroConfig,
    evaluation: bool = True,
) -> R2D2Policy:
  """Returns ActorCore for MuZero."""
  
  assert config.action_source in ['policy', 'value', 'mcts']
  if config.action_source == 'policy':
    select_action = functools.partial(policy_select_action,
                                      networks=networks,
                                      evaluation=evaluation)
  elif config.action_source == 'value':
    num_epsilons = config.num_epsilons
    evaluation_epsilon = config.evaluation_epsilon
    select_action = functools.partial(value_select_action,
                                      networks=networks,
                                      tx_pair=config.tx_pair,
                                      num_bins=config.num_bins,
                                      discount=config.discount)
  elif config.action_source == 'mcts':
    raise NotImplementedError

  def init(
      rng: networks_lib.PRNGKey
  ) -> MuZeroActorState[actor_core_lib.RecurrentState]:
    rng, state_rng = jax.random.split(rng, 2)
    initial_core_state = networks.init_recurrent_state(state_rng, None)
    if config.action_source == 'value':
      if evaluation:
        epsilon = evaluation_epsilon
      else:
        rng, epsilon_rng = jax.random.split(rng, 2)
        epsilon = jax.random.choice(epsilon_rng,
                                    np.logspace(1, 3, num_epsilons, base=0.1))
    else:
      epsilon = None
    return MuZeroActorState(
        rng=rng,
        epsilon=epsilon,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
      state: MuZeroActorState[actor_core_lib.RecurrentState]) -> actor_core_lib.Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)


# TODO(wilka): Deprecate this in favour of MuZeroBuilder.make_policy.
def make_behavior_policy(networks: types.MuZeroNetworks,
                         config: r2d2_config.R2D2Config,
                         evaluation: bool = False) -> actor_core_lib.RecurrentPolicy:
  """Selects action according to the policy."""

  def behavior_policy(params: types.MuZeroParams,
                      key: networks_lib.PRNGKey,
                      observation: types.NestedArray,
                      core_state: types.NestedArray):
    logits, core_state = networks.apply(params.unroll, key, observation, core_state)
    rng, policy_rng = jax.random.split(key)
    if evaluation:
      action = jnp.argmax(logits.policy_logits, axis=-1)
    else:
      action = jax.random.categorical(policy_rng, logits.policy_logits)
    return action, core_state

  return behavior_policy
