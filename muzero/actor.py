
import functools

from typing import Generic

from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.jax import networks as networks_lib
import chex
import distrax
import jax
import numpy as np
import jax.numpy as jnp

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
    model_share_params: bool=True,
    evaluation: bool = True):
  rng, policy_rng = jax.random.split(state.rng)

  params = params if model_share_params else params.unroll
  logits, recurrent_state = networks.apply(params, policy_rng, observation,
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
    discount: float):
  del discount
  rng, policy_rng = jax.random.split(state.rng)

  predictions, recurrent_state = networks.apply(params,
                                                policy_rng,
                                                observation,
                                                state.recurrent_state)
  # assert predictions.q_values is not None, 'network needs to return q_values'
  # Q(s_t, a_t) = r(s_t, a_t, s_t+1) + gamma*V(s_t+1)
  rng, sample_rng = jax.random.split(rng)
  action = distrax.EpsilonGreedy(
    predictions.q_value, state.epsilon, dtype=jnp.int32).sample(seed=sample_rng)

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
    model_share_params = config.action_source == "value" or not config.seperate_model_nets
    select_action = functools.partial(policy_select_action,
                                      networks=networks,
                                      evaluation=evaluation,
                                      model_share_params=model_share_params)
  elif config.action_source == 'value':
    select_action = functools.partial(value_select_action,
                                      networks=networks,
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
        epsilon = config.evaluation_epsilon
      else:
        rng, epsilon_rng = jax.random.split(rng, 2)
        epsilon = jax.random.choice(epsilon_rng,
                                    np.logspace(config.epsilon_min, config.epsilon_max, config.num_epsilons, base=0.1))
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

  return actor_core_lib.ActorCore(init=init,
                                  select_action=select_action,
                                  get_extras=get_extras)
