
from typing import Generic

from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.jax import networks as networks_lib
import chex
import jax
import jax.numpy as jnp

from muzero import types

@chex.dataclass(frozen=True, mappable_dataclass=False)
class MuZeroActorState(Generic[actor_core_lib.RecurrentState]):
  rng: networks_lib.PRNGKey
  recurrent_state: actor_core_lib.RecurrentState
  prev_recurrent_state: actor_core_lib.RecurrentState


R2D2Policy = actor_core_lib.ActorCore[
    MuZeroActorState[actor_core_lib.RecurrentState], actor_core_lib.Extras]


def get_actor_core(
    networks: types.MuZeroNetworks,
    evaluation: bool = True,
) -> R2D2Policy:
  """Returns ActorCore for MuZero."""

  def select_action(params: types.MuZeroParams,
                    observation: networks_lib.Observation,
                    state: MuZeroActorState[actor_core_lib.RecurrentState]):
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

  def init(
      rng: networks_lib.PRNGKey
  ) -> MuZeroActorState[actor_core_lib.RecurrentState]:
    rng, state_rng = jax.random.split(rng, 2)
    initial_core_state = networks.init_recurrent_state(state_rng, None)
    return MuZeroActorState(
        rng=rng,
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
