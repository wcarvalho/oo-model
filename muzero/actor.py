
import functools

from typing import Generic, Union, Optional, Callable, Any

from acme import types
from acme.agents.jax import actors
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.jax import networks as networks_lib
import chex
import distrax
import dm_env
import jax
import numpy as np
import jax.numpy as jnp

NestedArray = Any

from muzero import types
from muzero.config import MuZeroConfig
from muzero import utils
import mctx

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

  rng, policy_rng = jax.random.split(state.rng)
  q_values = networks.compute_q_values(
    params, policy_rng, predictions.state)

  rng, sample_rng = jax.random.split(rng)
  action = distrax.EpsilonGreedy(
      q_values, state.epsilon, dtype=jnp.int32).sample(seed=sample_rng)

  return action, MuZeroActorState(
      rng=rng,
      epsilon=state.epsilon,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def mcts_select_action(
    params: networks_lib.Params,
    observation: networks_lib.Observation,
    state: MuZeroActorState[actor_core_lib.RecurrentState],
    discretizer: utils.Discretizer,
    mcts_policy: Union[mctx.muzero_policy, mctx.gumbel_muzero_policy],
    networks: types.MuZeroNetworks,
    get_state = lambda preds: preds.state,
    discount: float = .99,
    evaluation: bool = True,
    ):

  rng, policy_rng = jax.random.split(state.rng)

  preds, recurrent_state = networks.apply(
    params, policy_rng, observation, state.recurrent_state)

  value = discretizer.logits_to_scalar(preds.value_logits)

  # MCTX assumes the following shapes
  # policy_logits [B, A]
  # value [B]
  # embedding [B, D]
  # here, we have B = 1
  # i.e MCTX assumes that input has batch dimension. add fake one.
  embedding = get_state(preds)
  embedding = jax.tree_map(lambda s: s[None], embedding)
  root = mctx.RootFnOutput(prior_logits=preds.policy_logits[None],
                            value=value,
                            embedding=embedding)

  # 1 step of policy improvement
  rng, improve_key = jax.random.split(rng)
  mcts_outputs = mcts_policy(
      params=params,
      rng_key=improve_key,
      root=root,
      recurrent_fn=functools.partial(
          utils.model_step,
          discount=jnp.full(value.shape, discount),
          networks=networks,
          discretizer=discretizer,
      ))

  # batch "0"
  policy_target = mcts_outputs.action_weights[0]

  if evaluation:
    action = jnp.argmax(policy_target, axis=-1)
  else:
    action = jax.random.categorical(policy_rng, policy_target)

  return action, MuZeroActorState(
      rng=rng,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def get_actor_core(
    networks: types.MuZeroNetworks,
    config: MuZeroConfig,
    evaluation: bool = True,
    discretizer: Optional[utils.Discretizer] = None,
    mcts_policy: Optional[Union[mctx.muzero_policy, mctx.gumbel_muzero_policy]] = None,
    get_state: Callable[[NestedArray], jax.Array] = lambda state: state,
) -> R2D2Policy:
  """Returns ActorCore for MuZero."""
  
  if config.action_source == 'policy':
    select_action = functools.partial(
      policy_select_action,
      networks=networks,
      evaluation=evaluation,
      model_share_params=True)
  elif config.action_source == 'value':
    select_action = functools.partial(
      value_select_action,
      networks=networks,
      discount=config.discount)
  elif config.action_source == 'mcts':
    select_action = functools.partial(
      mcts_select_action,
      discretizer=discretizer,
      get_state=get_state,
      mcts_policy=mcts_policy,
      networks=networks,
      discount=config.discount,
      evaluation=evaluation)
  elif config.action_source == 'mcts_eval':
    if evaluation:
      select_action = functools.partial(
        mcts_select_action,
        discretizer=discretizer,
        get_state=get_state,
        mcts_policy=mcts_policy,
        networks=networks,
        discount=config.discount,
        evaluation=evaluation)
    else:
      select_action = functools.partial(
        policy_select_action,
        networks=networks,
        evaluation=evaluation,
        model_share_params=True)
  else:
    raise NotImplementedError(config.action_source)


  def init(
      rng: networks_lib.PRNGKey,
      params: types.MuZeroParams,
  ) -> MuZeroActorState[actor_core_lib.RecurrentState]:
    rng, state_rng = jax.random.split(rng, 2)
    initial_core_state = networks.init_recurrent_state(
      params, state_rng)
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


class LearnableStateActor(actors.GenericActor):
  """Only difference is to have initial state use params."""

  def observe_first(self, timestep: dm_env.TimeStep):
    self._random_key, key = jax.random.split(self._random_key)
    # NOTE: key difference is line below.
    self._state = self._init(key, self._params)
    if self._adder:
      self._adder.add_first(timestep)
    if self._variable_client and self._per_episode_update:
      self._variable_client.update_and_wait()
