
from typing import Optional, Tuple, Iterator, Optional

import acme
from acme import adders

from acme import core
from acme import specs
from acme.agents.jax import actors
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import observation_action_reward

import dataclasses
import haiku as hk
import jax
import numpy as np
import optax
import reverb
import rlax

from modules import vision_language

from experiments.utils import update_config



@dataclasses.dataclass
class R2D2Config(r2d2.R2D2Config):
  # Architecture
  vocab_size: int = 50  # vocab size for env
  sentence_dim: int = 128  # dimensionality of sentence embeddings
  word_dim: int = 128  # dimensionality of word embeddings
  task_dim: int = 128  # projection of task to lower dimension
  state_dim: int = 256
  q_dim: int = 512
  conv_out_dim: int = 0

  # value-based action-selection options
  num_epsilons: int = 256
  epsilon_min: float = 1
  epsilon_max: float = 3
  epsilon_base: float = .1

  # Learner options
  discount: float = 0.99
  burn_in_length: int = 0
  num_steps: int = 3e6
  seed: int = 1
  max_grad_norm: float = 80.0
  adam_eps: float = 1e-3

  # Replay options
  # samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 10.0
  min_replay_size: int = 1_000
  max_replay_size: int = 80_000
  batch_size: Optional[int] = 64
  trace_length: Optional[int] = 20
  sequence_period: Optional[int] = 20
  prefetch_size: int = 0
  num_parallel_calls: int = 1

  # Priority options
  importance_sampling_exponent: float = 0.0
  priority_exponent: float = 0.0


class R2D2Arch(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
               torso: vision_language.Torso,
               memory: hk.RNNCore,
               head: hk.Module,
               name: str = 'r2d2_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._torso(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._memory(embeddings, state)
    q_values = self._head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(
      self._memory, embeddings, state)
    q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return q_values, new_states


class R2D2Builder(r2d2.R2D2Builder):

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec
    optimizer_chain = []
    if self._config.max_grad_norm:
      optimizer_chain.append(
        optax.clip_by_global_norm(self._config.max_grad_norm))
    optimizer_chain.append(
      optax.adam(self._config.learning_rate, eps=self._config.adam_eps)),
    # The learner updates the parameters (and initializes them).
    return r2d2_learning.R2D2Learner(
        networks=networks,
        batch_size=self._batch_size_per_device,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        optimizer=optax.chain(*optimizer_chain),
        bootstrap_n=self._config.bootstrap_n,
        tx_pair=self._config.tx_pair,
        clip_rewards=self._config.clip_rewards,
        replay_client=replay_client,
        counter=counter,
        logger=logger_fn('learner'))

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: r2d2_actor.R2D2Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    # Create variable client.
    variable_client = variable_utils.VariableClient(
        variable_source,
        key='actor_variables',
        update_period=self._config.variable_update_period)

    return actors.GenericActor(
        policy, random_key, variable_client, adder, backend='cpu')

  def make_policy(self,
                  networks: r2d2_networks.R2D2Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> r2d2_actor.R2D2Policy:

    return get_actor_core(
        networks,
        config=self._config,
        evaluation=evaluation)


def get_actor_core(
    networks: r2d2_networks.R2D2Networks,
    config: R2D2Config,
    evaluation: bool = False,
) -> r2d2_actor.R2D2Policy:
  """Returns ActorCore for R2D2."""

  num_epsilons = config.num_epsilons
  evaluation_epsilon = config.evaluation_epsilon
  if (not num_epsilons and evaluation_epsilon is None) or (num_epsilons and
                                                           evaluation_epsilon):
    raise ValueError(
        'Exactly one of `num_epsilons` or `evaluation_epsilon` must be '
        f'specified. Received num_epsilon={num_epsilons} and '
        f'evaluation_epsilon={evaluation_epsilon}.')

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: r2d2_actor.R2D2ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    q_values, recurrent_state = networks.apply(params, policy_rng, observation,
                                               state.recurrent_state)
    action = rlax.epsilon_greedy(state.epsilon).sample(policy_rng, q_values)

    return action, r2d2_actor.R2D2ActorState(
        rng=rng,
        epsilon=state.epsilon,
        recurrent_state=recurrent_state,
        prev_recurrent_state=state.recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> r2d2_actor.R2D2ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    if not evaluation:
      epsilon = jax.random.choice(epsilon_rng,
                                  np.logspace(config.epsilon_min, config.epsilon_max, config.num_epsilons, base=config.epsilon_base))
    else:
      epsilon = evaluation_epsilon
    initial_core_state = networks.init_recurrent_state(state_rng, None)
    return r2d2_actor.R2D2ActorState(
        rng=rng,
        epsilon=epsilon,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
      state: r2d2_actor.R2D2ActorState[actor_core_lib.RecurrentState]
      ) -> r2d2_actor.R2D2Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)
