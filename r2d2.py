import functools

from typing import Optional, Tuple, Iterator, Optional

from acme import core
from acme.jax.networks import base
from acme import specs
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax.networks import duelling
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import observation_action_reward

import dataclasses
import haiku as hk
import optax
import reverb

from modules import vision
from modules import language
from modules import vision_language


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

  # Learner options
  discount: float = 0.99
  burn_in_length: int = 0
  trace_length: int = 20
  num_steps: int = 3e6
  seed: int = 1
  max_gradient_norm: float = 80.0
  adam_eps: float = 1e-3

  # Replay options
  # samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 50.0
  min_replay_size: int = 1_000
  max_replay_size: int = 80_000
  batch_size: Optional[int] = 64
  trace_length: Optional[int] = 20
  prefetch_size: int = 0
  num_parallel_calls: int = 1

  # Priority options
  importance_sampling_exponent: float = 0.0
  priority_exponent: float = 0.9


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
    optimizer_chain = [  # Change: adding clipping + eps
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.adam(self._config.learning_rate, eps=self._config.adam_eps),
    ]
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


def make_r2d2_babyai_networks(
        env_spec: specs.EnvironmentSpec,
        config: r2d2.R2D2Config) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values
  vision_torso = vision.BabyAIVisionTorso(conv_dim=config.conv_out_dim, flatten=False)
  task_encoder = language.LanguageEncoder(
          vocab_size=config.vocab_size,
          word_dim=config.word_dim,
          sentence_dim=config.sentence_dim,
      )
  def make_core_module() -> R2D2Arch:
    observation_fn = vision_language.Torso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder=task_encoder,
      image_dim=config.state_dim,
      task_dim=config.task_dim,
    )
    return R2D2Arch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=duelling.DuellingMLP(num_actions,
                                hidden_sizes=[config.q_dim]))

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)


def make_r2d2_builder(
    launch: bool=True,
    config_kwargs: dict = None):
  if not launch: #DEBUG
    config_kwargs['min_replay_size'] = 100
    config_kwargs["samples_per_insert"] = 1.0
    config_kwargs['batch_size'] = 2
    config_kwargs['trace_length'] = 6
    config_kwargs['discount'] = .99
    config_kwargs['bootstrap_n'] = 3
    config_kwargs['burn_in_length'] = 0

  config = R2D2Config()

  builder = r2d2.R2D2Builder(config)

  network_factory = functools.partial(
          make_r2d2_babyai_networks, config=config)
  
  return config, builder, network_factory
