import functools
from acme import specs
from acme.agents.jax import r2d2
from acme.jax import networks as networks_lib
from acme.jax.networks import duelling

import haiku as hk

from modules import vision
from modules import language
from modules import vision_language

from r2d2 import R2D2Config, R2D2Arch, R2D2Builder


def make_r2d2_babyai_networks(
        env_spec: specs.EnvironmentSpec,
        config: R2D2Config) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values
  def make_core_module() -> R2D2Arch:
    vision_torso = vision.BabyAIVisionTorso(conv_dim=config.conv_out_dim)
    task_encoder = language.LanguageEncoder(
            vocab_size=config.vocab_size,
            word_dim=config.word_dim,
            sentence_dim=config.sentence_dim,
        )
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


def setup(
    debug: bool=False,
    config_kwargs: dict = None):
  config_kwargs = config_kwargs or dict()
  if debug: #DEBUG
    config_kwargs['min_replay_size'] = 100
    config_kwargs["samples_per_insert"] = 1.0
    config_kwargs['batch_size'] = 2
    config_kwargs['burn_in_length'] = 0
    config_kwargs['trace_length'] = 6
    config_kwargs['sequence_period'] = 6
    config_kwargs['discount'] = .99
    config_kwargs['bootstrap_n'] = 3

  config = R2D2Config(**config_kwargs)

  builder = R2D2Builder(config)

  network_factory = functools.partial(
          make_r2d2_babyai_networks, config=config)
  
  return config, builder, network_factory
