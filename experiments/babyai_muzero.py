from absl import logging

import functools
import distrax
import jax
import mctx
from pprint import pprint
import rlax

from muzero import utils as muzero_utils
from muzero import networks as muzero_networks
from muzero.builder import MuZeroBuilder
from muzero.config import MuZeroConfig
from muzero.ve_losses import ValueEquivalentLoss


def setup(
    launch: bool=True,
    config_kwargs: dict = None):
  config_kwargs = config_kwargs or dict()
  if not launch: #DEBUG
    config_kwargs.update(
      min_replay_size=100,
      # samples_per_insert=1.0,
      # batch_size=4,
      # trace_length=6,
      # simulation_steps=2,
      # num_simulations=1,
      # td_steps=3,
      burn_in_length=0,
      # weight_decay=0.0,
      # show_gradients=0,
      # model_learn_prob=.1,
    )
  logging.info(f'Config arguments')
  pprint(config_kwargs)

  config = MuZeroConfig(**config_kwargs)

  if config.sequence_period is None:
    config.sequence_period = config.trace_length

  discretizer = muzero_utils.Discretizer(
      num_bins=config.num_bins,
      step_size=config.scalar_step_size,
      max_value=config.max_scalar_value,
      tx_pair=config.tx_pair,
  )
  config.num_bins = discretizer._num_bins

  muzero_policy = functools.partial(
      mctx.gumbel_muzero_policy,
      max_depth=config.max_sim_depth,
      gumbel_scale=config.gumbel_scale)

  policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)

  ve_loss_fn = functools.partial(ValueEquivalentLoss,
    muzero_policy=muzero_policy,
    policy_loss_fn=policy_loss_fn,
    simulation_steps=config.simulation_steps,
    discretizer=discretizer,
    num_simulations=config.num_simulations,
    discount=config.discount,
    td_steps=config.td_steps,
    model_coef=config.model_coef,
    policy_coef=config.policy_coef,
    root_policy_coef=config.root_policy_coef,
    value_coef=config.value_coef,
    reward_coef=config.reward_coef,
    v_target_source=config.v_target_source,
    # reanalyze_ratio=config.reanalyze_ratio,
    # metrics=config.metrics,
  )

  builder = MuZeroBuilder(config, loss_fn=ve_loss_fn)

  network_factory = functools.partial(
      muzero_networks.make_babyai_networks,
      config=config,
      discretizer=discretizer)
  
  return config, builder, network_factory