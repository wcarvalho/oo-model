from absl import logging

import copy
import functools
import jax
import mctx
from pprint import pprint
import rlax

from muzero import utils as muzero_utils
from muzero.builder import MuZeroBuilder
from muzero.ve_losses import ValueEquivalentLoss

from factored_muzero import networks
from factored_muzero.config import FactoredMuZeroConfig

from experiments.config_utils import update_config

def load_config(
    config_kwargs: dict = None,
    config_class: FactoredMuZeroConfig=FactoredMuZeroConfig,
    strict_config: bool = False):
  config_kwargs = config_kwargs or dict()
  logging.info(f'Config arguments')
  pprint(config_kwargs)
  config = config_class()
  update_config(config, strict=strict_config, **config_kwargs)

  if config.sequence_period is None:
    config.sequence_period = config.trace_length

  return config

def setup(
    config: FactoredMuZeroConfig,
    network_kwargs=None,
    builder_kwargs: dict = None,
    **kwargs):
  network_kwargs = network_kwargs or dict()
  builder_kwargs = builder_kwargs or dict()

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
    root_policy_coef=config.root_policy_coef,
    root_value_coef=config.root_value_coef,
    model_policy_coef=config.model_policy_coef,
    model_value_coef=config.model_value_coef,
    model_reward_coef=config.model_reward_coef,
    v_target_source=config.v_target_source,
    # reanalyze_ratio=config.reanalyze_ratio,
    # metrics=config.metrics,
    )

  builder = MuZeroBuilder(
      config=config,
      loss_fn=ve_loss_fn,
      **builder_kwargs)

  network_factory = functools.partial(
          networks.make_babyai_networks,
          config=config,
          **network_kwargs)
  
  return builder, network_factory