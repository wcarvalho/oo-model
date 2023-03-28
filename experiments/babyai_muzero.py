from absl import logging

import functools
import distrax
import jax
from pprint import pprint
import mctx
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
      samples_per_insert=1.0,
      batch_size=4,
      trace_length=6,
      discount=.99,
      simulation_steps=2,
      num_simulations=1,
      td_steps=3,
      burn_in_length=0,
      weight_decay=0.0,
      show_gradients=0,
      root_policy_coef=5.,
      v_target_source='reanalyze',
      # metrics='sparse',
      scale_grad=0.0,
      importance_sampling_exponent=0.6,
      priority_exponent=0.9,
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

  assert config.muzero_policy in ["muzero", "gumbel_muzero"]
  if config.muzero_policy == "muzero":
    muzero_policy = functools.partial(
        mctx.muzero_policy,
        dirichlet_fraction=config.dirichlet_fraction,
        dirichlet_alpha=config.dirichlet_alpha,
        pb_c_init=config.pb_c_init,
        pb_c_base=config.pb_c_base,
        temperature=config.temperature)
  elif config.muzero_policy == "gumbel_muzero":
    muzero_policy = functools.partial(
        mctx.gumbel_muzero_policy,
        gumbel_scale=config.gumbel_scale)

  assert config.policy_loss in ["cross_entropy", "kl_forward", "kl_back"]
  if config.policy_loss == 'cross_entropy':
    policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)
  elif config.policy_loss == 'kl_forward':
    def kl_forward(p, l):
      return distrax.Categorical(probs=p).kl_divergence(distrax.Categorical(logits=l))
    policy_loss_fn = jax.vmap(kl_forward)
  elif config.policy_loss == 'kl_back':
    def kl_back(p, l):
      return distrax.Categorical(logits=l).kl_divergence(distrax.Categorical(probs=p))
    policy_loss_fn = jax.vmap(kl_back)

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
    reanalyze_ratio=config.reanalyze_ratio,
    metrics=config.metrics,
  )

  builder = MuZeroBuilder(config, loss_fn=ve_loss_fn)

  network_factory = functools.partial(
      muzero_networks.make_babyai_networks,
      config=config,
      discretizer=discretizer)
  
  return config, builder, network_factory