from absl import logging

import copy
import functools
import jax
import mctx
from pprint import pprint
import rlax

from muzero import utils as muzero_utils
from muzero.types import TaskAwareRep
from muzero.builder import MuZeroBuilder

from factored_muzero import networks
from factored_muzero import types
from factored_muzero import attention
from factored_muzero.ve_losses import ValueEquivalentLoss
from factored_muzero.config import FactoredMuZeroConfig

from experiments.config_utils import update_config

def load_config(
    config_kwargs: dict = None,
    config_class: FactoredMuZeroConfig=None,
    strict_config: bool = True):
  config_kwargs = config_kwargs or dict()
  config_class = config_class or FactoredMuZeroConfig
  logging.info(f'Config arguments')
  pprint(config_kwargs)
  config = config_class()
  update_config(config, strict=strict_config, **config_kwargs)

  if config.sequence_period is None:
    config.sequence_period = config.trace_length

  return config


def get_state_remove_attention(outputs: types.RootOutput):
  """Remove attention weights before returning state in outputs.
  
  MCTS does not accept attention weights for some weird shaping issue.
  """
  state: types.TaskAwareSaviState = outputs.state
  savi_state: attention.SaviState = state.rep
  return state._replace(
    rep=attention.TransformerOutput(
      factors=savi_state.factors))


def setup(
    config: FactoredMuZeroConfig,
    network_kwargs: dict = None,
    loss_kwargs: dict = None,
    builder_kwargs: dict = None,
    invalid_actions = None,
    agent_name: str = 'factored',
    **kwargs):
  network_kwargs = network_kwargs or dict()
  loss_kwargs = loss_kwargs or dict()
  builder_kwargs = builder_kwargs or dict()

  discretizer = muzero_utils.Discretizer(
      num_bins=config.num_bins,
      step_size=config.scalar_step_size,
      max_value=config.max_scalar_value,
      tx_pair=config.tx_pair,
      clip_probs=config.clip_probs,
  )
  config.num_bins = discretizer._num_bins

  muzero_policy = functools.partial(
      mctx.gumbel_muzero_policy,
      max_depth=config.max_sim_depth,
      gumbel_scale=config.gumbel_scale)

  if config.policy_loss == "cross_entropy":
    policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)
  elif config.policy_loss == 'reverse_kl':
      import distrax
      def reverse_kl(prior_probs, online_logits):
        pmf = distrax.Categorical(logits=online_logits)
        return pmf.kl_divergence(distrax.Categorical(probs=prior_probs))
      policy_loss_fn = jax.vmap(reverse_kl)

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
    state_loss=config.state_model_loss,
    state_model_coef=config.state_model_coef,
    extra_contrast=config.extra_contrast,
    contrast_gamma=config.contrast_gamma,
    contrast_temp=config.contrast_temp,
    mask_model=config.mask_model,
    attention_penalty=config.attention_penalty,
    invalid_actions=invalid_actions,
    get_state=get_state_remove_attention,
    **loss_kwargs,
    )

  network_factory = functools.partial(
      networks.make_babyai_networks,
      config=config,
      invalid_actions=invalid_actions,
      agent_name=agent_name,
      **network_kwargs)
  

  builder = MuZeroBuilder(
      config=config,
      loss_fn=ve_loss_fn,
      network_factory=network_factory,
      **builder_kwargs)

  
  return builder, network_factory