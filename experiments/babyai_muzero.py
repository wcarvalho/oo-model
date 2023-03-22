import functools
import distrax
import jax
import mctx
import rlax

from muzero import utils as muzero_utils
from muzero.builder_broken import MuZeroBuilder
from muzero import networks as muzero_networks
from muzero.config import MuZeroConfig

from experiments.utils import update_config

def setup(
    launch: bool=True,
    config_kwargs: dict = None):
  config_kwargs = config_kwargs or dict()
  if not launch: #DEBUG
    config_kwargs['min_replay_size'] = 100
    config_kwargs["samples_per_insert"] = 1.0
    config_kwargs['batch_size'] = 4
    config_kwargs['trace_length'] = 6
    config_kwargs['discount'] = .99
    config_kwargs['simulation_steps'] = 2
    config_kwargs['num_simulations'] = 1
    config_kwargs['td_steps'] = 3
    config_kwargs['burn_in_length'] = 0
    config_kwargs['weight_decay'] = 0.0
    config_kwargs['show_gradients'] = 1
    config_kwargs['metrics'] = 'sparse'
    config_kwargs['scale_grad'] = 0.0
    config_kwargs['network_fn'] = 'babyai'
    config_kwargs['builder'] = 'old'
    config_kwargs['loss_fn'] = 'new'
    config_kwargs['num_sgd_steps_per_step'] = 4


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


  if config.network_fn == 'babyai':
    network_fn = muzero_networks.make_babyai_networks
  # elif config.network_fn == "old_babyai":
  #   from muzero_old import networks as muzero_networks_old
  #   network_fn = muzero_networks_old.make_babyai_networks
  elif config.network_fn == "simple_babyai":
    raise NotImplementedError
    # network_fn = muzero_networks.make_simple_babyai_networks
  else:
    raise NotImplementedError(config.network_fn)

  if config.loss_fn == 'new':
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

    from muzero.ve_losses import ValueEquivalentLoss
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
      value_coef=config.value_coef,
      reward_coef=config.reward_coef,
      v_target_source=config.v_target_source,
      metrics=config.metrics,
    )
  elif config.loss_fn == 'old':
    assert config.muzero_policy in ["muzero", "gumbel_muzero"]
    if config.muzero_policy == "muzero":
      muzero_policy = functools.partial(
          mctx.muzero_policy,
          dirichlet_fraction=config.dirichlet_fraction,
          dirichlet_alpha=config.dirichlet_alpha,
          pb_c_init=config.pb_c_init,
          pb_c_base=config.pb_c_base,
          temperature=config.temperature)
      policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)
    elif config.muzero_policy == "gumbel_muzero":
      muzero_policy = functools.partial(
          mctx.gumbel_muzero_policy,
          gumbel_scale=config.gumbel_scale)
      policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)
    from muzero_old.ve_losses import ValueEquivalentLoss
    ve_loss_fn = functools.partial(ValueEquivalentLoss,
      muzero_policy=muzero_policy,
      policy_loss_fn=policy_loss_fn,
      discretizer=discretizer,
      simulation_steps=config.simulation_steps,
      num_simulations=config.num_simulations,
      discount=config.discount,
      td_steps=config.td_steps,
      model_coef=config.model_coef,
      policy_coef=config.policy_coef,
      value_coef=config.value_coef,
      reward_coef=config.reward_coef,
      v_target_source=config.v_target_source,
      metrics=config.metrics,
      model_share_params=True,
    )

  if config.builder == 'new':
    builder = MuZeroBuilder(config, discretizer=discretizer, loss_fn=ve_loss_fn)
  elif config.builder == 'old':
    from muzero.builder import MuZeroBuilder as MuZeroBuilderOld
    builder = MuZeroBuilderOld(config, discretizer=discretizer, loss_fn=ve_loss_fn)

  network_factory = functools.partial(
          network_fn,
          config=copy.deepcopy(config),
          discretizer=discretizer)
  
  return config, builder, network_factory