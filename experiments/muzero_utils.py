import functools

from acme import specs
from acme.agents.jax import r2d2
from acme.jax import networks as networks_lib

from muzero import utils as muzero_utils
from muzero.builder import MuZeroBuilder
from muzero import networks as muzero_networks
from muzero.config import MuZeroConfig

from modules import vision
from modules import language
from modules import vision_language

def make_muzero_builder(
    launch: bool=True,
    config_kwargs: dict = None):
  if not launch: #DEBUG
    config_kwargs['min_replay_size'] = 100
    config_kwargs["samples_per_insert"] = 1.0
    config_kwargs['batch_size'] = 2
    config_kwargs['trace_length'] = 6
    config_kwargs['discount'] = .99
    config_kwargs['simulation_steps'] = 2
    config_kwargs['num_simulations'] = 1
    config_kwargs['td_steps'] = 3
    config_kwargs['burn_in_length'] = 0
    config_kwargs['show_gradients'] = 0
    config_kwargs['metrics'] = 'sparse'
    config_kwargs['model_combine_state_task'] = 'add_head_bias'

  config = MuZeroConfig()

  discretizer = muzero_utils.Discretizer(
      num_bins=config.num_bins,
      step_size=config.scalar_step_size,
      max_value=config.max_scalar_value,
      tx_pair=config.tx_pair,
  )
  config.num_bins = discretizer._num_bins

  if config.network_fn == 'babyai':
    network_fn = muzero_networks.make_babyai_networks
  elif config.network_fn == "simple_babyai":
    network_fn = muzero_networks.make_simple_babyai_networks
  else:
    raise NotImplementedError(config.network_fn)

  builder = MuZeroBuilder(config, discretizer=discretizer)

  network_factory = functools.partial(
          network_fn, config=config, discretizer=discretizer)
  
  return config, builder, network_factory