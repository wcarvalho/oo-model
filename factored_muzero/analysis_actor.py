
from absl import logging
import collections
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme import types
from acme.jax import utils as jax_utils

import numpy as np
import dm_env
import wandb
import matplotlib.pyplot as plt

from experiments import attn_analysis

class BaseLogger:

  def __init__(self):
    self.data = None

  def reset(self):
    self.data = collections.defaultdict(list)

  @property
  def has_data(self):
    return self.data is not None and len(self.data)

  def collect_data(
      self,
      state: network_lib.RecurrentState,
      observation: network_lib.Observation,
      action: int = None,
      ):
    pass

  def log_data(self):
    pass

class AttnLogger(BaseLogger):

  def __init__(self, tile_size: int = 8):
    super().__init__()
    self.tile_size = tile_size

  def collect_data(
      self,
      state: network_lib.RecurrentState,
      observation: network_lib.Observation,
      action: int = None,
  ):
    del action
    observation = jax_utils.to_numpy(observation.observation)
    image = observation.image
    
    state = jax_utils.to_numpy(state.recurrent_state)
    attn = state.attn

    slots = attn.shape[0]
    width = image.shape[1] // self.tile_size
    img_attn = attn.reshape(slots, width, width)

    self.data['attn'].append(attn)
    img_attn = attn_analysis.timestep_img_attn(
      image=image, img_attn=img_attn,
      shared_min_max='timestep',
      time_with_x=False,
      im_only=True,
      vmin_pre=0,
      vmax_pre=1.0,
      base_width=1)
    self.data['img_attn'].append(img_attn)

  def log_data(self, step: int):
    if self.has_data:
      attn = np.asarray(self.data['attn'])
      attn_entropy = attn_analysis.slot_attn_entropy(attn, normalize=True)
      max_attn = attn_analysis.slot_attn_max_likelihood(attn)

      wandb.log({
          "images/img_attns": [wandb.Image(img) for img in self.data['img_attn']],
          "images/attn_entropy": wandb.Image(attn_entropy),
          "images/max_attn": wandb.Image(max_attn),
          "images/step": step,
        })

class VisualizeActor(actors.GenericActor):

  def __init__(self,
               *args,
               logger: BaseLogger = AttnLogger(),
               log_frequency: int = 1000,  # every 1000 episodes
               verbosity: int=1,
               **kwargs):
    super().__init__(*args, **kwargs)
    logging.info('Initializing actor visualizer.')
    self.logger = logger
    self.idx = 0
    self.log_frequency = log_frequency
    self.verbosity = verbosity

  def observe_first(self, timestep: dm_env.TimeStep):
    super().observe_first(timestep)

    if self.logger.has_data and self.idx % self.log_frequency == 0:
      if self.verbosity:
        logging.info('logging actor data')
      self.logger.log_data(self.idx)
    self.logger.reset()
    self.idx += 1


  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    action = super().select_action(observation)
    self.logger.collect_data(
        state=self._state,
        action=action,
        observation=observation)
    return action
