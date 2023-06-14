
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
    self._log_data = False

  def set_logging(self, log_data: bool = False):
    self._log_data = log_data

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
    if not self._log_data:
      return
    del action
    observation = jax_utils.to_numpy(observation.observation)
    image = observation.image
    
    state = jax_utils.to_numpy(state.recurrent_state)
    attn = state.attn

    slots = attn.shape[0]
    width = image.shape[1] // self.tile_size
    img_attn = attn.reshape(slots, width, width)

    self.data['attn'].append(attn)
    img_attn_01 = attn_analysis.timestep_img_attn(
      image=image, img_attn=img_attn,
      shared_min_max='timestep',
      time_with_x=False,
      im_only=True,
      vmin_pre=0,
      vmax_pre=1.0,
      base_width=1)
    img_attn_reg = attn_analysis.timestep_img_attn(
      image=image, img_attn=img_attn,
      shared_min_max='timestep',
      time_with_x=False,
      im_only=True,
      vmin_pre=None,
      vmax_pre=None,
      base_width=1)
      
    self.data['img_attn_01'].append(img_attn_01)
    self.data['img_attn_reg'].append(img_attn_reg)

  def log_data(self, step: int):
    if self.has_data:
      attn = np.asarray(self.data['attn'])
      attn_entropy = attn_analysis.slot_attn_entropy(attn, normalize=True)
      max_attn = attn_analysis.slot_attn_max_likelihood(attn)

      wandb.log({
          "actor_images/img_attn_01": [wandb.Image(img) for img in self.data['img_attn_01']],
          "actor_images/img_attn_reg": [wandb.Image(img) for img in self.data['img_attn_reg']],
          "actor_images/attn_entropy": wandb.Image(attn_entropy),
          "actor_images/max_attn": wandb.Image(max_attn),
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

    if self.logger.has_data:
      if self.verbosity:
        logging.info(f'logging actor data. idx {self.idx}')
      self.logger.log_data(self.idx)

    log_episode = self.idx % self.log_frequency == 0
    self.logger.reset()
    self.logger.set_logging(log_episode)  # if logging is expensive, only log sometimes
    self.idx += 1


  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    action = super().select_action(observation)
    self.logger.collect_data(
        state=self._state,
        action=action,
        observation=observation)
    return action
