
from typing import List
from absl import logging

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import wandb



from acme import types as acme_types
from muzero import types as muzero_types
from analysis import utils as analyis_utils
from experiments import attn_analysis
from muzero import learner_logger

State = acme_types.NestedArray


def softmax(x):
    # Subtracting the maximum value for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LearnerLogger(learner_logger.LearnerLogger):

  def __init__(self, image_columns: int = 5, **kwargs):
    super().__init__(**kwargs)
    self.image_columns = image_columns

  def create_data_metrics(
      self,
      data: acme_types.NestedArray,
      in_episode: acme_types.NestedArray,
      is_terminal_mask: acme_types.NestedArray,
      online_outputs: muzero_types.RootOutput,
      online_state: State,
  ):
    metrics = super().create_metrics(
      data=data,
      in_episode=in_episode,
      is_terminal_mask=is_terminal_mask,
      online_outputs=online_outputs,
      online_state=online_state,
    )
    metrics['data_metrics']['attn'] = jax.tree_map(lambda x: x[:, 0], online_state.attn)

    return metrics

  def create_log_metrics(self, metrics):
    to_log = super().create_log_metrics(metrics)
    data = metrics.get("data_metrics", None)

    if not data: return to_log

    ######################
    # plot image attention
    ######################
    attn = data['attn']      # [T, num_slots, spatial_positions]
    ntime, slots, spatial_positions = attn.shape[0]
    images = data['images']  # [T, H, W, C]

    assert images.shape[1] == images.shape[2]
    width = np.sqrt(spatial_positions)
    spatial_attn = attn.reshape(-1, slots, width, width)

    img_attn_01 = []
    img_attn_reg = []
    for idx in range(ntime):
      img_attn_01.append(attn_analysis.timestep_img_attn(
          image=images[idx], img_attn=spatial_attn[idx],
          shared_min_max='timestep',
          time_with_x=False,
          im_only=True,
          vmin_pre=0,
          vmax_pre=1.0,
          base_width=1))
      img_attn_reg.append(attn_analysis.timestep_img_attn(
          image=images[idx], img_attn=spatial_attn[idx],
          shared_min_max='timestep',
          time_with_x=False,
          im_only=True,
          vmin_pre=None,
          vmax_pre=None,
          base_width=1))
    to_log['img_attn_01_normalized'] = [wandb.Image(img) for img in img_attn_01]
    to_log['img_attn_unnormalized'] = [wandb.Image(img) for img in img_attn_reg]

    ######################
    # plot slot entropy
    ######################
    attn_entropy = attn_analysis.slot_attn_entropy(attn, normalize=True)
    to_log['attn_entropy'] = attn_entropy

    return to_log
