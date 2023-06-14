
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

  def create_log_metrics(self, metrics):
    # NOTE: RNN hidden is available, not RNN state...
    return super().create_log_metrics(metrics)

    metrics = metrics.get('visualize_metrics', {})
    # get data from batch-idx = 0
    metrics = jax.tree_map(lambda x: x[0], metrics)
    root_data = metrics.get("visualize_root_data", {})
    import ipdb; ipdb.set_trace()
    if not root_data: return to_log

    ######################
    # plot image attention
    ######################
    online_state = root_data['online_state']
    import ipdb; ipdb.set_trace()
    attn = jax.tree_map(lambda x: x[:, 0], online_state.attn)  # [T, num_slots, spatial_positions]
    ntime, slots, spatial_positions = attn.shape[0]
    images = root_data['images']  # [T, H, W, C]

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
    to_log['0.img_attn_01_normalized'] = [wandb.Image(img) for img in img_attn_01]
    to_log['0.img_attn_unnormalized'] = [wandb.Image(img) for img in img_attn_reg]

    ######################
    # plot slot entropy
    ######################
    attn_entropy = attn_analysis.slot_attn_entropy(attn, normalize=True)
    to_log['0.attn_entropy'] = attn_entropy

    return to_log
