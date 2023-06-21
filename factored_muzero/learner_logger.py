
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
    to_log = super().create_log_metrics(metrics)

    metrics = metrics.get('visualize_metrics', {})
    # get data from batch-idx = 0
    metrics = jax.tree_map(lambda x: x[0], metrics)
    root_data = metrics.get("visualize_root_data", {})

    if not root_data: return to_log

    ######################
    # plot image attention
    ######################
    online_outputs = root_data['online_outputs']

    # [T, num_slots, spatial_positions]
    slot_attn = online_outputs.state.rep.attn
    ntime, slots, spatial_positions = slot_attn.shape

    images = root_data['data'].observation.observation.image
    assert images.shape[1] == images.shape[2]
    width = int(np.sqrt(spatial_positions))
    spatial_attn = slot_attn.reshape(-1, slots, width, width)

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
    # plot attention in prediction layers
    ######################
    # [T, num_layers, num_heads, num_factors, num_factors]
    # num_factors = num_slots + 1 (for task?)
    pred_attn = online_outputs.pred_attn_outputs.attn

    pred_attn_images = []
    for t in range(ntime):
      img = attn_analysis.plot_perlayer_attn(
        attn=pred_attn[t],
        title=f"Timestep {t+1}",
        factor_labels=['Task'] + [f"Factor {f+1}" for f in range(slots)],
      )
      pred_attn_images.append(img)

    to_log['0.pred_attn_01'] = [
        wandb.Image(img) for img in pred_attn_images]

    ######################
    # plot slot entropy
    ######################
    attn_entropy = attn_analysis.slot_attn_entropy(
      slot_attn, normalize=True)
    to_log['1.slot_attn_entropy'] = wandb.Image(attn_entropy)

    return to_log
