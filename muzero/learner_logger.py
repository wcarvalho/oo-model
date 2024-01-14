
from typing import List, Optional
from absl import logging

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import wandb

from functools import partial


from acme import types as acme_types
from muzero import types as muzero_types
from analysis import utils as analyis_utils


State = acme_types.NestedArray

def compute_episode_return(rewards, gamma):
    episode_return = np.zeros_like(rewards)
    episode_return[-1] = rewards[-1]

    for t in range(len(rewards)-2, -1, -1):
        episode_return[t] = rewards[t] + gamma * episode_return[t+1]

    return episode_return


def apply_mask(x, mask, fn='product'):
  if fn == 'product':
    return x*mask
  elif fn == 'index':
    mask = mask > 0
    return x[mask]
  else:
    raise NotImplementedError(fn)

class BaseLogger:

  def __init__(self, label: str='LearnerLogger', log_frequency: int = 1000):
    self._log_frequency = log_frequency
    self._label = label
    self._idx = 0

  def create_metrics(
      self,
      data: acme_types.NestedArray,
      in_episode: acme_types.NestedArray,
      is_terminal_mask: acme_types.NestedArray,
      online_outputs: muzero_types.RootOutput,
      online_state: State,
  ):
    pass

  def log_metrics(self):
    pass


class LearnerLogger(BaseLogger):

  def __init__(self,
               discount: float,
               action_names: List[str] = None,
               invalid_actions: Optional[np.ndarray] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._action_names = action_names
    self._discount = discount
    self._mask_fn = partial(apply_mask, fn='index')
    self._invalid_actions = invalid_actions

  def create_log_metrics(self, metrics):

    metrics = metrics.get('visualize_metrics', {})
    # get data from batch-idx = 0
    metrics = jax.tree_map(lambda x: x[0], metrics)


    root_data = metrics.get("visualize_root_data", {})
    images = root_data['data'].observation.observation.image
    reward = root_data['data'].reward
    discounts = root_data['data'].discount
    actions = root_data['data'].action
    is_terminal = root_data['is_terminal']
    in_episode = root_data['in_episode']

    policy_probs = root_data['policy_root_prediction']
    policy_root_target = root_data['policy_root_target']
    policy_root_prediction = root_data['policy_root_prediction']

    mcts_values = root_data['mcts_values']
    value_probs = jax.nn.softmax(root_data['value_root_logits'])
    value_prediction = root_data['value_root_prediction']
    value_root_target = root_data['value_root_target']
    value_loss = root_data['value_root_ce']
    value_loss_mask = root_data['value_root_mask']

    policy_loss = root_data['policy_root_ce']
    policy_loss_mask = root_data['policy_root_mask']

    model_data_t0 = metrics.get("visualize_model_data_t0", {})
    # simulation_actions = model_data_t0['simulation_actions']
    policy_model_target = model_data_t0['policy_model_target']
    policy_model_prediction = model_data_t0['policy_model_prediction']
    value_model_mask = model_data_t0['value_model_mask']
    reward_model_mask = model_data_t0['reward_model_mask']
    reward_model_ce = model_data_t0['reward_model_ce']
    policy_model_ce = model_data_t0['policy_model_ce']
    value_model_ce = model_data_t0['value_model_ce']
    reward_model_mask = model_data_t0['reward_model_mask']
    policy_model_mask = model_data_t0['policy_model_mask']
    value_model_mask = model_data_t0['value_model_mask']

    to_log = dict()
    def wandb_images(images_):
      return [wandb.Image(img) for img in images_]

    ######################
    # plot episode images
    ######################
    plot_images = []
    r_tm1 = 'X'
    for idx in range(len(images)):
      a_t = self._action_names[int(actions[idx])]
      r_t = str(reward[idx])
      d_t =str(is_terminal[idx])

      # if in_episode[idx] == 0:
      #   break

      title = f't={idx}, r_tm1={r_tm1}, a_t={a_t}, r_t={r_t}, d_t={d_t}'
      r_tm1 = r_t

      plot_images.append(analyis_utils.plot_image(images[idx], title=title))

    to_log['0.0.images'] = wandb_images(plot_images)

    ######################
    # plot value/policy entropy 
    ######################
    value_entropy = analyis_utils.compute_entropy(
      prob=self._mask_fn(value_probs, in_episode[:-1]),
    )
    prob_entropy = analyis_utils.compute_entropy(
      prob=self._mask_fn(policy_probs, in_episode),
      mask=self._invalid_actions,
    )
    entropy = analyis_utils.plot_entropy(
        data=[value_entropy,
              prob_entropy,
              ],
        labels=["Value",
                "Policy"])
    to_log['0.1.entropy'] = wandb.Image(entropy)


    ######################
    # Value Prediction loss
    ######################
    # root
    episode_return = compute_episode_return(reward, gamma=self._discount)
    value_plot = analyis_utils.plot_line(
        ys=[
          self._mask_fn(value_prediction, value_loss_mask),
          self._mask_fn(mcts_values, value_loss_mask),
          self._mask_fn(episode_return[:-1], value_loss_mask),
          self._mask_fn(value_root_target, value_loss_mask),
        ],
        labels=[
          "Value Prediction",
          "MCTS Values",
          "Episode Return",
          "Value Target",
        ],
        title='Value Predictions',
    )
    to_log['1.value/predictions/0.root'] = wandb.Image(
        value_plot)

    # Model
    labels = ['value_model_prediction', 'value_model_target']
    plot = analyis_utils.plot_line(
        ys=[self._mask_fn(model_data_t0[l], value_model_mask)
              for l in labels],
        labels=labels,
        x_0=1,
        title='Value Model Predictions',
        xlabel='Simulation steps',
    )
    to_log['1.value/predictions/model'] = wandb.Image(plot)

    ######################
    # Reward Prediction loss
    ######################
    labels = ['reward_model_prediction', 'reward_model_target']
    plot = analyis_utils.plot_line(
        ys=[self._mask_fn(model_data_t0[l], reward_model_mask)
              for l in labels],
        labels=labels,
        x_0=1,
        title='Reward Model Predictions',
        xlabel='Simulation steps',
    )
    to_log['2.reward/predictions/model'] = wandb.Image(plot)

    ######################
    # Policy Predictions (root vs model)
    ######################
    # root

    num_actions = policy_probs.shape[-1]
    onehot_env_actions = jax.nn.one_hot(actions, num_classes=num_actions)

    plot_images = []
    for idx in range(policy_root_target.shape[0]):
      if in_episode[idx] == 1:
        plot_images.append(analyis_utils.plot_compare_pmfs(
          xlabels=self._action_names,
          pmfs=[policy_root_target[idx],
                policy_root_prediction[idx],
                onehot_env_actions[idx],
                ],
          pmf_labels=['Target', 'Prediction', 'Environment Action'],
          title=f'Policy Root Predictions (T={idx})',
          ))
    to_log['3.policy/predictions/1.root'] = wandb_images(plot_images)

    # model
    # compare actions by model w/ actions in environment
    # as reminder: simulation simulated future states using
    # s_0, a_0, a_1, ....
    # environment_actions = jax.nn.one_hot(
    #   simulation_actions[0, 1:], num_classes=num_actions)
    plot_images = []
    sim_steps = len(policy_model_target)
    for idx in range(sim_steps):
        if idx + 1 < len(in_episode):
          sim_in_episode = in_episode[idx+1] == 1
        else:
          sim_in_episode = False
        if sim_in_episode:
          plot_images.append(analyis_utils.plot_compare_pmfs(
              xlabels=self._action_names,
              pmfs=[
                policy_model_target[idx],
                policy_model_prediction[idx],
                onehot_env_actions[idx+1],
                ],
              pmf_labels=[
                'Target',
                'Prediction',
                'Environment Action',
              ],
              # starts at at pred for T=1
              title=f'Policy Model Predictions (T={idx+1})',
          ))
    to_log['3.policy/predictions/model'] = wandb_images(plot_images)


    ######################
    # Cross-entropy (Root, Simulation, Episode)
    ######################
    # root node

    plot = analyis_utils.plot_line(
        ys=[self._mask_fn(value_loss, value_loss_mask),
            self._mask_fn(policy_loss, policy_loss_mask)],
        labels=["Value", "Policy"],
        title='Cross Entropy Losses (Root)',
    )
    to_log['4.loss_cross_entropy/0.root'] = wandb.Image(plot)

    # In simulation
    plot = analyis_utils.plot_line(
        ys=[self._mask_fn(value_model_ce, value_model_mask),
            self._mask_fn(policy_model_ce, policy_model_mask),
            self._mask_fn(reward_model_ce, reward_model_mask)
            ],
        labels=["Value",
                "Policy",
                "Reward"
                ],
        x_0=1,
        title='Model Cross Entropy Losses (Simulation, T=0)',
        xlabel='Simulation steps',
    )
    to_log['4.loss_cross_entropy/model_simulation'] = wandb.Image(plot)

    # Across episode
    model_data_t_all = metrics.get("visualize_model_data_t_all", {})
    reward_model_ce = model_data_t_all['reward_model_ce']
    policy_model_ce = model_data_t_all['policy_model_ce']
    value_model_ce = model_data_t_all['value_model_ce']
    reward_model_mask = model_data_t_all['reward_model_mask']
    policy_model_mask = model_data_t_all['policy_model_mask']
    value_model_mask = model_data_t_all['value_model_mask']
    plot = analyis_utils.plot_line(
        ys=[self._mask_fn(value_model_ce, value_model_mask),
            self._mask_fn(policy_model_ce, policy_model_mask),
            self._mask_fn(reward_model_ce, reward_model_mask)
            ],
        labels=["Value",
                "Policy",
                "Reward",
                ],
        title='Model Cross Entropy Losses (Avg starting @T)',
        xlabel='Timesteps',
    )
    to_log['4.loss_cross_entropy/model_average'] = wandb.Image(plot)

    ######################
    # plot epsiode stats
    ######################
    labels = ['reward', 'discounts', 'in_episode']
    ys = [reward, discounts, in_episode]
    plot_img = analyis_utils.plot_line(
        ys=ys,
        xlabel='Time',
        labels=labels)
    to_log['z.episode_stats'] = wandb.Image(plot_img)

    plot_img = analyis_utils.plot_line(
        ys=[model_data_t_all['reward_model_mask'],
            model_data_t_all['policy_model_mask'],
            model_data_t_all['value_model_mask']],
        xlabel='Time',
        labels=['Reward', "Policy", "Value"],
        title="Masks over Episode"
        )
    to_log['z.episode_masks'] = wandb.Image(plot_img)

    plot_img = analyis_utils.plot_line(
        ys=[model_data_t0['reward_model_mask'],
            model_data_t0['policy_model_mask'],
            model_data_t0['value_model_mask']],
        xlabel='Time',
        labels=['Reward', "Policy", "Value"],
        title="Masks in Simulation"
        )
    to_log['z.simulation_masks'] = wandb.Image(plot_img)

    return to_log

  def step(self):
    self._idx += 1

  def log_metrics(self, metrics, label: str = None, config = None):
    if not (self._idx % self._log_frequency == 0): return

    label = label or self._label
    logging.info(f'creating {label} data. idx {self._idx}')
    to_log = self.create_log_metrics(metrics, config=config)

    to_log = {f'{label}/{k}': v for k, v in to_log.items()}
    if not to_log: return

    try:
      wandb.log(to_log)
      logging.info(f'logged {label} data. idx {self._idx}')
    except Exception as e:
      logging.warning(e)
      logging.warning(f"{label}: turning off logging.")
      self._log_frequency = np.inf
