
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


State = acme_types.NestedArray

def compute_episode_return(rewards, gamma):
    episode_return = np.zeros_like(rewards)
    episode_return[-1] = rewards[-1]

    for t in range(len(rewards)-2, -1, -1):
        episode_return[t] = rewards[t] + gamma * episode_return[t+1]

    return episode_return


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
               image_columns: int = 5,
               action_names: List[str] = None,
               **kwargs):
    super().__init__(**kwargs)
    self.image_columns = image_columns
    self._action_names = action_names
    self._discount = discount

  def create_log_metrics(self, metrics):

    metrics = metrics.get('visualize_metrics', {})
    # get data from batch-idx = 0
    metrics = jax.tree_map(lambda x: x[0], metrics)

    root_data = metrics.get("visualize_root_data", {})
    model_data_t0 = metrics.get("visualize_model_data_t0", {})
    model_data_t_all = metrics.get("visualize_model_data_t_all", {})

    reward = root_data['data'].reward
    discounts = root_data['data'].discount
    actions = root_data['data'].action
    is_terminal = root_data['is_terminal']
    in_episode = root_data['in_episode']

    to_log = dict()
    ######################
    # plot episode images
    ######################
    images = root_data['data'].observation.observation.image
    to_log['0.0.images'] = [wandb.Image(img)
                                      for img in images]

    ######################
    # plot value/policy entropy 
    ######################
    value_probs = jax.nn.softmax(root_data['value_root_logits'])
    policy_probs = root_data['policy_root_prediction']
    entropy = analyis_utils.plot_entropy(
        data=[value_probs, policy_probs],
        labels=["Value",
                "Policy"])
    to_log['0.1.entropy'] = wandb.Image(entropy)

    ######################
    # Policy Predictions (root vs model)
    ######################
    # root
    policy_root_target = root_data['policy_root_target']
    policy_root_prediction = root_data['policy_root_prediction']

    num_actions = policy_probs.shape[-1]
    onehot_actions = jax.nn.one_hot(actions, num_classes=num_actions)

    to_log['1.policy/predictions/1.root'] = [
        wandb.Image(analyis_utils.plot_compare_pmfs(
          xlabels=self._action_names,
          pmfs=[policy_root_target[idx],
                policy_root_prediction[idx],
                onehot_actions[idx],
                ],
          pmf_labels=['Target', 'Prediction', 'Environment Action'],
          title=f'Policy Root Predictions (T={idx+1})',
          ))
      for idx in range(policy_root_target.shape[0])
    ]

    # model
    simulation_actions = model_data_t0['simulation_actions']
    onehot_simulation_actions = jax.nn.one_hot(simulation_actions[0], num_classes=num_actions)
    policy_model_target = model_data_t0['policy_model_target']
    policy_model_prediction = model_data_t0['policy_model_prediction']
    to_log['1.policy/predictions/model'] = [
        wandb.Image(analyis_utils.plot_compare_pmfs(
            xlabels=self._action_names,
            pmfs=[
              policy_model_target[idx],
              policy_model_prediction[idx],
              onehot_simulation_actions[idx],
              ],
            pmf_labels=['Target',
                    'Prediction',
                    'Environment Action',
                    ],
            # starts at at pred for T=2
            title=f'Policy Model Predictions (T={idx+2})',
        ))
        for idx in range(policy_model_target.shape[0])
    ]


    ######################
    # Value Predictions
    ######################
    # root
    value_loss_mask = root_data['value_root_mask']
    mcts_values = root_data['mcts_values']
    episode_return = compute_episode_return(reward, gamma=self._discount)
    value_prediction = root_data['value_root_prediction']
    value_root_target = root_data['value_root_target']
    value_plot = analyis_utils.plot_line(
        ys=[
          value_prediction,
          mcts_values,
          episode_return[:-1],
          value_root_target,
        ],
        labels=[
          "Value Prediction",
          "MCTS Values",
          "Episode Return",
          "Value Target",
        ],
        title='Value Predictions',
    )
    to_log['2.value/predictions/0.root'] = wandb.Image(
        value_plot)

    # Model
    labels = ['value_model_prediction', 'value_model_target']
    plot = analyis_utils.plot_line(
        ys=[model_data_t0[l] for l in labels],
        labels=labels,
        title='Value Model Predictions',
        xlabel='Simulation steps',
    )
    to_log['2.value/predictions/model'] = wandb.Image(plot)

    ######################
    # Reward Predictions
    ######################
    labels = ['reward_model_prediction', 'reward_model_target']
    plot = analyis_utils.plot_line(
        ys=[model_data_t0[l] for l in labels],
        labels=labels,
        title='Reward Model Predictions',
        xlabel='Simulation steps',
    )
    to_log['3.reward/predictions/model'] = wandb.Image(plot)

    ######################
    # Cross-entropy (Root, Simulation, Episode)
    ######################
    # root node
    value_loss = root_data['value_root_ce']
    value_loss_mask = root_data['value_root_mask']
    policy_loss = root_data['policy_root_ce']
    policy_loss_mask = root_data['policy_root_mask']
    # plot = analyis_utils.plot_line(
    #     ys=[value_loss, value_loss*value_loss_mask,
    #         policy_loss, policy_loss*policy_loss_mask],
    #     labels=["Value", "Value (Masked)", "Policy", "Policy (Masked)"],
    #     title='Cross Entropy Losses (Root)',
    # )
    plot = analyis_utils.plot_line(
        ys=[value_loss*value_loss_mask,
            policy_loss*policy_loss_mask],
        labels=["Value", "Policy"],
        title='Cross Entropy Losses (Root)',
    )
    to_log['4.loss_cross_entropy/0.root'] = wandb.Image(plot)

    # In simulation
    reward_model_ce = model_data_t0['reward_model_ce']
    policy_model_ce = model_data_t0['policy_model_ce']
    value_model_ce = model_data_t0['value_model_ce']
    reward_model_mask = model_data_t0['reward_model_mask']
    policy_model_mask = model_data_t0['policy_model_mask']
    value_model_mask = model_data_t0['value_model_mask']
    plot = analyis_utils.plot_line(
        ys=[value_model_ce*value_model_mask,
            policy_model_ce*policy_model_mask,
            reward_model_ce*reward_model_mask
            ],
        labels=["Value",
                "Policy",
                "Reward"
                ],
        title='Model Cross Entropy Losses (Simulation, T=0)',
        xlabel='Simulation steps',
    )
    to_log['4.loss_cross_entropy/model_simulation'] = wandb.Image(plot)

    # Across episode
    reward_model_ce = model_data_t_all['reward_model_ce']
    policy_model_ce = model_data_t_all['policy_model_ce']
    value_model_ce = model_data_t_all['value_model_ce']
    reward_model_mask = model_data_t_all['reward_model_mask']
    policy_model_mask = model_data_t_all['policy_model_mask']
    value_model_mask = model_data_t_all['value_model_mask']
    plot = analyis_utils.plot_line(
        ys=[value_model_ce, value_model_ce*value_model_mask,
            policy_model_ce, policy_model_ce*policy_model_mask,
            reward_model_ce, reward_model_ce*reward_model_mask
            ],
        labels=["Value", "Value (Masked)",
                "Policy", "Policy (Masked)",
                "Reward", "Reward (Masked)",
                ],
        title='Model Cross Entropy Losses (Avg Across Episode)',
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

    to_log = {f'{self._label}/{k}': v for k, v in to_log.items()}

    return to_log

  def log_metrics(self, metrics):
    self._idx += 1
    if not (self._idx % self._log_frequency == 0): return

    logging.info(f'creating {self._label} data. idx {self._idx}')
    to_log = self.create_log_metrics(metrics)

    if not to_log: return

    try:
      wandb.log(to_log)
      logging.info(f'logged {self._label} data. idx {self._idx}')
    except Exception as e:
      logging.warning(e)
      logging.warning(f"{self._label}: turning off logging.")
      self._log_frequency = np.inf
