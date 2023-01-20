
from typing import NamedTuple

import functools
import gym_minigrid.window

from envs.babyai_kitchen.bot import KitchenBot
from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
from envs.babyai_kitchen import babyai_utils


import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

# from acme import types
import yaml
import pickle


class Episode(NamedTuple):
  observations: np.array
  actions: np.array
  rewards: np.array
  discounts: np.array
  task: str


def save_data(data):
  NotImplementedError


def main():
  """
  Notes:
  what tasks are you going to choose?
    1. goto x
    2. place x in y
  
  """

  import argparse
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--tasks',
    help='tasks file', 
    default='envs/babyai_kitchen/tasks/place.yaml')
  parser.add_argument('--sets',
    help='sets file',
    default="envs/babyai_kitchen/tasks/default_sets.yaml")
  parser.add_argument('--room-size', type=int, default=10)
  parser.add_argument('--agent-view-size', type=int, default=9)
  parser.add_argument('--episodes', type=int, default=int(1e6))
  parser.add_argument('--tile-size', type=int, default=8)
  parser.add_argument('--train', type=int, default=0)
  parser.add_argument('--partial', type=int, default=0)
  parser.add_argument('--visualize', type=int, default=0)
  parser.add_argument('--seed', type=int, default=9)
  parser.add_argument('--testing', type=int, default=0)
  args = parser.parse_args()



  # ======================================================
  # create environment
  # ======================================================
  # -----------------------
  # load file w/ sets of objects
  # -----------------------
  with open(args.sets, 'r') as f:
    sets = yaml.load(f, Loader=yaml.SafeLoader)

  # -----------------------
  # load file w/ tasks 
  # -----------------------
  with open(args.tasks, 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)

  level_kwargs=dict(
      room_size=args.room_size,
      agent_view_size=args.agent_view_size,
      verbosity=0,
      tile_size=args.tile_size,
      use_time_limit=False,
      )


  data = None

  key = 'train'
  level_kwargs = babyai_utils.constuct_kitchenmultilevel_kwargs(
      task_dicts=tasks[key],
      level_kwargs=level_kwargs,
      sets=sets)

  if args.partial:
    wrappers = [functools.partial(RGBImgPartialObsWrapper,
      tile_size=args.tile_size)]
  else:
    wrappers = [functools.partial(RGBImgFullyObsWrapper,
      tile_size=args.tile_size)]

  env = MultiLevel(level_kwargs, wrappers=wrappers)

  # ======================================================
  # Sample episodes
  # ======================================================
  episodes = []
  nepisodes = 100 if args.testing else int(1e6)
  for episode_idx in tqdm(range(nepisodes)):
      # env.seed(0)
      obs = env.reset()

      if args.visualize:
        fig, ax = plt.subplots()
        im = ax.imshow(obs['image'])

        def update(obs):
          im.set_data(obs['image'])
          plt.pause(0.2)

      bot = KitchenBot(env)
      obss, actions, rewards, dones = bot.generate_traj(
        plot_fn=update if args.visualize else lambda x:x)

      observations = np.array([obs['image']] + [i['image'] for i in obss])
      actions = np.array(actions + [0])
      rewards = np.array([0] + rewards)
      discounts = (np.array([False] + dones)==False).astype(np.float32)
      discounts[-1] = 0.0
      task = obs['mission']

      episode = Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        discounts=discounts,
        task=task)
      episodes.append(episode)


  tasks_file = os.path.basename(args.tasks)
  if args.testing:
    filename=f'data/{tasks_file}.debug.pkl'
  else:
    filename=f'data/{tasks_file}.pkl'
  with open(filename, 'wb') as handle:
    pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved', filename)



if __name__ == '__main__':
  main()