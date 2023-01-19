import gym_minigrid.window

from envs.babyai_kitchen.bot import GotoAvoidBot

from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
import tqdm
import numpy as np

from acme import types


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
  parser.add_argument('--tasks', help='tasks file', 
    default='envs/babyai_kitchen/tasks/unseen_arg/length=2_no_dist.yaml')
  parser.add_argument('--sets', help='sets file',
      default="envs/babyai_kitchen/tasks/default_sets.yaml")
  parser.add_argument('--room-size', type=int, default=10)
  parser.add_argument('--agent-view-size', type=int, default=9)
  parser.add_argument('--episodes', type=int, default=int(1e6))
  parser.add_argument('--tile-size', type=int, default=8)
  parser.add_argument('--train', type=int, default=0)
  parser.add_argument('--visualize', type=int, default=0)
  parser.add_argument('--seed', type=int, default=9)
  args = parser.parse_args()

  if args.visualize:
    import cv2
    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)

    def combine(full, partial):
        full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
        return np.concatenate((full_small, partial), axis=1)

    def plot_fn(obs):
      full = env.render('rgb_array', tile_size=args.tile_size, highlight=True)
      window.set_caption(obs['mission'])
      window.show_img(combine(full, obs['image']))

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
      random_object_state=args.random_object_state,
      verbosity=args.verbosity,
      tile_size=args.tile_size,
      use_time_limit=False,
      # task_reset_behavior='respawn',
      use_subtasks=True,
      seed=args.seed,
      )

  num_episodes = dict(
    train=int(.2*1e6),
    test=int(.8*1e6))

  data = None
  for key in ['train', 'test']:
    level_kwargs = babyai_utils.constuct_kitchenmultilevel_kwargs(
        task_dicts=tasks[key],
        level_kwargs=level_kwargs,
        sets=sets)


    env = MultiLevel(level_kwargs)
    env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)

    # ======================================================
    # Sample episodes
    # ======================================================
    for episode_idx in range(num_episodes[key]):
        env.seed(episode_idx)
        obs = env.reset()
        if args.visualize:
          plot_fn(obs)

        bot = KitchenBot(env)
        obss, actions, rewards, dones = bot.generate_traj(
          plot_fn=plot_fn if args.visualize else lambda x:x)
        import ipdb; ipdb.set_trace()

  save_data(data)
  import ipdb; ipdb.set_trace()


if __name__ == '__main__':
  main()