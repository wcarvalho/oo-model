import yaml
import ipdb
import cv2
import numpy as np
import time
from envs.babyai_kitchen import babyai_utils
from envs.babyai_kitchen.bot import KitchenBot

from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper
import gym_minigrid.window


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tasks', help='tasks file', 
      default='envs/babyai_kitchen/tasks/unseen_arg/length=2_no_dist.yaml')
    parser.add_argument('--task_kind', help='task_kind', 
      default=None)
    parser.add_argument('--sets', help='sets file',
        default="envs/babyai_kitchen/tasks/default_sets.yaml")
    parser.add_argument('--missions', help='# of unique missions',
        default=10)
    parser.add_argument('--room-size', type=int, default=7)
    parser.add_argument('--agent-view-size', type=int, default=5)
    parser.add_argument('--random-object-state', type=int, default=0)
    parser.add_argument('--num-rows', type=int, default=1)
    parser.add_argument('--tile-size', type=int, default=8)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--show', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--check', type=int, default=0)
    parser.add_argument('--check-end', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=2)
    args = parser.parse_args()

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
    if args.num_rows:
        level_kwargs['num_rows'] = args.num_rows
        level_kwargs['num_cols'] = args.num_rows

    key = 'train' if args.train else 'test'
    level_kwargs = babyai_utils.constuct_kitchenmultilevel_kwargs(
        task_dicts=tasks[key],
        level_kwargs=level_kwargs,
        sets=sets)


    env = MultiLevel(level_kwargs)
    # mimic settings during training
    env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)
    render_kwargs = {'tile_size' : env.tile_size}

    if args.show:
      window = gym_minigrid.window.Window('kitchen')
      window.show(block=False)

    def combine(full, partial):
      full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
      return np.concatenate((full_small, partial), axis=1)


    def move(action : str):
      # idx2action = {idx:action for action, idx in env.actions.items()}
      obs, reward, done, info = env.step(env.actiondict[action])
      full = env.render('rgb_array', tile_size=env.tile_size, highlight=True)
      window.show_img(combine(full, obs['image']))

    def show(obs):
      full = env.render('rgb_array', **render_kwargs)
      window.set_caption(obs['mission'])
      window.show_img(combine(full, obs['image']))
      if int(args.check):
        import ipdb; ipdb.set_trace()
      else:
        time.sleep(.05)


    for mission_indx in range(int(args.missions)):
        env.seed(mission_indx)
        obs = env.reset()
        print("="*50)
        print(f"Reset {mission_indx}")
        print("="*50)
        print("Task:", obs['mission'])
        # print("Image Shape:", obs['image'].shape)

        if args.show:
          full = env.render('rgb_array', **render_kwargs)
          window.set_caption(obs['mission'])
          window.show_img(combine(full, obs['image']))

        bot = KitchenBot(env)
        obss, actions, rewards, dones = bot.generate_traj(
          plot_fn=show if args.show else lambda x:x)
        print('Rewards:', sum(rewards))
        print('Epsiode Length:', len(rewards))
        if args.check_end:
          import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()
