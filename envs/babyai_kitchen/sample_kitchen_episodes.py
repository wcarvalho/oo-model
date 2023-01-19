import ipdb
import cv2
import numpy as np
import time
from envs.babyai_kitchen.bot import KitchenBot
from envs.babyai_kitchen.levelgen import KitchenLevel
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
import gym_minigrid.window


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--missions', help='# of unique missions', default=10)
    parser.add_argument('--num-distractors', type=int, default=0)
    parser.add_argument('--room-size', type=int, default=5)
    parser.add_argument('--agent-view-size', type=int, default=8)
    parser.add_argument('--render-mode', type=str, default='human')
    parser.add_argument('--task-kinds', type=str, default=[
      'cook', 'clean', 'slice'], nargs="+")

    parser.add_argument('--objects', type=str, default=[], nargs="+")
    parser.add_argument('--random-object-state', type=int, default=0)
    parser.add_argument('--num-rows', type=int, default=1)
    parser.add_argument('--tile-size', type=int, default=8)
    parser.add_argument('--partial-obs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--check', type=int, default=0)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--check-end', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=2)
    args = parser.parse_args()


    kwargs={}

    kwargs['num_dists'] = args.num_distractors
    if args.num_rows:
        kwargs['num_rows'] = args.num_rows
        kwargs['num_cols'] = args.num_rows
    env = KitchenLevel(
        room_size=args.room_size,
        agent_view_size=args.agent_view_size,
        random_object_state=args.random_object_state,
        task_kinds=args.task_kinds,
        objects=args.objects,
        verbosity=args.verbosity,
        tile_size=args.tile_size,
        load_actions_from_tasks=False,
        use_time_limit=False,
        use_subtasks=True,
        task_reset_behavior='remove',
        seed=args.seed,
        **kwargs)
    # mimic settings during training
    if args.partial_obs:
      env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)
    else:
      env = RGBImgFullyObsWrapper(env, tile_size=args.tile_size)
    render_kwargs = {'tile_size' : env.tile_size}

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

    all_rewards = []
    for mission_indx in range(int(args.missions)):
        env.seed(mission_indx)
        obs = env.reset()
        print("="*50)
        print("Reset")
        print("="*50)
        print("Task:", obs['mission'])
        print("Distractors:", [d.type for d in env.distractors_added])
        print("Image Shape:", obs['image'].shape)


        full = env.render('rgb_array', **render_kwargs)
        window.set_caption(obs['mission'])
        window.show_img(combine(full, obs['image']))


        bot = KitchenBot(env)
        obss, actions, rewards, dones = bot.generate_traj(plot_fn=show)
        for _ in range(3):
          obs, reward, done, info = env.step(env.action_space.sample())
          rewards.append(reward)
        print("Reward:", sum(rewards))
        all_rewards.append(sum(rewards))
        if args.check_end and ((mission_indx+1) % args.check_end == 0):
          import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    main()
