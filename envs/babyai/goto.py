import numpy as np
import collections
from gym import spaces


from gym_minigrid.minigrid import Grid, WorldObj
from babyai.levels.levelgen import RoomGridLevel, RejectSampling
from babyai.levels.verifier import GoToInstr, ObjDesc, PickupInstr

class GotoLevel(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self,
        room_size=8,
        num_dists=4,
        task_colors=None,
        task_types=None,
        instr='goto',
        seed=None,
        **kwargs):
      self.all_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
      self.all_types = ['key', 'box', 'ball']
      self.task_types = task_types or self.all_types
      self.task_colors = task_colors or self.all_colors
      self.num_dists = num_dists

      instr = instr.lower()
      assert instr in ("goto", 'pickup')
      if instr == "goto":
        self.InstrCls = GoToInstr
      else:
        self.InstrCls = PickupInstr

      super().__init__(
          num_rows=1,
          num_cols=1,
          room_size=room_size,
          seed=seed,
          **kwargs
      )

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, types=None):
      """
      Add random objects that can potentially distract/confuse the agent.
      Change: "types" is argument now and used for selecting.
      """

      # Collect a list of existing objects
      objs = []
      for row in self.room_grid:
          for room in row:
              for obj in room.objs:
                  objs.append((obj.type, obj.color))

      # List of distractors added
      dists = []

      types = types or self.task_types
      while len(dists) < num_distractors:

          _color = self._rand_elem(self.all_colors)
          _type = self._rand_elem(types)
          obj = (_type, _color)

          if all_unique and obj in objs:
              continue

          # Add the object to a random room if no room specified
          room_i = i
          room_j = j
          if room_i == None:
              room_i = self._rand_int(0, self.num_cols)
          if room_j == None:
              room_j = self._rand_int(0, self.num_rows)

          dist, pos = self.add_object(room_i, room_j, *obj)

          objs.append(obj)
          dists.append(dist)

      return dists

    def gen_mission(self):
      """Sample (color, type). Add n distractors off any color but different type."""
      self.place_agent()

      # -----------------------
      # task object
      # -----------------------
      task_color = self._rand_elem(self.task_colors)

      task_type = self._rand_elem(self.task_types)

      obj, _ = self.add_object(0, 0, task_type, task_color)

      # -----------------------
      # distractors
      # -----------------------
      distractor_types = set(self.all_types) - set([task_type])

      dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False, types=distractor_types)

      for dist in dists:
        overlap = dist.type == obj.type and dist.color == obj.color
        dist_tuple = (dist.type, dist.color)
        obj_tuple = (obj.type, obj.color)

        assert overlap is False, f"{obj_tuple}=={dist_tuple}"

      # Make sure no unblocking is required
      self.check_objs_reachable()

      self.instrs = self.InstrCls(ObjDesc(obj.type, obj.color))




if __name__ == '__main__':
    import gym_minigrid.window
    import time
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper
    import matplotlib.pyplot as plt 
    import cv2

    tile_size=12
    train = True
    check_end=True
    ntest = 1
    nresets=10
    nenv_steps = 100

    instr='pickup'
    room_size=5
    num_dists=3
    agent_view_size=5


    all_colors=['green', 'red', 'blue', 'purple', 'yellow', 'grey']
    if train:
      task_colors = all_colors[ntest:]
    else:
      task_colors = all_colors[:ntest]
    env = GotoLevel(
      room_size=room_size,
      agent_view_size=agent_view_size,
      num_dists=num_dists,
      instr=instr,
      task_colors=task_colors)
    env = RGBImgPartialObsWrapper(env, tile_size=tile_size)

    def combine(full, partial):
        full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
        return np.concatenate((full_small, partial), axis=1)

    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)

    def move(action : str):
      # idx2action = {idx:action for action, idx in env.actions.items()}
      obs, reward, done, info = env.step(env.actions[action])
      full = env.render('rgb_array', tile_size=tile_size, highlight=True)
      window.show_img(combine(full, obs['image']))

    for nenv in range(nresets):
      obs = env.reset()
      full = env.render('rgb_array', tile_size=tile_size, highlight=True)
      window.set_caption(obs['mission'])
      window.show_img(combine(full, obs['image']))

      rewards = []
      for step in range(nenv_steps):
          obs, reward, done, info = env.step(env.action_space.sample())
          rewards.append(reward)
          full = env.render('rgb_array', tile_size=tile_size, highlight=True)
          window.show_img(combine(full, obs['image']))
          if done:
            break

      print(f"Rewards={sum(rewards)}")
      if check_end:
        import ipdb; ipdb.set_trace()
    import ipdb; ipdb.set_trace()


