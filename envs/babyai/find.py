"""
Check:
1. reward function
"""
import numpy as np
from gym import spaces


from gym_minigrid.minigrid import Grid, WorldObj
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import GoToInstr, ObjDesc, PickupInstr

import itertools
class NewGotoLevel(RoomGridLevel):
  """
  Go to the red ball, single room, with distractors.
  The distractors are all grey to reduce perceptual complexity.
  This level has distractors but doesn't make use of language.
  """

  def __init__(self,
      num_dists=6,
      room_size=6,
      dist_types=None,
      instr='goto',
      seed=None,
      num_rows=1,
      num_cols=1,
      **kwargs):
      


    self.task_color = "green"
    self.task_type = "box"

    self.all_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
    all_dist_objects = list(itertools.product(self.all_colors, dist_types))
    task_object = (self.task_color, self.task_type)

    self.all_dist_objects = set(all_dist_objects) - set([task_object])
    self.num_dists = num_dists
    self.num_rows = num_rows
    self.num_cols = num_cols

    instr = instr.lower()
    assert instr in ("goto", 'pickup')
    if instr == "goto":
      self.InstrCls = GoToInstr
    else:
      self.InstrCls = PickupInstr

    super().__init__(
        num_rows=num_rows,
        num_cols=num_cols,
        room_size=room_size,
        seed=seed,
        **kwargs
    )

  def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, possible_dists=None):
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
    while len(dists) < num_distractors:

        _color, _type = self._rand_elem(possible_dists)
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


  def step(self, action):
    obs, reward, done, info = super().step(action)
    reward = float(reward > 0)
    return obs, reward, done, info

  def gen_mission(self):
    """Sample (color, type). Add n distractors off any color but different type."""
    self.place_agent(0,0)
    self.connect_all()
    # -----------------------
    # task object
    # -----------------------

    obj, _ = self.add_object(self.num_cols-1, self.num_rows-1, self.task_type, self.task_color)

    # -----------------------
    # distractors
    # -----------------------

    dists = self.add_distractors(
      num_distractors=self.num_dists,
      all_unique=False,
      possible_dists=self.all_dist_objects)


    for dist in dists:
      overlap = dist.type == obj.type and dist.color == obj.color
      dist_tuple = (dist.type, dist.color)
      obj_tuple = (obj.type, obj.color)

      assert overlap is False, f"{obj_tuple}=={dist_tuple}"

    # Make sure no unblocking is required
    self.check_objs_reachable()

    self.instrs = self.InstrCls(ObjDesc(obj.type, obj.color))


class NewGotoLevelTrain3rooms(NewGotoLevel):
  def __init__(self,
    *args,
    dist_types = ['ball', 'box'],
    num_rows=3,
    num_cols=3,
    room_size=6,
    num_dists=10,
    **kwargs):
      super().__init__(*args,
        dist_types=dist_types,
        num_rows=num_rows,
        num_cols=num_cols,
        room_size=room_size,
        num_dists=num_dists,
        **kwargs)

class NewGotoLevelTest3rooms(NewGotoLevel):
  def __init__(self,
    *args,
    dist_types = ['ball', 'box', 'key'],
    num_rows=3,
    num_cols=3,
    room_size=6,
    num_dists=10,
    **kwargs):
      super().__init__(*args,
        dist_types=dist_types,
        num_rows=num_rows,
        num_cols=num_cols,
        room_size=room_size,
        num_dists=num_dists,
        **kwargs)


class NewGotoLevelTrain1room(NewGotoLevel):
  def __init__(self,
    *args,
    dist_types = ['ball', 'box'],
    num_rows=1,
    num_cols=1,
    room_size=8,
    num_dists=6,
    **kwargs):
      super().__init__(*args,
        dist_types=dist_types,
        num_rows=num_rows,
        num_cols=num_cols,
        room_size=room_size,
        num_dists=num_dists,
        **kwargs)

class NewGotoLevelTest1room(NewGotoLevel):
  def __init__(self,
    *args,
    dist_types = ['ball', 'box', 'key'],
    num_rows=1,
    num_cols=1,
    room_size=8,
    num_dists=6,
    **kwargs):
      super().__init__(*args,
        dist_types=dist_types,
        num_rows=num_rows,
        num_cols=num_cols,
        room_size=room_size,
        num_dists=num_dists,
        **kwargs)



if __name__ == '__main__':
    import gym_minigrid.window
    import time
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper
    import matplotlib.pyplot as plt 
    import cv2

    tile_size=12
    train = True
    check_end=False
    ntest = 1
    nresets=10
    nenv_steps = 1000

    # kwargs=dict(
    #   instr='goto',
    #   agent_view_size=7,
    #   num_dists=6,
    #   num_rows=1,
    #   num_cols=1,
    #   room_size=8,
    #   )


    # if train:
    #   dist_types = ['ball', 'box']
    # else:
    #   dist_types = ['ball', 'box', 'key']
    levelcls = NewGotoLevelTest3rooms
    levelcls = NewGotoLevelTrain3rooms
    # levelcls = NewGotoLevelTest1room
    # levelcls = NewGotoLevelTrain1room

    env = levelcls()
    print(levelcls)
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

      print(f"Rewards={sum(rewards), type(rewards[0])}")
      if check_end:
        import ipdb; ipdb.set_trace()
    import ipdb; ipdb.set_trace()


