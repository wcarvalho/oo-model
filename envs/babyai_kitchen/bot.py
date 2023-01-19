import pprint
import collections
from babyai.bot import Bot, GoNextToSubgoal, manhattan_distance, PickupSubgoal, DropSubgoal
from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)

from envs.babyai_kitchen.levelgen import KitchenLevel
from envs.babyai_kitchen.goto_avoid import GotoAvoidEnv

class KitchenBot(Bot):
  """docstring for KitchenBot"""
  def __init__(self, env : KitchenLevel):

    # Mission to be solved
    self.mission = mission = env

    # Grid containing what has been mapped out
    self.grid = Grid(mission.width, mission.height)

    # Visibility mask. True for explored/seen, false for unexplored.
    self.vis_mask = np.zeros(shape=(mission.width, mission.height), dtype=np.bool)

    # Stack of tasks/subtasks to complete (tuples)
    self.subgoals = subgoals = self.mission.task.subgoals()
    self.stack = [GoNextToSubgoal(self, tuple(subgoal.goto.cur_pos)) for subgoal in subgoals]
    self.stack.reverse()

    # How many BFS searches this bot has performed
    self.bfs_counter = 0

    # How many steps were made in total in all BFS searches
    # performed by this bot
    self.bfs_step_counter = 0


  def generate_traj(self, action_taken=None, plot_fn=lambda x:x):

    steps_left = len(self.stack)
    env = self.mission

    all_obs = []
    all_action = []
    all_reward = []
    all_done = []

    def step_update(_action):
      _obs, _reward, _done, _info = env.step(_action)
      all_obs.append(_obs)
      all_action.append(_action)
      all_reward.append(_reward)
      all_done.append(_done)
      return _obs, _reward, _done, _info

    idx = 0

    subgoals = iter(self.subgoals)
    current_subgoal = next(subgoals)
    while self.stack:
      idx += 1
      if idx > 1000:
        raise RuntimeError("Taking too long")

      action = self.replan(action_taken)

      # -----------------------
      # done??
      # -----------------------
      if action == env.actions.done:
        return all_obs, all_action, all_reward, all_done

      # -----------------------
      # take actions
      # -----------------------
      obs, reward, done, info = step_update(action)

      plot_fn(obs)

      # -----------------------
      # subgoal object in front? do actions
      # -----------------------
      object_infront = env.grid.get(*env.front_pos)
      if object_infront and object_infront.type == current_subgoal.goto.type:
        for action_str in current_subgoal.actions:
          interaction = env.actiondict[action_str]
          obs, reward, done, info = step_update(interaction)

          plot_fn(obs)

        try:
          current_subgoal = next(subgoals)
        except:
          pass

      # -----------------------
      # book-keeping
      # -----------------------
      steps_left = len(self.stack)

      action_taken = action



  def _check_erroneous_box_opening(self, action):
    # ignore this
    pass


class GotoAvoidBot(Bot):
  """docstring for KitchenBot"""
  def __init__(self, env: GotoAvoidEnv):

    # Mission to be solved
    self.mission = mission = env

    # Grid containing what has been mapped out
    self.grid = Grid(mission.width, mission.height)

    # Visibility mask. True for explored/seen, false for unexplored.
    # EVERYTHING VISIBLE
    self.vis_mask = np.ones(shape=(mission.width, mission.height), dtype=np.bool)

    # Stack of tasks/subtasks to complete (tuples)
    # import ipdb; ipdb.set_trace()
    # self.subgoals = subgoals = self.mission.task.subgoals()
    # self.stack = [PickupSubgoal(self, tuple(subgoal.goto.cur_pos)) for subgoal in subgoals]
    # self.stack.reverse()

    # How many BFS searches this bot has performed
    self.bfs_counter = 0

    # How many steps were made in total in all BFS searches
    # performed by this bot
    self.bfs_step_counter = 0

  def get_task_objects(self):
    objects = self.mission.get_room(0, 0).objs
    task_objects = filter(lambda x: x.type in self.mission.task_objects, objects)
    return list(task_objects)


  def generate_traj(self, action_taken=None, plot_fn=lambda x:x, plot_verb=0):
    """
        - first, pick closest object
        - then replan until get to it
        - pick it up
        - repeat

    Args:
        action_taken (None, optional): Description
        plot_fn (TYPE, optional): Description
    
    Returns:
        TYPE: Description
    
    Raises:
        RuntimeError: Description
    """
    all_obs = []
    all_action = []
    all_reward = []
    all_done = []
    env = self.mission

    def step_update(_action):
      _obs, _reward, _done, _info = env.step(_action)
      if plot_verb > 0:
        plot_fn(_obs)
      all_obs.append(_obs)
      all_action.append(_action)
      all_reward.append(_reward)
      all_done.append(_done)
      return _obs, _reward, _done, _info

    idx = 0



    def get_neighbors(cell):
      """Get all neihboring cells of agents.
      """
      right = (cell[0], cell[1] + 1)
      left = (cell[0], cell[1] - 1)
      up = (cell[0] + 1, cell[1])
      down = (cell[0] - 1, cell[1])
      return [right, left, up, down]

    def which_position_closest(positions):
      """Get closest neighbor pos of object
      """
      closest_pos = None
      closest_len = 1e10
      paths = []
      for desired in positions:
        def match(pos, cell):
           is_desired = tuple(pos) == desired
           empty = self.mission.grid.get(*pos) is None
           return is_desired and empty

        path, finish, with_blockers = self._shortest_path(
          accept_fn=match)
        paths.append((path, finish, with_blockers))
        if path and len(path) < closest_len:
          closest_len = len(path)
          closest_pos = desired

      return closest_pos, closest_len

    def get_closest_object(objects):
      """Get closest object from collection of objects
      """
      closest_object = None
      closest_pos = None
      closest_len = 1e10
      task2closest=collections.defaultdict(list)
      for obj in task_objects:
        neighbors = get_neighbors(obj.cur_pos)
        neighbor_pos, neighbor_len = which_position_closest(neighbors)
        if neighbor_len < closest_len:
          closest_len = neighbor_len
          closest_object = obj
          closest_pos = neighbor_pos

        task2closest[obj.type].append(dict(pos=neighbor_pos,len=neighbor_len))
      return closest_object, task2closest

    while True:
      task_objects = self.get_task_objects()
      closest_object, task2closest = get_closest_object(task_objects)
      if closest_object is None:
        print("="*10, "Impossible level", "="*10)
        return None, None, None, None

      self.stack = [GoNextToSubgoal(self, tuple(closest_object.cur_pos))]
      while self.stack:
        while isinstance(self.stack[-1], DropSubgoal):
          raise RuntimeError
          self.stack.pop()
        idx += 1
        if idx > 1000:
          raise RuntimeError("Taking too long")

        action = self.replan(action_taken)

        # -----------------------
        # Went to it? Pick it up.
        # -----------------------
        if action == env.actions.done:
          action = env.actions.pickup
          obs, reward, done, info = step_update(action)

          if done:
            return all_obs, all_action, all_reward, all_done

          action_taken = action
          break

        # -----------------------
        # take actions
        # -----------------------
        obs, reward, done, info = step_update(action)

        if done:
          return all_obs, all_action, all_reward, all_done

        action_taken = action






  def _check_erroneous_box_opening(self, action):
    # ignore this
    pass
