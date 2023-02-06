import pprint
import collections
from babyai.bot import Bot, GoNextToSubgoal, manhattan_distance, ExploreSubgoal
from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)

from envs.babyai_kitchen.levelgen import KitchenLevel

# from envs.babyai_kitchen.goto_avoid import GotoAvoidEnv

# class KitchenGoNextToSubgoal(GoNextToSubgoal):
#     """The subgoal for going next to objects or positions.

#     Parameters:
#     ----------
#     datum : (int, int) tuple or `ObjDesc` or object reference
#         The position or the decription of the object or
#         the object to which we are going.
#     reason : str
#         One of the following:
#         - `None`: go the position (object) and face it
#         - `"PutNext"`: go face an empty position next to the object specified by `datum`
#         - `"Explore"`: going to a position, just like when the reason is `None`. The only
#           difference is that with this reason the subgoal will be considered
#           exploratory

#     """

#     def replan_before_action(self):
#         target_obj = None
#         if isinstance(self.datum, WorldObj):
#             target_obj = self.datum
#             target_pos = target_obj.cur_pos
#         else:
#             target_pos = tuple(self.datum)

#         # The position we are on is the one we should go next to
#         # -> Move away from it
#         if manhattan_distance(target_pos, self.pos) == (1 if self.reason == 'PutNext' else 0):
#             def steppable(cell):
#                 return cell is None or (cell.type == 'door' and cell.is_open)
#             if steppable(self.fwd_cell):
#                 return self.actions.forward
#             if steppable(self.bot.mission.grid.get(*(self.pos + self.right_vec))):
#                 return self.actions.right
#             if steppable(self.bot.mission.grid.get(*(self.pos - self.right_vec))):
#                 return self.actions.left
#             # Spin and hope for the best
#             return self.actions.left

#         # We are facing the target cell
#         # -> subgoal completed
#         if self.reason == 'PutNext':
#             if manhattan_distance(target_pos, self.fwd_pos) == 1:
#                 if self.fwd_cell is None:
#                     self.bot.stack.pop()
#                     return
#                 if self.fwd_cell.type == 'door' and self.fwd_cell.is_open:
#                     # We can't drop an object in the cell where the door is.
#                     # Instead, we add a subgoal on the stack that will force
#                     # the bot to move the target object.
#                     self.bot.stack.append(GoNextToSubgoal(
#                         self.bot, self.fwd_pos + 2 * self.dir_vec))
#                     return
#         else:
#             if np.array_equal(target_pos, self.fwd_pos):
#                 self.bot.stack.pop()
#                 return

#         # We are still far from the target
#         # -> try to find a non-blocker path
#         path, _, _ = self.bot._shortest_path(
#             lambda pos, cell: pos == target_pos,
#         )

#         # No non-blocker path found and
#         # reexploration within the room is not allowed or there is nothing to explore
#         # -> Look for blocker paths
#         if not path:
#             path, _, _ = self.bot._shortest_path(
#                 lambda pos, cell: pos == target_pos,
#                 try_with_blockers=True
#             )

#         # No path found
#         # -> explore the world
#         if not path:
#             self.bot.stack.append(ExploreSubgoal(self.bot))
#             return

#         # So there is a path (blocker, or non-blockers)
#         # -> try following it
#         next_cell = path[0]

#         # Choose the action in the case when the forward cell
#         # is the one we should go next to
#         if np.array_equal(next_cell, self.fwd_pos):
#             if self.fwd_cell:
#                 # drop_pos = self.bot._find_drop_pos()
#                 # self.bot.stack.append(DropSubgoal(self.bot))
#                 # self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos))
#                 # self.bot.stack.append(PickupSubgoal(self.bot))
#                 return
#             else:
#                 return self.actions.forward

#         # The forward cell is not the one we should go to
#         # -> turn towards the direction we need to go
#         if np.array_equal(next_cell - self.pos, self.right_vec):
#             return self.actions.right
#         elif np.array_equal(next_cell - self.pos, -self.right_vec):
#             return self.actions.left

#         # If we reacher this point in the code,  then the cell is behind us.
#         # Instead of choosing left or right randomly,
#         # let's do something that might be useful:
#         # Because when we're GoingNextTo for the purpose of exploring,
#         # things might change while on the way to the position we're going to, we should
#         # pick this right or left wisely.
#         # The simplest thing we should do is: pick the one
#         # that doesn't lead you to face a non empty cell.
#         # One better thing would be to go to the direction
#         # where the closest wall/door is the furthest
#         distance_right = self.bot._closest_wall_or_door_given_dir(self.pos, self.right_vec)
#         distance_left = self.bot._closest_wall_or_door_given_dir(self.pos, -self.right_vec)
#         if distance_left > distance_right:
#             return self.actions.left
#         return self.actions.right


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
    self.stack = [GoNextToSubgoal(self, tuple(
        subgoal.goto.cur_pos)) for subgoal in subgoals]
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

