# pylint: disable=g-bad-file-header
# Copyright 2019 The dm_env Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Kitchen reinforcement learning environment."""
from typing import NamedTuple
import os.path
import yaml

import dm_env
from dm_env import specs

from acme import types
from acme.wrappers import GymWrapper

import numpy as np

from envs.babyai_kitchen import babyai_utils
from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.levelgen import KitchenLevel

class Observation(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest

class SymbolicObservation(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest
  direction: types.Nest

def convert_rawobs(obs, symbolic=False):
    obs.pop('mission_idx', None)
    if symbolic:
      obs['image'] = obs['image'].astype(np.int32)
      return SymbolicObservation(**obs)
    else:
      return Observation(**obs)


class MultitaskKitchen(dm_env.Environment):
  """
  """

  def __init__(self,
    task_dicts: dict=None,
    task_kinds: list=None,
    tasks_file: dict=None,
    separate_eval: bool=False,
    room_size=10,
    task_reps=None,
    sets: str=None,
    agent_view_size=7,
    path='.',
    tile_size=8,
    step_penalty=0.0,
    wrappers=None,
    num_dists=0,
    symbolic=False,
    test_larger=False,
    **kwargs):
    """Initializes a new Kitchen environment.
    Args:
      rows: number of rows.
      columns: number of columns.
      seed: random seed for the RNG.
    """

    self.separate_eval = separate_eval
    level_kwargs = dict(
      room_size=room_size,
      agent_view_size=agent_view_size,
      tile_size=tile_size,
      num_dists=num_dists,
      task_reps=task_reps,
    )

    self.symbolic = symbolic
    self.step_penalty = step_penalty
    self.tasks_file = tasks_file
    # -----------------------
    # load level kwargs
    # -----------------------
    if task_dicts and task_kinds: raise RuntimeError
    if not (task_dicts or task_kinds): raise RuntimeError

    if task_dicts is not None:
      # load sets of objects to use
      if sets is None:
        sets = os.path.join(path, "envs/babyai_kitchen/tasks/default_sets.yaml")
      with open(sets, 'r') as f:
        sets = yaml.load(f, Loader=yaml.SafeLoader)
      # load all kwargs for all levels
      all_level_kwargs = babyai_utils.constuct_kitchenmultilevel_kwargs(
        task_dicts=task_dicts,
        level_kwargs=level_kwargs,
        sets=sets)
    else:
      all_level_kwargs = dict()
      for task in task_kinds:
          all_level_kwargs[task]=dict(
              task_kinds=[task],
              **level_kwargs
          )
    if test_larger:
      new_all_level_kwargs = dict()
      for key, level_kwargs in all_level_kwargs.items():
        room_size = level_kwargs['room_size']
        bigger_room_size = room_size + 2
        num_dists = level_kwargs['num_dists']
        more_dists = num_dists + 2

        key1 = f"{key}_r={bigger_room_size}_d={num_dists}"
        key2 = f"{key}_r={bigger_room_size}_d={more_dists}"
        new_all_level_kwargs[key] = level_kwargs
        new_all_level_kwargs[key1] = dict(level_kwargs)
        new_all_level_kwargs[key1].update(
          room_size=bigger_room_size,
        )
        new_all_level_kwargs[key2] = dict(level_kwargs)
        new_all_level_kwargs[key2].update(
          room_size=bigger_room_size,
          num_dists=more_dists,
        )
      all_level_kwargs = new_all_level_kwargs
    # from pprint import pprint
    # pprint(all_level_kwargs)
    self.all_level_kwargs = all_level_kwargs
    # -----------------------
    # load env
    # -----------------------
    self.env = MultiLevel(
        all_level_kwargs=all_level_kwargs,
        LevelCls=KitchenLevel,
        wrappers=wrappers,
        path=path,
        **kwargs)

    if wrappers:
      self.default_gym = self.env.env
      self.default_env = GymWrapper(self.env.env)
    else:
      self.default_gym = self.env
      self.default_env = GymWrapper(self.env)




  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    obs = self.env.reset()
    obs = convert_rawobs(obs, symbolic=self.symbolic)

    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    obs = convert_rawobs(obs, symbolic=self.symbolic)
    if self.step_penalty:
      reward = reward - self.step_penalty
    if done:
      if info['success']:
        return dm_env.termination(reward=reward, observation=obs)
      else:
        return dm_env.truncation(reward=reward, observation=obs)
    else:
      return dm_env.transition(reward=reward, observation=obs)


  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return self.default_env.action_spec()

  def observation_spec(self):
    default = self.default_env.observation_spec()
    spec = Observation(**default)
    return spec
