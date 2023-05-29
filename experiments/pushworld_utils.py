import dm_env
import os.path
import acme
import gym
import numpy as np
from typing import NamedTuple
from collections import namedtuple
from gym.spaces import Box, Dict
from pushworld.gym_env import PushWorldEnv
from pushworld.dm_env import PushWorldEnv as PushWorldEnvDM
from acme.wrappers import base
from acme.wrappers import GymWrapper
from dm_env import specs


class ObservationTuple(NamedTuple):
  image: acme.types.Nest
  mission: acme.types.Nest


class ObservationWrapper(acme.wrappers.base.EnvironmentWrapper):
  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action: acme.types.NestedArray) -> dm_env.TimeStep:
    # TODO this dtype conversion happens everytime
    # it may be better to set action space dtype in pushworld source to be
    # int32 as default (no need to must be int64)
    if action.dtype == np.int32:
      action = action.astype(np.int64)
    timestep = self._environment.step(action)
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    new_observation = ObservationTuple(
      image=timestep.observation,
      mission=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    return timestep._replace(observation=new_observation)

  def observation_spec(self):
    return ObservationTuple(
      image=self._environment.observation_spec(),
      mission=specs.BoundedArray((10,), np.int64, minimum=0, maximum=1)
    )


def make_environment(
  evaluation: bool = False,
  path='.',
  debug=False,
  nseeds=0,
  **kwargs,
  ) -> dm_env.Environment:

  # initialize dm_env
  dm_env = PushWorldEnvDM(
    puzzle_path="_pushworld/benchmark/puzzles/level0/base/train/level_0_base_train_0.pwp",
    max_steps=kwargs["horizon"],
    pixels_per_cell=12,
  )

  # set levelname for logging purpose
  dm_env.current_levelname = "pushworld_level_0_base_train_0"

  # apply wrappers to dm_env
  wrapper_list = [
    ObservationWrapper,
    acme.wrappers.ObservationActionRewardWrapper,
    acme.wrappers.SinglePrecisionWrapper,
  ]
  dm_env = acme.wrappers.wrap_all(dm_env, wrapper_list)

  # return dm_env
  return dm_env
