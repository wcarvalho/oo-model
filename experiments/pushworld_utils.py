import dm_env
import os.path
import tree
import acme
import gym
import numpy as np
from typing import NamedTuple
from collections import namedtuple
from gym.spaces import Box, Dict
from pushworld.gym_env import PushWorldEnv
from acme.wrappers import base
from acme.wrappers import GymWrapper
from dm_env import specs


class ObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = Dict({
      "image": Box(0., 1., (84, 84, 3), np.float32),
      "mission": Box(0., 1024., (10,), np.int32),
    })
    
  def observation(self, obs):
    if isinstance(obs, tuple):
      obs = obs[0]
    elif isinstance(obs, np.ndarray):
      obs = obs
    else:
      raise ValueError("{}".format(type(obs)))

    # NOTE dummy mission values for now
    return {
      "image": obs,
      "mission": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    }


class ObservationTuple(NamedTuple):
  image: acme.types.Nest
  mission: acme.types.Nest


class CustomPushWorldEnv(dm_env.Environment):
  def __init__(self, max_steps):
    # set gym_env
    # ref: https://github.com/deepmind/pushworld/blob/main/python3/src/pushworld/gym_env.py
    self.env = PushWorldEnv(
      puzzle_path="_pushworld/benchmark/puzzles/level0/base/train/level_0_base_train_0.pwp",
      max_steps=max_steps,
      pixels_per_cell=12,
    )
    self.env = ObservationWrapper(self.env)

    # set dm_env
    self.dm_env = GymWrapper(self.env)

    # set levelname for logging purpose
    self.env.current_levelname = "pushworld_level_0_base_train_0"

    self.counter = 1
  
  def reset(self):
    obs = self.env.reset()
    obs = ObservationTuple(
      image=obs["image"],
      mission=obs["mission"],
    )
    return dm_env.restart(obs)

  def step(self, action: int) -> dm_env.TimeStep:
    obs, reward, done, truncated, info = self.env.step(action)
    obs = ObservationTuple(image=obs["image"], mission=obs["mission"])

    if done or truncated:
      return dm_env.termination(reward=reward, observation=obs)
    else:
      return dm_env.transition(reward=reward, observation=obs)

  def action_spec(self) -> specs.DiscreteArray:
    return self.dm_env.action_spec()
  
  def observation_spec(self):
    default = self.dm_env.observation_spec()
    spec = ObservationTuple(**default)
    return spec


def make_environment(
  evaluation: bool = False,
  path='.',
  debug=False,
  nseeds=0,
  **kwargs,
  ) -> dm_env.Environment:

  # initialize dm_env
  dm_env = CustomPushWorldEnv(max_steps=kwargs["horizon"])

  # apply wrappers to dm_env
  wrapper_list = [
    acme.wrappers.ObservationActionRewardWrapper,
    acme.wrappers.SinglePrecisionWrapper,
  ]
  dm_env = acme.wrappers.wrap_all(dm_env, wrapper_list)

  # return dm_env
  return dm_env
