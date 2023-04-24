"""Evaluation Observers."""

from absl import logging
import collections
import abc
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Sequence, Union
from acme.utils import signals
import os.path
from acme.utils.loggers.base import Logger
from acme.utils.observers import EnvLoopObserver
from acme.utils import paths

import dm_env
import pandas as pd
from dm_env import specs
import jax.numpy as jnp
import numpy as np
import operator
import tree

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

Number = Union[int, float, np.float32, jnp.float32]


class LevelAvgReturnObserver(EnvLoopObserver):
  """Metric: Average return over many episodes"""
  def __init__(self, reset=100, get_task_name=None):
    super(LevelAvgReturnObserver, self).__init__()
    self.returns = collections.defaultdict(list)
    self.level = None
    self.reset = reset
    self.idx = 0
    if get_task_name is None:
      get_task_name = lambda env: "Episode"
      logging.info("WARNING: if multi-task, suggest setting `get_task_name` in `LevelAvgReturnObserver`. This will log separate statistics for each task.")
    self._get_task_name = get_task_name

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state __after__ reset."""
    self.idx += 1
    if self.level is not None:
      self.returns[self.level].append(self._episode_return)

    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())
    self.level = self._get_task_name(env)


  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    result = {}

    if self.idx % self.reset == 0:
      # add latest (otherwise deleted)
      self.returns[self.level].append(self._episode_return)

      for key, returns in self.returns.items():
        if not returns: continue
        avg = np.array(returns).mean()
        result[f'0.task/{key}/avg_return'] = float(avg)
        self.returns[key] = []

      result['log_data'] = True

    return result


class Queue:
  def __init__(self, max_len=5):
      self.queue = np.zeros(max_len)
      self.max_len = max_len
      self.head = 0
      self.tail = 0

  def enqueue(self, item):
      if self.tail == self.head and self.queue[self.tail] != 0:
          self.tail = (self.tail + 1) % self.max_len
      self.queue[self.head] = item
      self.head = (self.head + 1) % self.max_len

  def dequeue(self):
      if self.is_empty():
          return None
      item = self.queue[self.tail]
      self.queue[self.tail] = 0
      self.tail = (self.tail + 1) % self.max_len
      return item

  def size(self):
      return np.count_nonzero(self.queue)

  def is_empty(self):
      return self.size() == 0

  def last(self):
      if self.is_empty():
          return np.array([])
      elif self.head > self.tail:
          return self.queue[self.tail:self.head]
      else:
          return np.concatenate((self.queue[self.tail:], self.queue[:self.head]))


class ExitObserver(EnvLoopObserver):
  """Observe running averga and exit if above threshold."""

  def __init__(self, exit_at_success=.99, window_length=500):
    super(ExitObserver, self).__init__()
    self.idx = 0
    raise NotImplementedError("not done checking")

    self.exit_at_success = exit_at_success
    self.window_length = window_length
    self.queue = Queue(max_len=window_length)

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state __after__ reset."""
    self._episode_return = tree.map_structure(
        _generate_zeros_from_spec,
        env.reward_spec())

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
        operator.iadd,
        self._episode_return,
        timestep.reward)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    self.queue.enqueue(float(self._episode_return))
    self.idx += 1

    if self.idx > self.window_length:
      avg = self.queue.last().mean()
      if avg > self.exit_at_success:
        logging.warning(
            f"Exiting because episode average {avg} > threshold {self.exit_at_success}")
        import launchpad as lp  # pylint: disable=g-import-not-at-top
        lp.stop()
    return {}
