# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

import collections
import datetime
import jax
import numpy as np
import time

from pathlib import Path

try:
  import wandb
  WANDB_AVAILABLE=True
except ImportError:
  WANDB_AVAILABLE=False

def gen_log_dir(
    base_dir="results/",
    date=True,
    hourminute=False,
    seed=None,
    return_kwpath=False,
    path_skip=[],
    **kwargs):

  strkey = '%Y.%m.%d'
  if hourminute:
    strkey += '-%H.%M'
  job_name = datetime.datetime.now().strftime(strkey)
  kwpath = ','.join([f'{key[:4]}={value}' for key, value in kwargs.items() if not key in path_skip])

  if date:
    path = Path(base_dir).joinpath(job_name).joinpath(kwpath)
  else:
    path = Path(base_dir).joinpath(kwpath)

  if seed is not None:
    path = path.joinpath(f'seed={seed}')

  if return_kwpath:
    return str(path), kwpath
  else:
    return str(path)


def flatten_dict(d, parent_key='', sep='_'):
  """Take a infinitely nested dict of dict and make it {key: value}."""
  items = []
  for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
          items.extend(flatten_dict(v, new_key, sep=sep).items())
      else:
          items.append((new_key, v))
  return dict(items)


def copy_numpy(values):
  return jax.tree_map(np.array, values)



def make_logger(
    log_dir: str,
    label: str,
    save_data: bool = False,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    use_tensorboard: bool = False,
    use_wandb: bool = True,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = copy_numpy,
    steps_key: str = 'steps',
    log_with_key: Optional[str] = None,
    max_number_of_steps: Optional[int] = None,
) -> base.Logger:
  """Makes a default Acme logger.

    Loggers/Filters used:
      - TerminalLogger: log to terminal
      - CSVLogger (off by default): save data as csv
      - WandbLogger: save data to wandb
      - Dispatcher: aggregates loggers (all before act independently)
      - NoneFilter: removes NaN data
      - AsyncLogger
      - HasKeyFilter: only write data for specified key
      - TimeFilter: how often to write data
  
  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.
    log_with_key: only log things with this key.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  # del steps_key
  if not print_fn:
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]

  if save_data:
    loggers.append(csv.CSVLogger(log_dir, label=label))

  if use_tensorboard:
    raise NotImplementedError

  if use_wandb and WANDB_AVAILABLE:
    loggers.append(WandbLogger(
      label=label,
      steps_key=steps_key,
      max_number_of_steps=max_number_of_steps
      ))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)

  if log_with_key is not None:
    logger = HasKeyFilter(logger, key=log_with_key)

  logger = filters.TimeFilter(logger, time_delta)

  return logger


def _format_key(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  new = key.title().replace("_", "").replace("/", "-")
  return new

def _format_loss(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  new = key.title().replace("_", "")
  return new

class WandbLogger(base.Logger):
  """Logs to a tf.summary created in a given logdir.
  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
      self,
      label: str = 'Logs',
      labels_skip=('Loss'),
      steps_key: Optional[str] = None,
      max_number_of_steps: Optional[int] = None,
  ):
    """Initializes the logger.
    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
      steps_key: key to use for steps. Must be in the values passed to write.
    """
    self._time = time.time()
    self.label = label
    self.labels_skip =labels_skip
    self._iter = 0
    # self.summary = tf.summary.create_file_writer(logdir)
    self._steps_key = steps_key
    self.max_number_of_steps = max_number_of_steps
    if max_number_of_steps is not None:
      logging.warning(f"Will exit after {max_number_of_steps} steps")

  def try_terminate(self, step: int):

    if step > int(1.05*self.max_number_of_steps):
      logging.warning("Exiting launchpad")
      import launchpad as lp  # pylint: disable=g-import-not-at-top
      lp.stop()
      import signal
      signal.raise_signal( signal.SIGTERM )


  def write(self, values: base.LoggingData):
    if self._steps_key is not None and self._steps_key not in values:
      logging.warning('steps key "%s" not found. Skip logging.', self._steps_key)
      logging.warning('Available keys:', str(values.keys()))
      return

    step = values[self._steps_key] if self._steps_key is not None else self._iter


    to_log={}
    for key in values.keys() - [self._steps_key]:

      if self.label in self.labels_skip: # e.g. [Loss]
        key_pieces = key.split("/")
        if len(key_pieces) == 1: # e.g. [step]
          name = f'{self.label}/{_format_key(key)}'
        else: 
          if 'grad' in key.lower():
          # e.g. [MeanGrad/FarmSharedOutput/~/FeatureAttention/Conv2D1] --> [Loss/MeanGrad-FarmSharedOutput-~-FeatureAttention-Conv2D1]
            name = f'z.grads/{_format_key(key)}'
          else: # e.g. [r2d1/xyz] --> [Loss_r2d1/xyz]
            name = f'{self.label}_{_format_loss(key)}'
      else: # e.g. [actor_SmallL2NoDist]
        name = f'{self.label}/{_format_key(key)}'

      to_log[name] = values[key]

    to_log[f'{self.label}/step']  = step

    wandb.log(to_log)

    self._iter += 1
    if self.max_number_of_steps is not None:
      if self._steps_key == 'actor_steps':
        self.try_terminate(step)
      else:
        try:
          self.try_terminate(values['actor_steps'])
        except Exception as e:
          pass


  def close(self):
    try:
      wandb.finish()
    except Exception as e:
      pass



class FlattenFilter(base.Logger):
  """"""

  def __init__(self, to: base.Logger):
    """Initializes the logger.
    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
    """
    self._to = to

  def write(self, values: base.LoggingData):
    values = flatten_dict(values, sep='/')
    self._to.write(values)

  def close(self):
    self._to.close()


class HasKeyFilter(base.Logger):
  """Logger which writes to another logger at a given time interval."""

  def __init__(self, to: base.Logger, key: str):
    """Initializes the logger.
    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      key: which key to to write
    """
    self._to = to
    self._key = key
    assert key is not None

  def write(self, values: base.LoggingData):
    hasdata = values.pop(self._key, None)
    if hasdata:
      self._to.write(values)

  def close(self):
    self._to.close()