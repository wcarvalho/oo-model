"""
[1, B, T, ...]
ReplaySample(info=SampleInfo(
    key=(1, 64),
    probability=(1, 64),
    table_size=(1, 64),
    priority=(1, 64),
    times_sampled=(1, 64)),
    data=Step(
      observation=OAR(
        observation=Observation(image=(1, 64, 21, 40, 40, 3),
        mission=(1, 64, 21, 10)),
        action=(1, 64, 21),
        reward=(1, 64, 21)),
      action=(1, 64, 21),            X
      reward=(1, 64, 21),            X
      discount=(1, 64, 21),          X
      start_of_episode=(1, 64, 21),  X
      extras={
        'core_state': LSTMState(hidden=(1, 64, 256), cell=(1, 64, 256))}))

Default"
{'action': (), 
  discount': (), 
  is_first': (), 
  is_last': (), 
  is_terminal': (), 
  observation': {
    'image': (56, 56, 3), 
    mission': (10,)
  }, 
  reward': ()}
"""
from typing import Callable, Iterator, Tuple, Dict, Any, Optional

from functools import partial
from absl import flags
from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import bc
from acme.datasets import tfds
from acme.datasets.numpy_iterator import NumpyIterator
from acme.adders.reverb import base as reverb_base

from absl import app
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import lp_utils
from acme.agents.jax import mbop
from acme.wrappers import observation_action_reward
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import launchpad as lp
import numpy as np

from experiments import babyai_env_utils
from experiments import babyai_collect_data

import tensorflow_datasets as tf_tfds
import tensorflow as tf
import tree

import reverb
import rlds


def zeros_tf2jax(x: tf.TensorSpec):
    if x.dtype == tf.int64:
      dtype = jnp.float64
    elif x.dtype == tf.int32:
      dtype = jnp.int32
    elif x.dtype == tf.float64:
      dtype = jnp.float64
    elif x.dtype == tf.float32:
      dtype = jnp.float32
    elif x.dtype == tf.bool:
      dtype = jnp.bool
    elif x.dtype == tf.uint8:
      dtype = jnp.uint8
    else:
       raise NotImplementedError(f'Do not know how to handle dtype: {x.dtype}')
    return jnp.zeros(shape=x.shape, dtype=dtype)

def make_dataset_environment_spec(
      environment: dm_env.Environment,
      dataset: tf.data.Dataset,
      ) -> specs.EnvironmentSpec:
  """Returns an `EnvironmentSpec` describing values used by an environment."""
  step_spec = rlds.transformations.step_spec(dataset)

  environment_env_spec = specs.make_environment_spec(environment)
  def map2jax(y):
     return (jax.tree_map(lambda x: zeros_tf2jax(x), y))

  dataset_env_spec = specs.EnvironmentSpec(
      observations=map2jax(step_spec['observation']),
      actions=map2jax(step_spec['action']),
      rewards=map2jax(step_spec['reward']),
      discounts=map2jax(step_spec['discount']))

  def reconcile(env, d):
    if type(env) == specs.Array:
       return specs.Array(
          shape=d.shape,
          dtype=env.dtype,
          name=env.name)
    elif type(env) == specs.DiscreteArray:
       return env

    elif type(env) == specs.BoundedArray:
       ones_like_d = np.ones_like(d)
       return specs.BoundedArray(
          shape=d.shape,
          dtype=env.dtype,
          minimum=ones_like_d*env.minimum.min(),
          maximum=ones_like_d*env.maximum.max(),
          name=env.name)
    else:
        raise NotImplementedError(f"Do not know how to handle {type(v)}")

  return jax.tree_map(
    reconcile, environment_env_spec, dataset_env_spec)

def data_step_to_reverb_step(step):
    input_dict=dict(
        observation=step['observation'],
        action=step['action'],
        reward=step['reward'],
        discount=step['discount'],
        start_of_episode=step["is_first"],
    )
    return reverb_base.Trajectory(**input_dict)    

def steps_to_replay_sample(step, sequence_length: int=5):
    def zero_pad_transform(trajectory_steps):
        unpadded_length = len(tree.flatten(trajectory_steps)[0])

        # Do nothing if the sequence is already full.
        if unpadded_length != sequence_length:
            to_pad = sequence_length - unpadded_length
            pad = lambda x: tf.pad(x, [[0, to_pad]] + [[0, 0]] * (len(x.shape) - 1))

            trajectory_steps = tree.map_structure(pad, trajectory_steps)

        # Set the shape to be statically known, and checks it at runtime.
        def _ensure_shape(x):
            shape = tf.TensorShape([sequence_length]).concatenate(x.shape[1:])
            return tf.ensure_shape(x, shape)

        return tree.map_structure(_ensure_shape, trajectory_steps)

    trajectory = zero_pad_transform(step)
    info = tf.zeros_like(trajectory[rlds.ACTION])
    trajectory = data_step_to_reverb_step(trajectory)
    return reverb.ReplaySample(
        info=info,
        data=trajectory)

def episode_steps_to_nstep_transition(episode, n) -> tf.data.Dataset:
    episode[rlds.STEPS] = rlds.transformations.batch(
          episode[rlds.STEPS], size=n, shift=n-1, drop_remainder=False).map(
          partial(steps_to_replay_sample, sequence_length=n))
    return episode


def steps_to_oar(step, obs_constructor=None):
    """Given (o1, a1, r1, o2, a2, r2) return (o2=(a1, r1, o2), a2, r2)."""
    new_step = tree.map_structure(lambda x: x[1], step)

    observation = tree.map_structure(lambda x: x[1], step[rlds.OBSERVATION])
    if obs_constructor:
        observation = obs_constructor(**observation)
        
    new_observation = observation_action_reward.OAR(
        observation=observation,
        action=tree.map_structure(lambda x: x[0], step[rlds.ACTION]),
        reward=tree.map_structure(lambda x: x[0], step[rlds.REWARD]),
    )
    new_step[rlds.OBSERVATION] = new_observation
    return new_step

def episode_steps_to_oar_observations(episode, obs_constructor) -> tf.data.Dataset:
    episode[rlds.STEPS] = rlds.transformations.batch(
        episode[rlds.STEPS], size=2, shift=1, drop_remainder=True).map(
        partial(steps_to_oar, obs_constructor=obs_constructor))
    return episode

def shift_episode(episode, timesteps: int = 1):
  # Uses `shift_keys` to shift observations 2 steps backwards in an episode.
  episode[rlds.STEPS] = rlds.transformations.alignment.shift_keys(
      episode[rlds.STEPS], [rlds.DISCOUNT], timesteps)
  return episode

class RestartableIteratorWrapper:
  def __init__(self, dataset, cnstr):
      self.dataset = dataset
      self.cnstr = cnstr
      self.iterator = self.cnstr(dataset)

  def __iter__(self):
      return self

  def __next__(self):
      try:
          return next(self.iterator)
      except StopIteration:
          self.iterator = self.cnstr(self.dataset)  # Restart the iterator
          return next(self.iterator)

def make_demonstration_dataset_factory(
  data_directory: str,
  batch_size: int,
  shift_discount: int = 1,
  trace_length: int = 10,
  obs_constructor=None) -> Callable[[jax_types.PRNGKey], Iterator[types.Transition]]:
  """Returns the demonstration dataset factory for the given dataset."""


  def demonstration_dataset_factory(
      random_key: jax_types.PRNGKey) -> Iterator[types.Transition]:
    """Returns an iterator of demonstration samples."""

    # load dataset from directory
    dataset = tf_tfds.builder_from_directory(data_directory).as_dataset(split='all')

    # turn observations into obs-action-reward
    dataset = dataset.map(
        partial(episode_steps_to_oar_observations, obs_constructor=obs_constructor))

    if shift_discount:
      # Shifts discountK1 steps backward in all episodes. (if negative, moves forward)
      dataset = dataset.map(partial(shift_episode, timesteps=shift_discount))

    # turn dataset into n-step trajectories per datapoint
    dataset = dataset.map(lambda e: episode_steps_to_nstep_transition(e, trace_length))

    # batch into n-step trajectories to get [B, T] data-points per sample
    dataset = rlds.transformations.batch(
        dataset.flat_map(
          lambda episode: episode[rlds.STEPS]), 
          batch_size,
          shift=1, drop_remainder=True)

    iterator = RestartableIteratorWrapper(
        dataset=dataset,
        cnstr = (lambda x: 
          utils.multi_device_put(x.as_numpy_iterator(),
                                 devices=jax.local_devices(),
                                 split_fn=None)
          )
    )
    return iterator

  return demonstration_dataset_factory
