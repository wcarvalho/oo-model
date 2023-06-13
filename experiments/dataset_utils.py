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
  shift_discount: bool = True,
  trace_length: int = 10,
  obs_constructor=None) -> Callable[[jax_types.PRNGKey], Iterator[types.Transition]]:
  """Returns the demonstration dataset factory for the given dataset.`

  Args:
      data_directory (str): directiry to get data from
      batch_size (int): batch size.
      shift_discount (bool, optional): if discount=1 at T-1, then we need to shift it so discount=0. Currently, BabyAI env logger does this. Defaults to True.
      trace_length (int, optional): length of batches. Defaults to 10.
      obs_constructor (_type_, optional): constructor of observations. Defaults to None.

  Returns:
      Callable[[jax_types.PRNGKey], Iterator[types.Transition]]: _description_
  """


  def demonstration_dataset_factory(
      random_key: jax_types.PRNGKey,
      split: str = 'train',
      buffer_size: int = 10_000) -> Iterator[types.Transition]:
    """Returns an iterator of demonstration samples."""

    # load dataset from directory
    builder = tf_tfds.builder_from_directory(data_directory)
    dataset = builder.as_dataset(split=split)

    # Shuffle the datasets
    total_examples = builder.info.splits[split].num_examples
    buffer_size = buffer_size or total_examples
    print(f"Buffer size: {buffer_size}/{total_examples}")

    buffer_percent = 100*(float(buffer_size)/total_examples)
    print("Buffer percent:", buffer_percent)
    dataset = dataset.shuffle(
        buffer_size=buffer_size,
        reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if shift_discount:
      def concatenate_episode(episode):
        episode[rlds.STEPS] = rlds.transformations.concatenate(
            episode[rlds.STEPS],
            rlds.transformations.zero_dataset_like(
                dataset.element_spec[rlds.STEPS]))
        return episode
      # Concatenates the existing dataset with a zeros-like dataset.
      # will later shift by 1 and this avoid deleting real data when doing so.
      # will instead consume zeros.
      dataset = dataset.map(concatenate_episode)

    # turn observations into obs-action-reward
    dataset = dataset.map(
        partial(episode_steps_to_oar_observations, obs_constructor=obs_constructor))

    if shift_discount:
      # Shifts discount 1 step backward in all episodes. (if negative, moves forward)
      dataset = dataset.map(partial(shift_episode, timesteps=1))

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


def make_supervised_rl_factory(
        data_directory: str,
        batch_size: int,
        trace_length: int = 10,
        obs_constructor=None,
        buffer_size: int = 1000,
        **kwargs) -> Callable[[jax_types.PRNGKey], Iterator[types.Transition]]:
  """Returns the demonstration dataset factory for the given dataset.`

  Args:
      data_directory (str): directiry to get data from
      batch_size (int): batch size.
      shift_discount (bool, optional): if discount=1 at T-1, then we need to shift it so discount=0. Currently, BabyAI env logger does this. Defaults to True.
      trace_length (int, optional): length of batches. Defaults to 10.
      obs_constructor (_type_, optional): constructor of observations. Defaults to None.

  Returns:
      Callable[[jax_types.PRNGKey], Iterator[types.Transition]]: _description_
  """

  def prepare_acme_dataset_iterator(dataset):
    dataset = dataset.shuffle(
          buffer_size=buffer_size,
          reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # Concatenates the existing dataset with a zeros-like dataset.
    # will later shift by 1 and this avoid deleting real data when doing so.
    # will instead consume zeros.
    def concatenate_episode(episode):
      episode[rlds.STEPS] = rlds.transformations.concatenate(
          episode[rlds.STEPS],
          rlds.transformations.zero_dataset_like(
              dataset.element_spec[rlds.STEPS]))
      return episode
    dataset = dataset.map(concatenate_episode)

    # turn observations into obs-action-reward
    dataset = dataset.map(
        partial(episode_steps_to_oar_observations, obs_constructor=obs_constructor))

    # Shifts discount 1 step backward in all episodes
    dataset = dataset.map(partial(shift_episode, timesteps=1))

    # turn dataset into n-step trajectories per datapoint
    dataset = dataset.map(
        lambda e: episode_steps_to_nstep_transition(e, trace_length))

    # batch into n-step trajectories to get [B, T] data-points per sample
    dataset = rlds.transformations.batch(
        dataset.flat_map(
            lambda episode: episode[rlds.STEPS]),
        batch_size,
        shift=1, drop_remainder=True)

    iterator = RestartableIteratorWrapper(
        dataset=dataset,
        cnstr=(lambda x:
               utils.multi_device_put(x.as_numpy_iterator(),
                                      devices=jax.local_devices(),
                                      split_fn=None)
               )
    )
    return iterator

  def demonstration_dataset_factory(
          random_key: jax_types.PRNGKey,
          split: str = 'train',
          split_size: int = 100,
          percent: int = 100) -> Iterator[types.Transition]:
    """Returns an iterator of demonstration samples."""
    del random_key

    # load dataset from directory
    builder = tf_tfds.builder_from_directory(data_directory)

    if split_size < 100:
      if percent < 100:
         split_size = int(split_size*percent/100)
      split = [
          f'{split}[:{split_size}%]',
          f'{split}[{split_size}%:]',
          ]
      train_dataset, validation_dataset = builder.as_dataset(split=split)
      print('split:', split)
    else:
      if percent < 100:
         split = f'{split}[:{percent}%]'
      print('split:', split)
      dataset = builder.as_dataset(split=split)
      return prepare_acme_dataset_iterator(dataset)

    # # Compute the size of the dataset
    # # dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    # # dataset_size = builder.info.splits[split].num_examples

    # if percent < 100:
    #   dataset_size = (percent / 100) * dataset_size
    #   print('percent_size', dataset_size)
    #   # Create the train and validation subsets
    #   dataset = dataset.take(int(dataset_size))

    # Compute the number of examples for the train split and for the subset
    # num_train_examples = int((split_size / 100) * dataset_size)
    # print('num_train_examples', num_train_examples)

    # # Create the train and validation subsets
    # train_dataset = dataset.take(num_train_examples)

    # if split_size == 100:
    #   train_iterator = prepare_acme_dataset_iterator(train_dataset)
    #   return train_iterator

    # num_valid_examples = int(((100 - split_size) / 100) * dataset_size)
    # print('validation_examples', num_valid_examples)
    # validation_dataset = dataset.skip(num_train_examples).take(
    #     num_valid_examples)

    train_iterator = prepare_acme_dataset_iterator(train_dataset)
    validation_iterator = prepare_acme_dataset_iterator(validation_dataset)

    return train_iterator, validation_iterator

  return demonstration_dataset_factory
