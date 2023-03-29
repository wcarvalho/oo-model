from typing import Callable
from acme import types as acme_types
from acme.wrappers import observation_action_reward

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from modules import vision
from modules import language

Array = acme_types.NestedArray
Image = acme_types.NestedArray
Action = acme_types.NestedArray
Reward = acme_types.NestedArray
Task = acme_types.NestedArray


@chex.dataclass(frozen=True)
class TorsoOutput:
  image: jnp.ndarray
  action: jnp.ndarray
  reward: jnp.ndarray
  task: jnp.ndarray


def struct_output(image, task, action, reward):
  return TorsoOutput(
    image=image,
    action=action,
    reward=reward,
    task=task,
  )


def concat(image, task, action, reward):
  return jnp.concatenate((image, task, action, reward), axis=-1)


class Torso(hk.Module):

  def __init__(self,
               num_actions: int,
               task_encoder: hk.Module,
               vision_torso: hk.Module,
               flatten_image: bool = True,
               image_dim: int = 0,
               task_dim: int = 0,
               output_fn: Callable[[Image, Task, Action, Reward], Array] = concat,
               name='torso'):
    super().__init__(name=name)
    self._num_actions = num_actions
    self._task_encoder = task_encoder
    self._vision_torso = vision_torso
    self._image_dim = image_dim
    self._output_fn = output_fn
    self._task_dim = task_dim
    self._flatten_image = flatten_image

  def __call__(self, inputs: observation_action_reward.OAR):
    batched = len(inputs.observation.image.shape) == 4
    observation_fn = self.unbatched
    if batched:
      observation_fn = jax.vmap(observation_fn)
    return observation_fn(inputs)

  def unbatched(self, inputs: observation_action_reward.OAR):
    """_no_ batch [B] dimension."""
    # compute task encoding
    task = self._task_encoder(inputs.observation.mission)
    if self._task_dim and self._task_dim > 0:
      task = hk.Linear(self._task_dim)(task)

    # get action one-hot
    action = jax.nn.one_hot(
        inputs.action, num_classes=self._num_actions)

    # compute image encoding
    inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)
    image = self._vision_torso(inputs.observation.image/255.0)

    if self._flatten_image:
      image = jnp.reshape(image, (-1))
    if self._image_dim and self._image_dim > 0:
      image = hk.Linear(self._image_dim)(image)

    # Map rewards -> [-1, 1].
    # reward = jnp.tanh(inputs.reward)
    reward = jnp.expand_dims(inputs.reward, axis=-1)

    return self._output_fn(
      image=image,
      task=task,
      action=action,
      reward=reward)
