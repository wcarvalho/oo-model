"""Combining observation, action, reward, task embeddings."""

from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp


class OARTEmbedding(hk.Module):
  """Module for embedding (observation, action, reward, task) inputs together."""

  def __init__(self, num_actions, concat=True, observation=True, reward=True, **kwargs):
    super().__init__()
    self.num_actions = num_actions
    self.concat = concat
    self.observation = observation
    self.reward = reward

  def __call__(self,
    inputs: observation_action_reward.OAR, obs: jnp.array=None, task=None) -> jnp.ndarray:
    """Embed each of the (observation, action, reward) inputs & concatenate."""

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs.action, num_classes=self.num_actions)  # [T?, B, A]

    # Map rewards -> [-1, 1].
    if self.reward:
      reward = jnp.tanh(inputs.reward)

      # Add dummy trailing dimensions to rewards if necessary.
      while reward.ndim < action.ndim:
        reward = jnp.expand_dims(reward, axis=-1)

      # Concatenate on final dimension.
      items = [action, reward]
    else:
      items = [action]

    if task is not None:
      items.append(task)

    if self.observation:
      assert obs is not None, "provide observation"
      items.append(obs)

    if self.concat:
      items = jnp.concatenate(items, axis=-1)  # [T?, B, D+A+1]

    return items