import haiku as hk
import jax
import jax.numpy as jnp

class Transition(hk.Module):
    """Dynamics transition module."""

    def __init__(
        self,
        num_blocks: int,
        name: str = "transition",
    ):
        """Init transition function."""
        super().__init__(name=name)
        self._num_blocks = num_blocks

    def __call__(
        self,
        action_onehot: jnp.ndarray,
        prev_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward transition function."""
        channels = prev_state.shape[-1]
        prev_state = jax.nn.relu(prev_state)

        action_w_init = hk.initializers.TruncatedNormal()
        encoded_action = hk.Linear(channels,
                                   w_init=action_w_init,
                                   with_bias=False)(action_onehot)
        x_and_h = prev_state + encoded_action
        out = hk.nets.MLP([channels]*self._num_blocks, with_bias=False)(x_and_h)
        return out


class PredictionMlp(hk.Module):
  def __init__(self,
               mlp_layers,
               num_predictions,
               output_init = None,
               name: str = 'basic_mlp'):
    super().__init__(name=name)
    self._num_predictions = num_predictions
    self._mlp_layers = mlp_layers
    self._output_init = output_init

  def __call__(self, x):
    x = jax.nn.relu(x)
    for l in self._mlp_layers:
      x = hk.Linear(l, with_bias=False)(x)
      x = jax.nn.relu(x)

    return hk.Linear(self._num_predictions, w_init=self._output_init)(x)
