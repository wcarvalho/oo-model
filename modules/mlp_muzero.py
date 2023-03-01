import haiku as hk
import jax
import jax.numpy as jnp


class ResMlpBlock(hk.Module):
    """A residual convolutional block in pre-activation style."""

    def __init__(
        self,
        channels: int,
        use_projection: bool,
        name: str = "res_conv_block",
    ):
        """Init residual block."""
        super().__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Linear(channels, with_bias=False)
        self._conv_0 = hk.Linear(channels, with_bias=False)
        self._ln_0 = hk.LayerNorm(
            axis=(-1), create_scale=True, create_offset=True
        )
        self._conv_1 = hk.Linear(channels, with_bias=False)
        self._ln_1 = hk.LayerNorm(
            axis=(-1), create_scale=True, create_offset=True
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward ResBlock."""
        # NOTE: Using LayerNorm is fine
        # (https://arxiv.org/pdf/2104.06294.pdf Appendix A).
        shortcut = out = x
        out = self._ln_0(out)
        out = jax.nn.relu(out)
        if self._use_projection:
            shortcut = self._proj_conv(out)
        out = hk.Sequential(
            [
                self._conv_0,
                self._ln_1,
                jax.nn.relu,
                self._conv_1,
            ]
        )(out)
        return shortcut + out

class Transition(hk.Module):
    """Dynamics transition module."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        action_dim: int = 32,
        name: str = "transition",
    ):
        """Init transition function."""
        super().__init__(name=name)
        self._channels = channels
        self._num_blocks = num_blocks
        self._action_dim = action_dim

    def __call__(
        self, encoded_action: jnp.ndarray, prev_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward transition function."""
        channels = prev_state.shape[-1]
        shortcut = prev_state

        prev_state = hk.LayerNorm(
            axis=(-1), create_scale=True, create_offset=True
        )(prev_state)
        prev_state = jax.nn.relu(prev_state)

        encoded_action = hk.Linear(self._action_dim, with_bias=False)(encoded_action)
        x_and_h = jnp.concatenate([prev_state, encoded_action], axis=-1)
        out = hk.Linear(channels, with_bias=False)(x_and_h)
        out += shortcut  # Residual link to maintain recurrent info flow.

        res_layers = [
            ResMlpBlock(channels, use_projection=False)
            for _ in range(self._num_blocks)
        ]
        out = hk.Sequential(res_layers)(out)
        return out


class BasicMlp(hk.Module):
  def __init__(self, mlp_layers, num_predictions, output_init: float = 0.0):
    super().__init__(name="basic_mlp")
    self._num_predictions = num_predictions
    self._mlp_layers = mlp_layers
    self._output_init = output_init

  def __call__(self, x):
    output_init = hk.initializers.VarianceScaling(scale=self._output_init)
    x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)
    x = jax.nn.relu(x)
    for l in self._mlp_layers:
      x = hk.Linear(l, with_bias=False)(x)
      x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
      x = jax.nn.relu(x)

    return hk.Linear(self._num_predictions, w_init=output_init)(x)


class ResMlp(hk.Module):
  def __init__(self, num_blocks,):
    super().__init__(name="res_mlp")
    self._num_blocks = num_blocks

  def __call__(self, x):
    res_layers = [
        ResMlpBlock(x.shape[-1], use_projection=False)
        for _ in range(self._num_blocks)
    ]
    return hk.Sequential(res_layers)(x)
