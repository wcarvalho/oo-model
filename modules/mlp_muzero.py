from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp

from muzero.utils import scale_gradient

class ResMlpBlock(hk.Module):
    """A residual convolutional block in pre-activation style."""

    def __init__(
        self,
        channels: int,
        use_projection: bool,
        ln: bool = True,
        gate = lambda x, y: x+y,
        name: str = "res_conv_block",
    ):
        """Init residual block."""
        super().__init__(name=name)
        self._use_projection = use_projection
        self._gate = gate
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
        if not ln:
           self._ln_0 = lambda o: o
           self._ln_1 = lambda o: o

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
        return self._gate(shortcut, out)


class Transition(hk.Module):
    """Dynamics transition module."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        ln: bool = True,
        name: str = "transition",
    ):
        """Init transition function."""
        super().__init__(name=name)
        self._channels = channels
        self._num_blocks = num_blocks
        self._ln = ln

    def __call__(
        self,
        action_onehot: jnp.ndarray,
        prev_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward transition function."""
        channels = prev_state.shape[-1]
        shortcut = prev_state
        
        if self._ln:
          prev_state = hk.LayerNorm(
              axis=(-1), create_scale=True, create_offset=True
          )(prev_state)
        prev_state = jax.nn.relu(prev_state)

        action_w_init = hk.initializers.TruncatedNormal()
        encoded_action = hk.Linear(channels,
                                   w_init=action_w_init,
                                   with_bias=False)(action_onehot)
        x_and_h = prev_state + encoded_action
        out = hk.Linear(channels, with_bias=False)(x_and_h)
        out += shortcut  # Residual link to maintain recurrent info flow.

        res_layers = [
            ResMlpBlock(channels, ln=self._ln, use_projection=False)
            for _ in range(self._num_blocks)
        ]
        out = hk.Sequential(res_layers)(out)

        return out


class PredictionMlp(hk.Module):
  def __init__(self,
               mlp_layers,
               num_predictions,
               ln: bool = True,
               w_init: Optional[hk.initializers.Initializer] = None,
               output_init = None,
               name="pred_mlp"):
    super().__init__(name=name)
    self._num_predictions = num_predictions
    self._mlp_layers = mlp_layers
    self._w_init = w_init
    self._output_init = output_init
    self._ln = ln

  def __call__(self, x):
    if self._ln:
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)
    x = jax.nn.relu(x)
    for l in self._mlp_layers:
      x = hk.Linear(l, w_init=self._w_init, with_bias=False)(x)
      if self._ln:
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
      x = jax.nn.relu(x)

    return hk.Linear(self._num_predictions, w_init=self._output_init)(x)


class ResMlp(hk.Module):
  def __init__(self, num_blocks, ln: bool = True, name="res_mlp"):
    super().__init__(name=name)
    self._num_blocks = num_blocks
    self._ln = ln

  def __call__(self, x):
    res_layers = [
        ResMlpBlock(x.shape[-1], ln=self._ln, use_projection=False)
        for _ in range(self._num_blocks)
    ]
    return hk.Sequential(res_layers)(x)
