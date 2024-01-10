
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

class ResConvBlock(hk.Module):
    """A residual convolutional block in pre-activation style."""

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        use_projection: bool = False,
        name: str = "res_conv_block",
    ):
        """Init residual block."""
        super().__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Conv2D(
                channels, kernel_shape=3, stride=stride, padding="SAME", with_bias=False
            )
        self._conv_0 = hk.Conv2D(
            channels, kernel_shape=3, stride=stride, padding="SAME", with_bias=False
        )
        self._ln_0 = hk.LayerNorm(
            axis=(-3, -2, -1), create_scale=True, create_offset=True
        )
        self._conv_1 = hk.Conv2D(
            channels, kernel_shape=3, stride=1, padding="SAME", with_bias=False
        )
        self._ln_1 = hk.LayerNorm(
            axis=(-3, -2, -1), create_scale=True, create_offset=True
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

class ResBlocks(hk.Module):
  def __init__(self, num_blocks: int, name="res_mlp"):
    super().__init__(name=name)
    self._num_blocks = num_blocks

  def __call__(self, x):
    res_layers = [
        ResConvBlock(x.shape[-1], use_projection=False)
        for _ in range(self._num_blocks)
    ]
    return hk.Sequential(res_layers)(x)

class VisionTorso(hk.Module):
    """Representation encoding module."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        name: str = "representation",
    ):
        """Init representatioin function."""
        super().__init__(name=name)
        self._channels = channels
        self._num_blocks = num_blocks

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Forward representation function."""
        # 1. Downsampling.
        torso = [
            hk.Conv2D(
                self._channels // 2,
                kernel_shape=3,
                stride=2,
                padding="SAME",
                with_bias=False,
            ),
        ]
        torso.extend(
            [
                ResConvBlock(self._channels // 2, stride=1, use_projection=False)
                for _ in range(1)
            ]
        )
        torso.append(ResConvBlock(self._channels, stride=2, use_projection=True))
        torso.extend(
            [
                ResConvBlock(self._channels, stride=1, use_projection=False)
                for _ in range(1)
            ]
        )
        torso.append(
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")
        )
        torso.extend(
            [
                ResConvBlock(self._channels, stride=1, use_projection=False)
                for _ in range(1)
            ]
        )
        torso.append(
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")
        )

        # 2. Encoding.
        torso.extend(
            [
                ResConvBlock(self._channels, stride=1, use_projection=False)
                for _ in range(self._num_blocks)
            ]
        )
        return hk.Sequential(torso)(observations)

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
          prev_state = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(prev_state)
        prev_state = jax.nn.relu(prev_state)

        action_w_init = hk.initializers.TruncatedNormal()
        encoded_action = hk.Linear(channels,
                                   w_init=action_w_init,
                                   with_bias=False)(action_onehot)

        # [H, W, C] + [1, 1, D], i.e. broadcast across H,W
        x_and_h = prev_state + encoded_action[None, None]
        out = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)(x_and_h)
        out += shortcut  # Residual link to maintain recurrent info flow.

        res_layers = [
            ResConvBlock(
                channels,
                stride=1,
                use_projection=False)
            for _ in range(self._num_blocks)
        ]
        out = hk.Sequential(res_layers)(out)

        return out

class PredictionNet(hk.Module):
  def __init__(self,
               mlp_layers,
               num_predictions,
               w_init: Optional[hk.initializers.Initializer] = None,
               output_init = None,
               name="pred_mlp"):
    super().__init__(name=name)
    self._num_predictions = num_predictions
    self._mlp_layers = mlp_layers
    self._w_init = w_init
    self._output_init = output_init

  def __call__(self, x):
    layers = [
      hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
      jax.nn.relu,
      hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
      hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
      jax.nn.relu,
      hk.Flatten(-3),
    ]
    for l in self._mlp_layers:
       layers.extend([
          hk.Linear(l, with_bias=False),
          hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
          jax.nn.relu,
       ])

    x = hk.Sequential(layers)(x)

    return hk.Linear(self._num_predictions, w_init=self._output_init)(x)