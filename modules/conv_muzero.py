
import haiku as hk
import jax
import jax.numpy as jnp

class ResConvBlock(hk.Module):
    """A residual convolutional block in pre-activation style."""

    def __init__(
        self,
        channels: int,
        stride: int,
        use_projection: bool,
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
