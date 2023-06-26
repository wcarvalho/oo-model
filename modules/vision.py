"""Vision Modules."""

import functools

from typing import Optional, Sequence, Callable, Tuple, Dict
import haiku as hk
import jax
import jax.numpy as jnp

Images = jnp.ndarray

class AtariVisionTorso(hk.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, flatten=True, conv_dim=16, out_dim=0):
    super().__init__(name='atari_torso')
    layers = [
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
    ]
    if conv_dim:
      layers.append(hk.Conv2D(conv_dim, [1, 1], 1))
    self._network = hk.Sequential(layers)

    self.flatten = flatten
    if out_dim:
      self.out_net = hk.Linear(out_dim)
    else:
      self.out_net = lambda x: x

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return self.out_net(flat)


class ResidualBlock(hk.Module):
  """Residual block."""

  def __init__(self, num_channels, name=None):
    super().__init__(name=name)
    self._num_channels = num_channels

  def __call__(self, x):
    main_branch = hk.Sequential([
        jax.nn.relu,
        hk.Conv2D(
            self._num_channels,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding='SAME'),
        jax.nn.relu,
        hk.Conv2D(
            self._num_channels,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding='SAME'),
    ])
    return main_branch(x) + x


class AtariImpalaTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper.
  Note: for 64 x 64 --> 8 x 8
  """

  def __init__(self, flatten=True, conv_dim=16, out_dim=256, name=None):
    super().__init__(name=name)
    self.flatten = flatten
    self.conv_dim = conv_dim
    self.out_dim = out_dim

  def __call__(self, inputs: jnp.ndarray):
    """Summary
    
    Args:
        inputs (jnp.ndarray): B x H x W x C
    
    Returns:
        TYPE: Description
    
    Raises:
        ValueError: Description
    """
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    torso_out = inputs
    for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
      conv = hk.Conv2D(
          num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME')
      torso_out = conv(torso_out)
      torso_out = hk.max_pool(
          torso_out,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
      )
      for j in range(num_blocks):
        block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j))
        torso_out = block(torso_out)

    torso_out = jax.nn.relu(torso_out)
    if self.flatten:
      torso_out = hk.Flatten()(torso_out)
      if self.out_dim:
        torso_out = hk.Linear(self.out_dim)(torso_out)
        torso_out = jax.nn.relu(torso_out)

    return torso_out


class BabyAIVisionTorso(hk.Module):
  """Convolutional stack used in BabyAI codebase."""

  def __init__(self, flatten=False, conv_dim=16, out_dim=0):
    super().__init__(name='babyai_torso')
    layers = [
        hk.Conv2D(128, [8, 8], stride=8),
        hk.Conv2D(128, [3, 3], stride=1),
        jax.nn.relu,
        hk.Conv2D(128, [3, 3], stride=1),
        jax.nn.relu,
    ]
    if conv_dim > 0:
      layers.append(hk.Conv2D(conv_dim, [1, 1], stride=1))
    self._network = hk.Sequential(layers)

    self.flatten = flatten
    if out_dim:
      self.out_net = hk.Linear(out_dim)
    else:
      self.out_net = lambda x: x

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return self.out_net(flat)


class SaviVisionTorso(hk.Module):
  """Flexible CNN model with conv. and normalization layers."""

  def __init__(
      self,
      features: Sequence[int],
      kernel_size: Sequence[Tuple[int, int]],
      strides: Sequence[Tuple[int, int]],
      layer_transpose: Sequence[bool],
      activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      w_init = None,
      norm_type: Optional[str] = None,
      axis_name: Optional[str] = None,  # Over which axis to aggregate batch stats.
      output_size: Optional[int] = None,
      name: str = None):
    super().__init__(name)
    self.features = features
    self.kernel_size = kernel_size
    self.strides = strides
    self.layer_transpose = layer_transpose
    self.activation_fn = activation_fn
    self.norm_type = norm_type
    self.axis_name = axis_name
    self.output_size = output_size
    self.w_init = w_init

  def __call__(
          self, inputs: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    num_layers = len(self.features)

    assert num_layers >= 1, "Need to have at least one layer."
    assert len(self.kernel_size) == num_layers, (
        "len(kernel_size) and len(features) must match.")
    assert len(self.strides) == num_layers, (
        "len(strides) and len(features) must match.")
    assert len(self.layer_transpose) == num_layers, (
        "len(layer_transpose) and len(features) must match.")

    if self.norm_type:
      assert self.norm_type in {"batch", "group", "instance", "layer"}, (
          f"{self.norm_type} is not a valid normalization module.")

    # Whether transpose conv or regular conv.
    conv_module = {False: hk.Conv2D, True: hk.Conv2DTranspose}

    if self.norm_type == "batch":
      raise NotImplementedError("need to convert flax to haiku")
      # norm_module = functools.partial(
      #     nn.BatchNorm, momentum=0.9, use_running_average=not train,
      #     axis_name=self.axis_name)
    elif self.norm_type == "group":
      norm_module = functools.partial(hk.GroupNorm, groups=32)
      raise NotImplementedError
    elif self.norm_type == "layer":
      norm_module = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)
      raise NotImplementedError

    x = inputs
    for i in range(num_layers):
      x = conv_module[self.layer_transpose[i]](
          name=f"conv_{i}",
          output_channels=self.features[i],
          kernel_shape=self.kernel_size[i],
          stride=self.strides[i],
          w_init=self.w_init,
          with_bias=False if self.norm_type else True)(x)

      # Normalization layer.
      if self.norm_type:
        if self.norm_type == "instance":
          x = hk.GroupNorm(
              groups=self.features[i],
              name=f"{self.norm_type}_norm_{i}")(x)
        else:
          norm_module(name=f"{self.norm_type}_norm_{i}")(x)

      # Activation layer.
      x = self.activation_fn(x)

    # Final dense layer.
    if self.output_size:
      x = hk.Linear(self.output_size, name="output_layer", use_bias=True)(x)
    return x

SmallSaviVisionTorso = functools.partial(
  SaviVisionTorso,
  features=[32, 32, 32, 32],
  kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
  strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
  layer_transpose=[False, False, False, False],

)