from typing import Callable, Sequence, Optional

import haiku as hk
import jax
import jax.numpy as jnp

Array = jnp.ndarray


class Mlp(hk.Module):
  """Copied from: https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/modules/misc.py#L69"""
  def __init__(self,
               mlp_layers,
               output_size: Optional[int] = None,
               layernorm: str = 'none',
               output_init = None,
               residual: bool = False,
               w_init: Optional[hk.initializers.Initializer] = None,
               name="pred_mlp"):
    super().__init__(name=name)
    self._output_size = output_size
    self._mlp_layers = mlp_layers
    self._output_init = output_init
    self._residual = residual
    self._w_init = w_init
    self._layernorm = layernorm
    assert layernorm in ('pre', 'post', 'none')

  def __call__(self, inputs):
    output_size = self._output_size or inputs.shape[-1]

    x = inputs
    if self._layernorm == 'pre':
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)

    for l in self._mlp_layers:
      x = hk.Linear(l, w_init=self._w_init)(x)
      x = jax.nn.relu(x)
    x = hk.Linear(output_size, w_init=self._w_init)(x)

    if self._residual:
      x = x + inputs

    if self._layernorm == 'post':
      x = hk.LayerNorm(axis=(-1), create_scale=True, create_offset=True)(x)

    return x

def create_gradient_grid(
    samples_per_dim: Sequence[int],
    value_range: Sequence[float] = (-1.0, 1.0)
) -> Array:
  """Creates a tensor with equidistant entries from -1 to +1 in each dim.
  Args:
    samples_per_dim: Number of points to have along each dimension.
    value_range: In each dimension, points will go from range[0] to range[1]
  Returns:
    A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
  """
  s = [jnp.linspace(value_range[0], value_range[1], n)
       for n in samples_per_dim]
  pe = jnp.stack(jnp.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
  return jnp.array(pe)


def convert_to_fourier_features(inputs: Array, basis_degree: int) -> Array:
  """Convert inputs to Fourier features, e.g. for positional encoding."""

  # inputs.shape = (..., n_dims).
  # inputs should be in range [-pi, pi] or [0, 2pi].
  n_dims = inputs.shape[-1]

  # Generate frequency basis.
  freq_basis = jnp.concatenate(  # shape = (n_dims, n_dims * basis_degree)
      [2**i * jnp.eye(n_dims) for i in range(basis_degree)], 1)

  # x.shape = (..., n_dims * basis_degree)
  x = inputs @ freq_basis  # Project inputs onto frequency basis.

  # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
  return jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))



class PositionEmbedding(hk.Module):
  """A module for applying N-dimensional position embedding.
  Attr:
    embedding_type: A string defining the type of position embedding to use. One
      of ["linear", "discrete_1d", "fourier", "gaussian_fourier"].
    update_type: A string defining how the input is updated with the position
      embedding. One of ["proj_add", "concat"].
    num_fourier_bases: The number of Fourier bases to use. For embedding_type ==
      "fourier", the embedding dimensionality is 2 x number of position
      dimensions x num_fourier_bases. For embedding_type == "gaussian_fourier",
      the embedding dimensionality is 2 x num_fourier_bases. For embedding_type
      == "linear", this parameter is ignored.
    gaussian_sigma: Standard deviation of sampled Gaussians.
    pos_transform: Optional transform for the embedding.
    output_transform: Optional transform for the combined input and embedding.
    trainable_pos_embedding: Boolean flag for allowing gradients to flow into
      the position embedding, so that the optimizer can update it.
  """
  def __init__(self,
               embedding_type: str,
               update_type: str,
               num_fourier_bases: int = 0,
               gaussian_sigma: float = 1.0,
               pos_transform: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
               output_transform: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
               trainable_pos_embedding: bool = False,
               w_init: Optional[hk.initializers.Initializer] = None,
               name="pos_embed",
               ):
    super().__init__(name=name)
    self.embedding_type = embedding_type
    self.update_type = update_type
    self.num_fourier_bases = num_fourier_bases
    self.gaussian_sigma = gaussian_sigma
    self.pos_transform = pos_transform
    self.output_transform = output_transform
    self.trainable_pos_embedding = trainable_pos_embedding
    self.w_init = w_init

  def _make_pos_embedding_tensor(self, rng, input_shape):
    has_batch_dim = len(input_shape) == 4
    idx = int(has_batch_dim)
    if self.embedding_type == "discrete_1d":
      # An integer tensor in [0, input_shape[-2]-1] reflecting
      # 1D discrete position encoding (encode the second-to-last axis).
      pos_embedding = jnp.broadcast_to(
          jnp.arange(input_shape[-2]), input_shape[idx:-1])
    else:
      # A tensor grid in [-1, +1] for each input dimension.
      # DEFAULT
      pos_embedding = create_gradient_grid(
          input_shape[idx:-1], [-1.0, 1.0])

    if self.embedding_type == "linear":  # DEFAULT
      pass
    elif self.embedding_type == "discrete_1d":
      pos_embedding = jax.nn.one_hot(pos_embedding, input_shape[-2])
    elif self.embedding_type == "fourier":
      # NeRF-style Fourier/sinusoidal position encoding.
      pos_embedding = convert_to_fourier_features(
          pos_embedding * jnp.pi, basis_degree=self.num_fourier_bases)
    elif self.embedding_type == "gaussian_fourier":
      # Gaussian Fourier features. Reference: https://arxiv.org/abs/2006.10739
      num_dims = pos_embedding.shape[-1]
      projection = jax.random.normal(
          rng, [num_dims, self.num_fourier_bases]) * self.gaussian_sigma
      pos_embedding = jnp.pi * pos_embedding.dot(projection)
      # A slightly faster implementation of sin and cos.
      pos_embedding = jnp.sin(
          jnp.concatenate([pos_embedding, pos_embedding + 0.5 * jnp.pi],
                          axis=-1))
    else:
      raise ValueError("Invalid embedding type provided.")

    # Add batch dimension.
    if has_batch_dim:
      pos_embedding = jnp.expand_dims(pos_embedding, axis=0)
      print("check this")
      import ipdb; ipdb.set_trace()

    return pos_embedding

  def __call__(self, inputs: Array) -> Array:

    # Compute the position embedding only in the initial call use the same rng
    # as is used for initializing learnable parameters.
    # [H, W, 2]
    pos_embedding = self._make_pos_embedding_tensor(hk.next_rng_key(), inputs.shape)

    if not self.trainable_pos_embedding:
      pos_embedding = jax.lax.stop_gradient(pos_embedding)

    # Apply optional transformation on the position embedding.
    pos_embedding = self.pos_transform(
        pos_embedding)  # pytype: disable=not-callable

    # Apply position encoding to inputs.
    if self.update_type == "project_add":
      # Here, we project the position encodings to the same dimensionality as
      # the inputs and add them to the inputs (broadcast along batch dimension).
      # This is roughly equivalent to concatenation of position encodings to the
      # inputs (if followed by a Dense layer), but is slightly more efficient.
      n_features = inputs.shape[-1]
      x = inputs + hk.Linear(n_features,
                             w_init=self.w_init,
                             name="dense_pe_0")(pos_embedding)
    elif self.update_type == "concat":
      # Repeat the position embedding along the first (batch) dimension.
      pos_embedding = jnp.broadcast_to(
          pos_embedding, shape=inputs.shape[:-1] + pos_embedding.shape[-1:])
      # concatenate along the channel dimension.
      x = jnp.concatenate((inputs, pos_embedding), axis=-1)
    else:
      raise ValueError("Invalid update type provided.")

    # Apply optional output transformation.
    x = self.output_transform(x)  # pytype: disable=not-callable
    return x

class PositionEncodingTorso(hk.Module):
  def __init__(self,
               img_embed,
               pos_embed: PositionEmbedding,
               name="pos_embed_torso"):
    super().__init__(name=name)
    self._img_embed = img_embed
    self._pos_embed = pos_embed

  def __call__(self, image):
    image = self._img_embed(image)
    image = self._pos_embed(image)
    height, width, n_features = image.shape
    image = jnp.reshape(image, (height * width, n_features))  # [H*W, D]
    return image

