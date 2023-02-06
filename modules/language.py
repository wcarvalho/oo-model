
import haiku as hk
import jax.numpy as jnp

class LanguageEncoder(hk.Module):
  """Module that embed words and then runs them through GRU. The Token`0` is treated as padding and masked out."""

  def __init__(self,
               vocab_size: int,
               word_dim: int,
               sentence_dim: int,
               compress: str = 'sum'):
    super(LanguageEncoder, self).__init__()
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.sentence_dim = sentence_dim
    self.compress = compress
    self.embedder = hk.Embed(
        vocab_size=vocab_size,
        embed_dim=word_dim)
    self.language_model = hk.GRU(sentence_dim)

  def __call__(self, x: jnp.ndarray):
    """Embed words, then run through GRU.
    
    Args:
        x (TYPE): N
    
    Returns:
        TYPE: Description
    """
    # -----------------------
    # embed words + mask
    # -----------------------
    words = self.embedder(x)  # N x D
    mask = (x > 0).astype(words.dtype)
    words = words*jnp.expand_dims(mask, axis=-1)

    # -----------------------
    # pass through GRU
    # -----------------------
    initial = self.language_model.initial_state(None)
    sentence, _ = hk.static_unroll(self.language_model,
                                   words, initial)

    if self.compress == "last":
      task = sentence[-1]  # embedding at end
    elif self.compress == "sum":
      task = sentence.sum(0)
    else:
      raise NotImplementedError(self.compress)

    return task