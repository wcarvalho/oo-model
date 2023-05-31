import jax
import haiku as hk


def get_gate_factory(name: str, b_init=None, w_init=None):
  assert name in ('sum', 'gru', 'sigtanh')
  # following options from GTRxL: https://arxiv.org/pdf/1910.06764.pdf

  def sum_gate_factory():
    gate = lambda x,y: x+y
    return hk.to_module(gate)(name='sum_gate')

  def gru_gate_factory():
    def gate(x,y):
      gru = hk.GRU(x.shape[-1])
      out, out = gru(inputs=y, state=x)
      return out
    return hk.to_module(gate)(name='gru_gate')

  def sigtanh_gate_factory():
    def gate(x,y):
      dim = x.shape[-1]
      linear = lambda x: hk.Linear(dim, with_bias=False, w_init=w_init)(x)

      b = hk.get_parameter("b_gate", [dim], x.dtype, b_init)
      gate = jax.nn.sigmoid(linear(y) - b)
      output = jax.nn.tanh(linear(y))
      return x + gate*output
    return hk.to_module(gate)(name='sigtanh_gate')

  if name == 'sum':
    return sum_gate_factory
  elif name == 'gru':
    return gru_gate_factory
  elif name == 'sigtanh':
    return sigtanh_gate_factory
  else:
    raise NotImplementedError("Gate not implemented: {name}")
