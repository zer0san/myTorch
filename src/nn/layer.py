from .parameter import Parameter
import weakref
import src.nn.layers as layers
import numpy as np
from src.cuda import get_array_module, as_cupy, as_numpy

__all__ = ['Linear']


class Layer:
    def __init__(self):
        self._layers = {}
        self._params = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Layer):
            self._layers[name] = value
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self):
        raise NotImplementedError()

    def params(self):
        for p in self._params.values():
            yield p
        for l in self._layers.values():
            yield from l.params()

    def named_params(self, prefix=''):
        for name, params in self._params.items():
            yield prefix + name, params
        for lname, layer in self._layers.items():
            new_prefix = prefix + lname + '.'
            yield from layer.named_params(new_prefix)

    def children(self):
        return self._layers.values()

    def cleargrad(self):
        for param in self.params():
            param.cleargrad()
        for layer in self._layers.values():
            layer.cleargrad()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
        for layer in self._layers.values():
            layer.to_gpu()
        return self

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
        for layer in self._layers.values():
            layer.to_cpu()
        return self


# 线性层
class Linear(Layer):
    def __init__(self, out_size, no_bias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()
        if no_bias:
            self.b = None
        else:
            b_data = np.zeros(out_size, dtype=dtype)
            self.b = Parameter(b_data, name='b')

    def _init_W(self):
        W_data = np.random.randn(self.in_size, self.out_size).astype(self.dtype) * np.sqrt(1 / self.in_size)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = layers.linear(x, self.W, self.b)
        return y
