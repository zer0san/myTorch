from .parameter import Parameter
import weakref
import src.nn.layers as layers
import numpy as np
from src.cuda import get_array_module, as_cupy, as_numpy

__all__ = ['Linear']


class Layer:
    def __init__(self):
        self._params_names = set()  # 保存所有参数

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params_names.add(name)
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
        for name in self._params_names:
            obj = getattr(self, name)
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrad(self):
        for param in self.params():
            param.cleargrad()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
        return self

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
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
