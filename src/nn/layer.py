from .parameter import Parameter
import weakref
import src.nn.layers as layers
import numpy as np

__all__ = ['Linear']

class Layer:
    def __init__(self):
        self._params = set()  # 保存所有参数

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
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
        for name in self._params:
            yield self.__dict__[name]

    def cleargrad(self):
        for param in self.params():
            param.cleargrad()


# 线性层
class Linear(Layer):
    def __init__(self, in_size, out_size, no_bias=False, dtype=np.float32):
        super().__init__()
        W_data = np.random.randn(in_size, out_size).astype(dtype) * np.sqrt(1 / in_size)
        self.W = Parameter(W_data, name='W')
        if no_bias:
            self.b = None
        else:
            b_data = np.zeros(out_size, dtype=dtype)
            self.b = Parameter(b_data, name='b')

    def forward(self, x):
        y = layers.linear(x, self.W, self.b)
        return y

