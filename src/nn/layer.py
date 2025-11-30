from .parameter import Parameter
import weakref
import src.nn.layers as layers
import numpy as np
import os
from src.cuda import cuda

__all__ = ['Linear', 'Conv2d']


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

    # 模型的保存和加载
    def save_params(self, file_path):
        param_dict = {}
        for name, param in self.named_params():
            param_dict[name] = param.data
        np.savez(file_path, **param_dict)

    def load_params(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} 不存在")
        npz = np.load(file_path)
        for full_name, data in npz.items():
            parts = full_name.split('.')
            current_layer = self
            for name in parts[:-1]:
                if not hasattr(current_layer, name):
                    setattr(current_layer, name, Layer())
                current_layer = getattr(current_layer, name)
                setattr(current_layer, parts[-1], Parameter(data))


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


class Conv2d(Layer):
    def __init__(self, out_channels, ksize, stride=1, pad=0, bias=False):
        super().__init__()
        self.W = Parameter(None, name='W')
        self.b = Parameter(None, name='b') if bias else None
        self.stride = stride
        self.pad = pad
        self.out_channels = out_channels
        self.ksize = ksize

    def forward(self, x):
        if self.W.data is None:
            C = x.shape[1]
            xp = cuda.get_array_module(x)
            scale = xp.sqrt(2 / (C * self.ksize ** 2))
            self.W.data = xp.random.randn(self.out_channels, C, self.ksize, self.ksize) * scale
            if self.b is not None:
                self.b.data = xp.zeros(self.out_channels)
        return layers.conv2d(x, self.W, self.b, self.stride, self.pad)


class MaxPool2d(Layer):
    def __init__(self, ksize, stride=1, pad=0):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return layers.max_pool2d(x, self.ksize, self.stride, self.pad)


class AvePool2d(Layer):
    def __init__(self, ksize, stride=1, pad=0):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return layers.ave_pool2d(x, self.ksize, self.stride, self.pad)
