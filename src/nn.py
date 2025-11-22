from src.Function import *


# 线性层
class Linear(Function):
    def forward(self, x, w, b):
        y = x @ w
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, w, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = gy @ w.T
        gw = x.T @ gy
        return gx, gw, gb


def linear(x, w, b=None):
    return Linear()(x, w, b)


# 激活函数Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


# ReLU
class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)
