from numpy.ma.core import shape

from src.core import Function, as_array, Variable, as_variable
import numpy as np


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = x * 2 * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return (y,)

    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        return gy * x1, gy * x0


# 相反数
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        return gy / x1, gy * (-x0 / x1 ** 2)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = cos(x) * gy
        return gx


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = -sin(x) * gy
        return gx


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = (1 - y ** 2) * gy
        return gx


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    # 前向传播修改形状
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    # 反向传播修改梯度形状
    def backward(self, gy):
        gx = reshape(gy, self.x_shape)
        return gx


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# 为了方便使用，定义函数
def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def setup_operations():
    Variable.__add__ = add
    Variable.__mul__ = mul
    Variable.__radd__ = add
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
