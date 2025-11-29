from src.core.variable import Variable
import numpy as np
import weakref
from src.cuda import cuda
from src.config import Config
import src.utils as utils

__all__ = ['setup_operations', 'as_array', 'as_variable', 'Function', 'matmul', 'sum_to', 'broadcast_to', 'sum',
           'perfume', 'transpose',
           'square', 'exp', 'add', 'mul', 'neg', 'sub', 'div', 'pow', 'sin', 'cos', 'tanh', 'get_item', 'log',
           'accuracy']


# 将numpy的标量转换为ndarray
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 将其它类型变量转换为Variable
def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


class Function:
    '''
    __call__方法是一个特殊的python方法，定义了这个方法后，
    当f=Function()时，就可以通过编写f(...)来调用__call__方法了
    '''

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        # 为了支持可变长参数，将输入、输出改为列表
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 使用星号解包
        if not isinstance(ys, tuple):  # 对非元组情况额外处理
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # 检查是否需要进行反向传播
        if Config.enable_backward:
            self.inputs = inputs
            for output in outputs:  # 保存
                output.set_creator(self)
            # 修改为弱引用，避免循环引用
            self.outputs = [weakref.ref(output) for output in outputs]
        # 如果列表只有一个元素，则返回第一个元素F
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        '''
        抛出异常，告诉使用了Function类的forward的人，这个方法应该通过继承来实现
        '''
        raise NotImplementedError()

    # gy 是链式传播过程中，前一步计算出的导数(链式法则)
    def backward(self, gy):
        raise NotImplementedError()


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
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


class Add(Function):
    def forward(self, x0, x1):
        # 正向传播是基于ndarray，已经实现了广播
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        # 反向传播的广播需要自己实现
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return (y,)

    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        gx0, gx1 = gy, gy
        if self.x1_shape != self.x0_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0 * x1, gx1 * x0


# 相反数
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, -gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0 / x1, gx1 * (-x0 / x1 ** 2)


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
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = cos(x) * gy
        return gx


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = -sin(x) * gy
        return gx


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
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


class Transpose(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.transpose(x)
        return y

    def backward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.transpose(gy)
        return gx


class Perfume(Function):
    def __init__(self, dims):
        self.dims = dims

    def forward(self, x):
        y = x.transpose(self.dims)
        return y

    def backward(self, gy):
        gx = perfume(gy, self.dims)
        return gx


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        # 将梯度恢复为原来的形状
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)  # 这一步后gy.shape不一定等于x_shape
        gx = broadcast_to(gy, self.x_shape)  # 因此进行广播
        return gx


# 广播
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


# 与广播相反
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


# 矩阵乘法
class MatMul(Function):
    def forward(self, x, w):
        y = x.dot(w)
        return y

    def backward(self, gy):
        x, w = self.inputs
        gx = matmul(gy, w.T)
        gw = matmul(x.T, gy)
        return gx, gw


def matmul(x, w):
    return MatMul()(x, w)


# 将形状变为指定形状，多余维度进行求和压缩
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


# 将变量广播到指定形状
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def perfume(x, dims):
    return Perfume(dims)(x)


def transpose(x):
    return Transpose()(x)


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


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape)
        xp.add.at(gx, self.slices, gy)
        return gx

    def backward(self, gy):
        return get_item(gy, self.slices)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.log(x)

    def backward(self, gy):
        x, = self.inputs
        return gy / x


def log(x):
    return Log()(x)


def accuracy(y_pred, y):
    y_pred, y = as_variable(y_pred), as_variable(y)
    t = y_pred.data.argmax(axis=1).reshape(y.shape)
    result = (t == y.data)
    acc = result.mean()
    return Variable(as_array(acc))


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
    Variable.__matmul__ = matmul
    Variable.__getitem__ = get_item
