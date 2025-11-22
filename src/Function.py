from src.core import as_array, Variable, as_variable
import numpy as np
import weakref
from src.Config import Config

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

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    def backward(self, gy):
        gx = transpose(gy)
        return gx

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
