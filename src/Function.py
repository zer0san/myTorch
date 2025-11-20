from src.Variable import Variable
import numpy as np
import weakref


# 将numpy的标量转换为ndarray
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    '''
    __call__方法是一个特殊的python方法，定义了这个方法后，
    当f=Function()时，就可以通过编写f(...)来调用__call__方法了
    '''

    def __call__(self, *inputs):
        # 为了支持可变长参数，将输入、输出改为列表
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 使用星号解包
        if not isinstance(ys, tuple):  # 对非元组情况额外处理
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:  # 保存
            output.set_creator(self)
        self.inputs = inputs
        # 修改为弱引用，避免循环引用
        self.outputs = [weakref.ref(output) for output in outputs]
        # 如果列表只有一个元素，则返回第一个元素
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
        x = self.inputs[0].data
        gx = x * 2 * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


# 为了方便使用，定义函数
def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)
