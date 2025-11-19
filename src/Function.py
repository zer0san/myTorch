from src.Variable import Variable
import numpy as np

class Function:
    '''
    __call__方法是一个特殊的python方法，定义了这个方法后，
    当f=Function()时，就可以通过编写f(...)来调用__call__方法了
    '''

    def __call__(self, input):
        x = input.data
        self.input = input  # 保存输入的变量
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 让输出变量保留创造者信息
        self.output = output # 保存输出变量
        return output

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
        x = self.input.data
        gx = x * 2 * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 为了方便使用，定义函数
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)