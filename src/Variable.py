import numpy as np

# 支持反向传播的变量类
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 获取函数
            x, y = f.input, f.output # 获取函数输入输出
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
