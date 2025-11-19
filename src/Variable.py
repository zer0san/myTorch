
# 支持反向传播的变量类
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # 获取函数
        if f is not None:
            x = f.input # 获取函数输入
            x.grad = f.backward(self.grad) # 调用函数的backward方法
            x.backward() # 递归