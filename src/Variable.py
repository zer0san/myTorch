
# 支持反向传播的变量类
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None



