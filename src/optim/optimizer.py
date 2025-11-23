
__all__ = ['SGD']

# 所有优化类的基类
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # 将None之外的参数汇总到列表
        params = [p for p in self.target.params() if p.grad is not None]

        # 预处理(可选)
        for f in self.hooks:
            f(params)

        # 更新参数
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data