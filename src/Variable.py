import numpy as np


# 支持反向传播的变量类
class Variable:
    def __init__(self, data):
        if data is not None:
            # 检查数据类型
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 构建拓扑排序
        topo_funcs = []
        visited = set()

        def build_topo(var):
            if var.creator is None:
                return
            func = var.creator
            if func not in visited:
                visited.add(func)
                for x in func.inputs:
                    build_topo(x)
                topo_funcs.append(func)

        build_topo(self)

        # 从后向前，进行反向传播
        for f in reversed(topo_funcs):
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
