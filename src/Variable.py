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

        # 构建拓扑排序,使用循环代替后序DFS
        topo_funcs = []
        stack = [self]
        # 防止一个节点被dfs多次访问，导致梯度累加错误
        visited = set()
        # 使用集合优化性能
        added = set()

        while stack:
            var = stack[-1]
            if var.creator is None or var.creator in visited:
                # 如果没有creator或creator已经处理过，出栈并加入 topo
                stack.pop()
                if var.creator is not None and var.creator not in added:
                    topo_funcs.append(var.creator)
                    added.add(var.creator)
            else:
                func = var.creator
                visited.add(func)
                for x in reversed(func.inputs):
                    if x.creator is not None and x.creator not in visited:
                        stack.append(x)

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
