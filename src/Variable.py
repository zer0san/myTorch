from collections import defaultdict
from collections import deque
import src.Oprations as Operations
import numpy as np


# 支持反向传播的变量类
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            # 检查数据类型
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None

    # 获取形状
    @property
    def shape(self):
        return self.data.shape

    # 获取维度
    @property
    def ndim(self):
        return self.data.ndim

    # 获取大小
    @property
    def size(self):
        return self.data.size

    # 获取数据类型
    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    # 自定义输出
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        return str(self.data)

    # 重载运算符
    def __mul__(self, other):
        return Operations.mul(self, other)

    def __add__(self, other):
        return Operations.add(self, other)

    def set_creator(self, func):
        self.creator = func

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 使用依赖计数
        deps_count = defaultdict(int)
        funcs_to_visit = deque([self.creator] if self.creator is not None else [])
        visited = set(funcs_to_visit)
        while funcs_to_visit:
            func = funcs_to_visit.popleft()
            for x in func.inputs:
                if x.creator is not None:
                    deps_count[x.creator] += 1
                    if x.creator not in visited:
                        funcs_to_visit.append(x.creator)
                        visited.add(x.creator)

        ready_queue = deque([f for f in visited if deps_count[f] == 0])

        while ready_queue:
            func = ready_queue.popleft()
            # 获取弱引用数值
            gys = [output().grad for output in func.outputs]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # 更新前驱依赖计数
                if x.creator is not None:
                    deps_count[x.creator] -= 1
                    if deps_count[x.creator] == 0:
                        ready_queue.append(x.creator)

            # 删除除了终端变量之外的所有变量的导数
            if not retain_grad:
                for output in func.outputs:
                    output().grad = None
