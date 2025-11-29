import numpy as np
from collections import defaultdict, deque
from src.core.modelConf import using_config
import src.core.function as function
from src.cuda import get_array_module, as_cupy, as_numpy

__all__ = ['Variable']

try:
    import cupy as cp
    array_types = (np.ndarray, cp.ndarray)
except ImportError:
    array_types = (np.ndarray)


# 支持反向传播的变量类
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            # 检查数据类型
            if not isinstance(data, array_types):
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
        if self.data.ndim == 0:
            return 1
        return len(self.data)

    def max(self):
        return self.data.max()

    # 自定义输出
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        return str(self.data)

    def set_creator(self, func):
        self.creator = func

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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
            # 当create_graph为False时，反向传播中所有中间变量的梯度不会被保留
            # 这只适用于求一阶导数，如果需要求多阶导数，需要将create_graph设置为True
            with using_config('enable_backward', create_graph):
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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return function.reshape(self, shape)

    def transpose(self):
        return function.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return function.sum(self, axis, keepdims)

    @property
    def T(self):
        return function.transpose(self)

    def to_cpu(self):
        if self.data is not None:
            self.data = as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = as_cupy(self.data)