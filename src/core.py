from collections import defaultdict
from collections import deque
import weakref
from src.Config import Config
import contextlib
import numpy as np

# 关闭反向传播
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value) # 更新状态
    try:
        yield
    finally:
        setattr(Config, name, old_value) # 恢复状态

# 定义no_grad()函数，方便调用
def no_grad():
    return using_config('enable_backward', False)

# 将numpy的标量转换为ndarray
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 将其它类型变量转换为Variable
def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)

# 支持反向传播的变量类
class Variable:
    __array_priority__ = 200
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


class Function:
    '''
    __call__方法是一个特殊的python方法，定义了这个方法后，
    当f=Function()时，就可以通过编写f(...)来调用__call__方法了
    '''

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        # 为了支持可变长参数，将输入、输出改为列表
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 使用星号解包
        if not isinstance(ys, tuple):  # 对非元组情况额外处理
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # 检查是否需要进行反向传播
        if Config.enable_backward:
            self.inputs = inputs
            for output in outputs:  # 保存
                output.set_creator(self)
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
