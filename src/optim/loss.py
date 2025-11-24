# 损失函数类
from src.core import Function
from src.utils import logsumexp
import numpy as np
from src.nn.activation import softmax

__all__ = ['MSE', 'softmax_cross_entropy']


# 均方误差
class MeanSquaredError(Function):
    def forward(self, y_pred, y):
        return ((y_pred - y) ** 2).sum() / len(y)

    def backward(self, gy):
        y_pred, y = self.inputs
        diff = y_pred - y
        gy0 = gy * diff / len(y) * 2
        gy1 = -gy0
        return gy0, gy1


def MSE(y_pred, y):
    return MeanSquaredError()(y_pred, y)


# 交叉熵误差
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        # 取批量大小
        N = x.shape[0]
        # 计算log(sum(exp(x)))，直接计算sum(exp(x))可能会溢出
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N = x.shape[0]
        y = softmax(x)

        y[np.arange(N), t.data] -= 1
        gy /= N
        return gy * y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
