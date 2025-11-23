# 损失函数类
from src.core import Function

__all__ = ['MSE']

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
