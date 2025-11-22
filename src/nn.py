import werkzeug.local

from src.Function import *
from src.core import Variable


class Linear(Function):
    def forward(self, x, w, b):
        y = x @ w
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, w, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = gy @ w.T
        gw = x.T @ gy
        return gx, gw, gb


def linear(x, w, b=None):
    return Linear()(x, w, b)
