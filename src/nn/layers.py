from src.core import Function, sum_to, img2col, col2img, Variable
from src.cuda import cuda

# 线性层
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

class Conv2d(Function):
    def __init__(self, stride, pad):
        super().__init__()
        self.stride = stride
        self.pad = pad

    def forward(self, x, W, b=None):
        kh, kw = W.shape[2], W.shape[3]
        xp = cuda.get_array_module(x)

        col, cache = img2col(x, (kh, kw), self.stride, self.pad)
        self.cache = cache

        # (OC, C*kh*kw) @ (C*kh*kw, N*OH*OW) -> (OC, N*OH*OW)
        out = W.reshape(W.shape[0], -1) @ col

        if b is not None:
            out += b.reshape(-1, 1)

        # reshape to (N, OC, OH, OW)
        padded_shape, out_h, out_w = cache
        out = out.reshape(W.shape[0], x.shape[0], out_h, out_w).transpose(1, 0, 2, 3)

        return out

    def backward(self, gy):
        x, W, b = self.inputs
        padded_shape, out_h, out_w = self.cache

        gy_flat = gy.data.transpose(1, 0, 2, 3).reshape(gy.shape[1], -1)

        # dW
        col, _ = img2col(x.data, W.shape[2:], stride=self.stride, padding=self.pad)
        dW = (gy_flat @ col.T).reshape(W.shape)

        # db
        db = None if b.data is None else gy_flat.sum(axis=1)

        # dx
        dcol = W.data.reshape(W.shape[0], -1).T @ gy_flat
        dx = col2img(dcol, padded_shape, W.shape[2:], stride=self.stride, padding=self.pad)

        return dx, dW, db

def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)

