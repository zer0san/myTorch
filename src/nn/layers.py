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


class MaxPool2d(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2
        self.kh, self.kw = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        xp = cuda.get_array_module(x)
        N, C, H, W = x.shape
        kh, kw = self.kh, self.kw
        sy, sx = self.stride, self.stride
        ph, pw = self.pad, self.pad

        x_pad = xp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else x
        Hp, Wp = x_pad.shape[2:]

        out_h = (Hp - kh) // sy + 1
        out_w = (Wp - kw) // sx + 1

        col = img2col(x_pad, (kh, kw), sy, 0)[0]  # 只取 cols
        col = col.reshape(N, C, out_h, out_w, kh * kw)

        # 按通道独立找最大值
        self.max_idx = xp.argmax(col, axis=-1)  # (N,C,OH,OW)
        out = xp.max(col, axis=-1)

        self.cache = (x.shape, (N, C, Hp, Wp), out_h, out_w, col.shape)
        return out

    def backward(self, gy):
        xp = cuda.get_array_module(gy)
        x_shape, pad_shape, out_h, out_w, col_shape = self.cache
        N, C, Hp, Wp = pad_shape

        dcol = xp.zeros(col_shape, dtype=gy.data.dtype)
        dcol[xp.arange(N)[:, None, None, None],
        xp.arange(C)[None, :, None, None],
        xp.arange(out_h)[None, None, :, None],
        xp.arange(out_w)[None, None, None, :],
        self.max_idx] = gy.data

        dx = col2img(dcol.reshape(-1, self.kh * self.kw), pad_shape, (self.kh, self.kw), self.stride, 0)
        if self.pad > 0:
            dx = dx[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dx

def max_pool2d(x,ksize,stride=1,pad=0):
    return MaxPool2d(ksize,stride,pad)(x)