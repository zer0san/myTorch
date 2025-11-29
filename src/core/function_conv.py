from src.core import Variable
from src.cuda import cuda
from src.core.function import Function
from src.cuda import cuda

__all__ = ['conv2d']

def img2col(input_image, kernel_size, stride=1, padding=0):
    xp = cuda.get_array_module(input_image)
    N, C, H, W = input_image.shape
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

    if padding > 0:
        x_padded = xp.pad(input_image, ((0,0),(0,0),(padding,padding),(padding,padding)), 'constant')
    else:
        x_padded = input_image
    H_p, W_p = x_padded.shape[2:]

    out_h = (H_p - kh) // stride + 1
    out_w = (W_p - kw) // stride + 1

    shape = (N, C, out_h, out_w, kh, kw)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2] * stride,
        x_padded.strides[3] * stride,
        x_padded.strides[2],
        x_padded.strides[3],
    )
    windows = xp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides, writeable=False)
    cols = windows.transpose(1,4,5,0,2,3).reshape(C*kh*kw, N*out_h*out_w)

    return cols, (x_padded.shape, out_h, out_w)  # padded_shape！


def col2img(cols, padded_shape, kernel_size, stride=1, padding=0):
    xp = cuda.get_array_module(cols)
    N, C, H_p, W_p = padded_shape
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    out_h = (H_p - kh) // stride + 1
    out_w = (W_p - kw) // stride + 1

    cols = cols.reshape(C, kh, kw, N, out_h, out_w).transpose(3,0,4,5,1,2)
    img = xp.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    for i in range(kh):
        for j in range(kw):
            h_start = i
            h_end = i + out_h * stride
            w_start = j
            w_end = j + out_w * stride
            img[:, :, h_start:h_end:stride, w_start:w_end:stride] += cols[:, :, :, :, i, j]

    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img


class Conv2d(Function):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, bias=True):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.pad = pad
        self.use_bias = bias
        self.W = None
        self.b = None

    def forward(self, x):
        kh, kw = self.kernel_size
        xp = cuda.get_array_module(x)
        N, C, H, W = x.shape

        if self.W is None:
            scale = xp.sqrt(2.0 / (C * kh * kw))
            W_data = xp.random.randn(self.out_channels, C, kh, kw).astype('f') * scale
            self.W = Variable(W_data)  # 必须是 Parameter！
            if self.use_bias:
                self.b = Variable(xp.zeros(self.out_channels, dtype='f'))

        col, cache = img2col(x, self.kernel_size, self.stride, self.pad)
        self.cache = cache  # 保存 padded_shape, out_h, out_w

        W_flat = self.W.data.reshape(self.out_channels, -1)
        out = W_flat @ col

        if self.use_bias:
            out += self.b.data.reshape(-1, 1)

        (padded_shape, out_h, out_w) = cache
        out = out.reshape(self.out_channels, N, out_h, out_w).transpose(1,0,2,3)
        return out

    def backward(self, gy):
        x = self.inputs[0].data
        padded_shape, out_h, out_w = self.cache
        OC = gy.shape[1]

        gy_flat = gy.data.transpose(1,0,2,3).reshape(OC, -1)

        col, _ = img2col(x, self.kernel_size, self.stride, self.pad)
        dW = (gy_flat @ col.T).reshape(self.W.shape)
        self.W.grad = dW

        if self.use_bias:
            self.b.grad = gy_flat.sum(axis=1)

        dcol = self.W.data.reshape(OC, -1).T @ gy_flat
        dx = col2img(dcol, padded_shape, self.kernel_size, self.stride, self.pad)
        return dx


def conv2d(x, out_channels, kernel_size, stride=1, pad=0, bias=True):
    return Conv2d(out_channels, kernel_size, stride, pad, bias)(x)