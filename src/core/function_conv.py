from src.cuda import cuda


def img2col(input_image, kernel_size, stride=1, padding=0):
    xp = cuda.get_array_module(input_image)
    # 获取批大小、通道数、图像高度、宽度
    N, C, H, W = input_image.shape
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if padding > 0:
        input_image = xp.pad(input_image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        H, W = input_image.shape[2], input_image.shape[3]

    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    # 使用stride_tricks创建滑动窗口视图
    # shape: (N,C,out_h,out_w,kh,kw)
    strides = (
        input_image.strides[0],  # N
        input_image.strides[1],  # C
        input_image.strides[2] * stride,  # H方向步进
        input_image.strides[3] * stride,  # W方向步进
        input_image.strides[2],  # kh
        input_image.strides[3],  # kw
    )
    windows = xp.lib.stride_tricks.as_strided(input_image, shape=(N, C, out_h, out_w, kh, kw), strides=strides)
    cols = windows.transpose(1, 4, 5, 0, 2, 3)  # (C, kh,kw, N, out_h, out_w)
    cols = cols.reshape(C * kh * kw, N * out_h * out_w)
    return cols
