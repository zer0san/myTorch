
__all__ = ['sum_to','reshape_sum_backward']

# 与广播相反，压缩多余的维度
def sum_to(x, shape):
    ndim = len(shape)
    # 维度之间差值
    lead = x.ndim - ndim
    # 前lead个维度的索引
    lead_axis = tuple(range(lead))
    # 找到大小为1的维度，进行求和。如果某个维度的大小是1，求和后该维度将被压缩
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


# 将梯度调整为正确的形状
def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]  # -1表示最后一个
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            # 对于每个axis中的维度，向shape中插入大小为1的维度
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy
