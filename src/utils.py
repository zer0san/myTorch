import os
import subprocess


# 计算图可视化
# 绘制单个节点
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}" color=gray, style=filled]\n'
    name = '' if v.name is None else v.name
    # verbose 属性用来展示节点详细信息
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    # 使用id函数，指定变量唯一节点
    return dot_var.format(id(v), name)


# 将函数绘制为dot语言
def _dot_func(f):
    dot_func = f'{id(f)} [label="{f.__class__.__name__}", shape=box, color=lightblue, style=filled]\n'

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        dot_func += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        dot_func += dot_edge.format(id(f), id(y()))
    return dot_func


# 绘制计算图
def get_dot_graph(output, verbose=False):
    txt = ''
    funcs = []
    visited = set()

    def add_func(f):
        if f not in visited:
            funcs.append(f)
            visited.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


# output为目标梯度
def plot_dot_graph(output, verbose=False, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)
    # 将dot数据保存至文件
    tmp_dir = os.path.join(os.getcwd(), '.mytorch')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # 调用dot命令
    extension = os.path.splitext(to_file)[1][1:]  # 获取扩展名
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)


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
