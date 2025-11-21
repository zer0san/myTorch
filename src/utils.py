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
