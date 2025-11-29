from src.utils.dot_plot import plot_dot_graph
import src.nn.layer as L
from src.nn.activation import relu, sigmoid

__all__ = ['Model','MLP']


# model类需要其它类继承实现
class Model(L.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)

# 多层感知机实现
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=relu):
        '''
        :param fc_output_sizes: 指定每层的输出数
        :param activation: 指定激活函数，默认为relu
        '''
        super().__init__()
        self.activation = activation
        # 这里只存储层的名字，具体的内容在层对应的对象中
        self._layer_names = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            name = f'l{i}'
            setattr(self, name, layer)
            self._layer_names.append(name)

    def forward(self, x):
        for i,name in enumerate(self._layer_names):
            layer = getattr(self, name)
            if i < len(self._layer_names):
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return x

