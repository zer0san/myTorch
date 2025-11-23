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
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)

