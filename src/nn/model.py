from src.utils.dot_plot import plot_dot_graph
from src.nn.layer import Layer

__all__=['Model']

# model类需要其它类继承实现
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)