from .parameter import Parameter

__all__ = ['Layer']

class Layer:
    def __init__(self):
        self._params = set()  # 保存所有参数

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

