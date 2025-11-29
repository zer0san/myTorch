from src.core import as_variable
from src.cuda import cuda
from src.config import Config

__all__ = ['dropout']

def dropout(x, dropout=0.5):
    x = as_variable(x)
    if Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout
        y = x * mask / (1.0 - dropout)
        return y
    else:
        return x