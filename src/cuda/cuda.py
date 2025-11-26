import numpy as np

__all__= ['get_array_module', 'as_numpy', 'as_cupy']

gpu_enable = True

try:
    import cupy as cp
except ImportError:
    gpu_enable = False

def is_variable(x):
    from src.core import Variable
    return isinstance(x, Variable)

def get_array_module(x):
    if is_variable(x):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if is_variable(x):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if is_variable(x):
        x = x.data

    if not gpu_enable:
        raise Exception('Cupy cannot be loaded.')

    if cp.isscalar(x):
        return cp.array(x)
    elif isinstance(x, cp.ndarray):
        return x
    return cp.asarray(x)
