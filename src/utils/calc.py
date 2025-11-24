import numpy as np

__all__ = ['logsumexp']

def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    s = np.exp(y).sum(axis=axis, keepdims=True)
    s = np.log(s)
    return m + s
