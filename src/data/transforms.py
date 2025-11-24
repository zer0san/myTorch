import numpy as np

__all__ = ['Normalize','Flatten','AsType','Compose']

class Compose:
    def __init__(self, transforms = []):
        self.transforms = transforms

    def __call__(self, x):
        if not self.transforms:
            return x
        for t in self.transforms:
            x = t(x)
        return x

class Normalize:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std
        return (array - mean) / std

class Flatten:
    def __call__(self, array):
        return array.flatten()

class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)

