import numpy as np

__all__ = ['Datasets']


class Datasets:
    def __init__(self, data, label=None, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.transform = transform or (lambda x:x)
        self.target_transform = target_transform or (lambda x:x)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is None:
            return self.transform(self.data[idx])
        else:
            return self.transform(self.data[idx]), self.target_transform(self.label[idx])
