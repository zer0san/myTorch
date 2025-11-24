from src.data.datasets import Datasets, BigDatasets
import numpy as np

__all__ = ['DataLoader']


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data_size = len(dataset)

    def __iter__(self):
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))
        self.pos = 0
        return self

    def __next__(self):
        if self.pos >= self.data_size:
            raise StopIteration

        start = self.pos
        end = start + self.batch_size

        if end > self.data_size:
            if self.drop_last:
                raise StopIteration
            end = self.data_size

        batch_index = self.index[start:end]
        self.pos = end

        batch = [self.dataset[i] for i in batch_index]
        # 判断是否有label
        sample = batch[0]
        if isinstance(sample, tuple):
            xs, ys = zip(*batch)
            return np.stack(xs), np.stack(ys) # 使用stack转换为numpy数组
        else:
            return np.stack(batch)
