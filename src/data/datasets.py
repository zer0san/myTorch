import numpy as np

__all__ = ['Datasets']


class Datasets:
    def __init__(self, data, label=None, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.transform = transform or (lambda x: x)
        self.target_transform = target_transform or (lambda x: x)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is None:
            return self.transform(self.data[idx])
        else:
            return self.transform(self.data[idx]), self.target_transform(self.label[idx])


# 专门处理大数据
class BigDatasets(Datasets):
    # 不加载全部数据，只保存索引
    def __init__(self, index_list, loader_func, label_loader_func=None, transform=None, target_transform=None):
        super().__init__(data=index_list, label=None, transform=transform, target_transform=target_transform)
        self.loader_func = loader_func
        self.label_loader_func = label_loader_func

    def __getitem__(self, idx):
        index = self.data[idx]
        x = self.transform(self.loader_func(index))
        if self.label_loader_func is None:
            return x

        y = self.target_transform(self.label_loader_func(index))
        return x, y
