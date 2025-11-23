注意，layer和layers文件不同

layer主要是一个类，其代表模型中的每一层。其里面定义了多种常用层：卷积，线性等

layers是一个创建layer的辅助模块，layer中层先通过layers实现基本功能，再在layer中进行封装

Model是对Layer的进一步封装，通过继承该类来实现模型
使用示例
```python
class TwoLayer(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    def forward(self,x):
        y = relu(self.l1(x))
        y = self.l2(y)
        return y


np.random.seed(42)
x = Variable(np.random.rand(5,10),name='x')
model = TwoLayer(100,10)
model.plot(x)
```