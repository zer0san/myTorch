# 在此文件中进行运算符重载，避免循环引用

from src.core import Variable
from src.Operations import add, mul

# Variable.__add__ = lambda self, other: add(self, other)
# Variable.__mul__ = lambda self, other: mul(self, other)

# 增加rmul和radd，防止左边不为Variable类型
Variable.__add__ = add
Variable.__mul__ = mul
Variable.__radd__ = add
Variable.__rmul__ = mul
