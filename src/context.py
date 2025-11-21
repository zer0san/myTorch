# 此文件中的函数，用于修改Config中的属性
import contextlib
from src.Config import Config

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value) # 更新状态
    try:
        yield
    finally:
        setattr(Config, name, old_value) # 恢复状态

# 定义no_grad()函数，方便调用
def no_grad():
    return using_config('enable_backward', False)