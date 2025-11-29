import contextlib
from src.config import Config

__all__ = ['using_config', 'no_grad', 'test_mode']

# 关闭反向传播
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)  # 更新状态
    try:
        yield
    finally:
        setattr(Config, name, old_value)  # 恢复状态


# 定义no_grad()函数，方便调用
def no_grad():
    return using_config('enable_backward', False)

# 切换到测试模式
def test_mode():
    return using_config('train',False)
