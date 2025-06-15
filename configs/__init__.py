"""
DH-SNN 配置模块
提供不同实验类型的专门配置
"""

from .config_manager import (
    ConfigManager,
    config_manager, 
    get_config,
    get_available_experiments
)

# 便捷导入各种配置
try:
    from . import config as default_config
    from . import delayed_xor_config
    from . import speech_config
    from . import neurovpr_config
    from . import sequential_config
    from . import multiscale_xor_config
except ImportError:
    pass

__all__ = [
    'ConfigManager',
    'config_manager',
    'get_config',
    'get_available_experiments'
]

# 版本信息
__version__ = "1.0.0"