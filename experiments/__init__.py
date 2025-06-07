"""
Experiments package with import compatibility layer
"""

# Legacy compatibility imports
try:
    from .legacy_spikingjelly import *
except ImportError:
    pass

# Common module shortcuts
from .common import metrics, trainer, utils, visualization

# Config shortcuts
from .configs import base_config

__all__ = [
    'metrics', 'trainer', 'utils', 'visualization',
    'base_config'
]
