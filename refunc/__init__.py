"""
REFUNC - A comprehensive ML utilities toolkit.

This package provides:
- Exception handling framework
- Performance decorators and monitoring
- Advanced logging system for ML workflows
- Configuration management
- Mathematical and statistical utilities
- Utility functions
"""

# Import main modules
from . import exceptions
from . import decorators
from . import logging
from . import config
from . import math_stats
from . import utils

# Version information
__version__ = "0.1.0"
__author__ = "kennedym-ds"
__description__ = "A comprehensive ML utilities toolkit"

__all__ = [
    'exceptions',
    'decorators',
    'logging',
    'config',
    'math_stats',
    'utils'
]