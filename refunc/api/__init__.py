"""
REST API module for refunc.

This module provides REST endpoints for all refunc functionality including:
- Data science operations (validation, cleaning, profiling)
- Mathematical and statistical computations
- ML operations (evaluation, features, models)
- Utility operations (file handling, caching)
"""

from .main import app

__all__ = ['app']