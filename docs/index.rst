🚀 Refunc - ML Utilities Toolkit
===================================

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.7+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

A comprehensive, production-ready ML utilities toolkit designed to accelerate machine learning development with robust, reusable components and professional development practices built-in.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/installation
   guides/quickstart
   guides/configuration
   guides/performance
   guides/logging
   guides/error_handling
   api/index
   examples/index
   developer/contributing

Quick Start
-----------

Install Refunc via pip:

.. code-block:: bash

   pip install refunc

Basic usage:

.. code-block:: python

   from refunc import MLLogger, time_it, memory_profile
   from refunc.exceptions import retry_on_failure

   # Performance monitoring
   @time_it
   @memory_profile
   def train_model(data):
       # Your ML training code here
       pass

   # Robust error handling
   @retry_on_failure(max_attempts=3)
   def api_call():
       # Your API call code here
       pass

Features
--------

🏗️ **Core Architecture**
   - Modular Design: Independent utilities that work together seamlessly
   - Type Safety: Comprehensive type hints throughout for better IDE support
   - Cross-platform: Works on Windows, macOS, and Linux

📊 **ML & Data Science**
   - Advanced logging with experiment tracking
   - Statistical utilities and data analysis tools
   - Performance monitoring decorators

🔧 **Development Tools**
   - Configuration management system
   - Robust exception handling framework
   - File handling utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`