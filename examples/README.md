# 💡 Refunc Examples

This directory contains practical usage examples for all Refunc modules. Each example is self-contained and demonstrates real-world usage patterns.

## 📁 Directory Structure

### [Basic Usage](basic_usage/)
Simple examples for getting started with each core module:
- [File Handling](basic_usage/file_handling.py) - FileHandler, caching, and format detection
- [Logging Setup](basic_usage/logging_setup.py) - MLLogger and experiment tracking basics
- [Error Handling](basic_usage/error_handling.py) - Exception handling and retry mechanisms

### [Decorators](decorators/)
Performance monitoring and validation examples:
- [Performance Monitoring](decorators/performance_monitoring.py) - Timing and memory profiling
- [Caching Examples](decorators/caching_examples.py) - Result caching strategies
- [Validation Examples](decorators/validation_examples.py) - Input/output validation

### [Data Science](data_science/)
Data analysis and preprocessing examples:
- [Data Validation](data_science/data_validation.py) - Quality assessment and schema validation
- [Preprocessing Pipeline](data_science/preprocessing_pipeline.py) - Data transformation workflows
- [Statistical Analysis](data_science/statistical_analysis.py) - Math/stats utilities usage

### [ML Workflows](ml_workflows/)
Machine learning pipeline examples:
- [Model Training](ml_workflows/model_training.py) - Model management and training
- [Experiment Tracking](ml_workflows/experiment_tracking.py) - Comprehensive experiment logging
- [Evaluation Pipeline](ml_workflows/evaluation_pipeline.py) - Model evaluation and comparison

### [Integration](integration/)
Cross-module usage and complete workflows:
- [End-to-End Pipeline](integration/end_to_end_pipeline.py) - Complete ML workflow
- [Multi-Module Example](integration/multi_module_example.py) - Cross-module patterns

### [Notebooks](notebooks/)
Interactive Jupyter notebook examples for exploration and learning.

## 🚀 Getting Started

1. **Choose your focus area** from the directories above
2. **Review the example code** - each file is well-documented
3. **Run the examples** - most handle missing dependencies gracefully
4. **Adapt to your needs** - examples are designed to be starting points

## 📋 Example Requirements

Most examples are designed to work without external dependencies by:
- Gracefully handling missing packages with informative error messages
- Providing mock data generation where needed
- Focusing on API usage patterns rather than complex computations

## 🔧 Installation

For full functionality, install refunc with all dependencies:

```bash
# Basic installation
pip install refunc

# Development installation (from source)
git clone https://github.com/kennedym-ds/refunc.git
cd refunc
pip install -e ".[dev]"
```

## 📖 Related Resources

- [API Documentation](../docs/api/) - Complete API reference
- [User Guides](../docs/guides/) - Step-by-step tutorials
- [Quick Start Guide](../docs/guides/quickstart.md) - Get started in 5 minutes

## 💬 Questions & Support

- 🐛 [Report Issues](https://github.com/kennedym-ds/refunc/issues)
- 💬 [Discussions](https://github.com/kennedym-ds/refunc/discussions)
- 📖 [Documentation](../docs/)

---

*Happy coding with Refunc! 🚀*