# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive documentation system with guides, API reference, and examples
- Cross-platform virtual environment setup scripts with Python auto-detection
- Enhanced statistics module with bootstrap confidence intervals
- Comprehensive testing framework setup with pytest configuration

### Changed

- Improved project structure with better organization
- Enhanced error handling with more specific exception types
- Updated development tools and pre-commit hooks

### Documentation

- Added comprehensive API documentation for all modules
- Created detailed installation and setup guides
- Added contributing guidelines and development workflow
- Created quickstart guide with practical examples

## [0.1.0] - 2025-09-17

### Added

- **Core Framework**
  - Exception handling framework with retry mechanisms
  - Performance monitoring decorators (`@time_it`, `@memory_profile`)
  - Advanced logging system for ML experiments
  - Configuration management system
  - Mathematical and statistical utilities

- **Exception System**
  - `RefuncError` base exception class
  - Specialized exceptions for data, model, and configuration errors
  - `@retry_on_failure` decorator with exponential backoff
  - `RetryableOperation` context manager

- **Performance Decorators**
  - `@time_it` - Execution time measurement
  - `@memory_profile` - Memory usage tracking
  - `@gpu_monitor` - GPU utilization monitoring (optional)
  - `@cache_result` - Intelligent result caching
  - `@log_execution` - Automatic operation logging
  - `@validate_inputs` - Input validation
  - `@profile_performance` - Combined performance monitoring

- **Logging System**
  - `MLLogger` class for experiment tracking
  - Integration with popular ML frameworks (MLflow, Weights & Biases)
  - Colored console output with progress bars
  - Rotating file handlers with compression
  - Metrics, parameters, and artifact logging

- **Configuration Management**
  - `ConfigManager` for dynamic configuration handling
  - YAML and JSON configuration file support
  - Environment variable integration
  - Configuration validation and schema support

- **Mathematical Utilities**
  - `StatisticsEngine` for comprehensive statistical analysis
  - Descriptive statistics with confidence intervals
  - Hypothesis testing (normality, correlation, group comparisons)
  - Outlier detection using multiple methods
  - Bootstrap confidence intervals for any statistic

- **Data Science Utilities**
  - File handling utilities with auto-format detection
  - Data loading and saving for multiple formats (CSV, JSON, Parquet, HDF5)
  - Data validation and schema checking
  - Intelligent caching system

- **Development Tools**
  - Cross-platform setup scripts (Windows, macOS, Linux)
  - Pre-commit hooks configuration (black, isort, flake8, mypy)
  - Comprehensive test suite with pytest
  - CI/CD configuration for GitHub Actions

### Technical Details

- **Python Compatibility**: 3.7+
- **Type Hints**: Complete type annotation coverage
- **Documentation**: Comprehensive docstrings with examples
- **Testing**: High test coverage with unit and integration tests
- **Performance**: Optimized for ML workflow requirements
- **Dependencies**: Minimal required dependencies, optional extras available

### Dependencies

#### Required

- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `psutil>=5.8.0`
- `colorama>=0.4.4`
- `tqdm>=4.62.0`
- `pyyaml>=5.4.0`
- `python-dateutil>=2.8.0`
- `pyarrow>=5.0.0`
- `openpyxl>=3.0.0`
- `h5py>=3.0.0`
- `joblib>=1.1.0`

#### Optional

- **GPU Support**: `GPUtil>=1.4.0`, `py3nvml>=0.2.0`
- **Development**: `black`, `isort`, `flake8`, `mypy`, `pre-commit`
- **Testing**: `pytest`, `pytest-cov`, `pytest-mock`, `pytest-benchmark`
- **Documentation**: `sphinx`, `sphinx-rtd-theme`, `jupyter`, `nbsphinx`

## [0.0.1] - 2025-09-01

### Added

- Initial project structure
- Basic module organization
- Core exception framework
- Initial logging implementation
- Project configuration files (pyproject.toml, setup.py)

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|--------------|--------------|
| 0.1.0   | 2025-09-17   | Complete ML utilities toolkit with comprehensive features |
| 0.0.1   | 2025-09-01   | Initial project setup and basic structure |

## Migration Guides

### Upgrading to 0.1.0 from 0.0.1

This is a major release with significant new features. If you're upgrading from 0.0.1:

1. **New Dependencies**: Install with updated dependencies

   ```bash
   pip install --upgrade refunc
   ```

2. **Import Changes**: Update imports to use new module structure

   ```python
   # Old (0.0.1)
   from refunc import basic_function
   
   # New (0.1.0)
   from refunc.decorators import time_it
   from refunc.logging import MLLogger
   from refunc.exceptions import retry_on_failure
   ```

3. **Configuration**: Update configuration files to use new format

   ```yaml
   # config.yaml (new format)
   logging:
     level: INFO
     format: detailed
   performance:
     enable_profiling: true
   ```

## Future Roadmap

### Planned for 0.2.0

- Enhanced GPU monitoring capabilities
- Integration with more ML frameworks
- Advanced statistical methods
- Performance optimizations
- Extended configuration options

### Planned for 0.3.0

- Web dashboard for experiment tracking
- Database integration for metrics storage
- Advanced visualization tools
- Distributed computing support

### Long-term Goals

- Plugin system for extensibility
- Cloud integration (AWS, Azure, GCP)
- Real-time monitoring and alerting
- Advanced AutoML capabilities

## Contributing

See [CONTRIBUTING.md](docs/developer/contributing.md) for information on how to contribute to this project.

## Support

- 📖 [Documentation](docs/README.md)
- 🐛 [Issue Tracker](https://github.com/kennedym-ds/refunc/issues)
- 💬 [Discussions](https://github.com/kennedym-ds/refunc/discussions)
- 📧 [Email Support](mailto:support@refunc.dev)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This changelog follows the [Keep a Changelog](https://keepachangelog.com/) format. For the complete list of changes, see the [commit history](https://github.com/kennedym-ds/refunc/commits/main).
