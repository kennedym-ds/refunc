# Refunc - ML Utilities Toolkit

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

**CRITICAL: Dependency Installation Issues**
- Network timeouts are common when installing dependencies from PyPI
- The setup process takes 5-10 minutes when working, but often fails due to network limitations
- NEVER CANCEL build or install commands - wait at least 10 minutes before timing out
- Set timeouts to 600+ seconds (10+ minutes) for all pip install commands

### Bootstrap and Setup
1. **Create virtual environment (always works):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate.bat  # Windows CMD
   # or venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. **Alternative: Use the advanced setup script (interactive):**
   ```bash
   python scripts/setup_venv.py --help  # See all options
   python scripts/setup_venv.py --force --venv-name my_env
   ```
   - **Timing:** Script takes 5-10 minutes when network works
   - **NEVER CANCEL:** Set timeout to 600+ seconds
   - Script will prompt for Python version selection (choose option 1-6)
   - Often fails with `ReadTimeoutError: HTTPSConnectionPool timeout` - this is normal

3. **Install dependencies (often fails due to network):**
   ```bash
   source venv/bin/activate
   pip install --timeout=600 -r requirements/base.txt
   pip install --timeout=600 -r requirements/dev.txt
   ```
   - **Expected failure:** `pip._vendor.urllib3.exceptions.ReadTimeoutError`
   - **Timing:** Takes 10-15 minutes when working, often times out
   - **NEVER CANCEL:** Always wait full timeout period

### Build and Installation
- **Project uses setuptools with pyproject.toml configuration**
- **Development install (often fails due to network):**
  ```bash
  pip install --timeout=600 -e .
  ```
  - **Expected failure:** Network timeouts during dependency resolution
  - **NEVER CANCEL:** Set timeout to 600+ seconds

### Testing
**Without Dependencies (always works):**
```bash
# Basic syntax checking
python -m py_compile refunc/__init__.py
find refunc -name "*.py" | head -10 | xargs python -m py_compile
```

**With Dependencies (requires successful pip install):**
```bash
# Full test suite - takes 5-10 minutes
pytest --timeout=600
pytest -v --cov=refunc --cov-report=term-missing
```

**Test structure:**
- Tests located in `tests/` directory
- Main test file: `tests/__init__.py`
- Minimal test infrastructure - mostly empty

### Linting and Code Quality
**Without Dependencies (limited checking):**
```bash
# Basic Python syntax validation
python -m py_compile file.py
```

**With Dependencies (requires successful pip install):**
```bash
# Pre-commit hooks (takes 2-3 minutes)
pre-commit install  # One-time setup
pre-commit run --all-files --timeout=300  # NEVER CANCEL: Set timeout to 300+ seconds

# Individual tools
black --check .
isort --check-only .
flake8 .
mypy refunc/
```

### CLI Tools Available

1. **Configuration Management CLI:**
   ```bash
   python -m refunc.config.cli --help
   python -m refunc.config.cli template --help
   python -m refunc.config.cli validate --help
   ```
   - **CRITICAL:** Requires full dependencies to be installed
   - **Will fail with:** `ModuleNotFoundError: No module named 'psutil'` if dependencies missing

2. **Setup Script (always works):**
   ```bash
   python scripts/setup_venv.py --help
   python scripts/setup_venv.py --dev  # Include dev dependencies
   python scripts/setup_venv.py --all  # Include all dependencies
   ```

### Running the Application
**Library Usage (requires dependencies):**
```python
from refunc import MLLogger, time_it, memory_profile, FileHandler
from refunc.exceptions import retry_on_failure

# This will fail if dependencies are not installed
```

**Configuration CLI (requires dependencies):**
```bash
python -m refunc.config.cli template --output config.yaml
python -m refunc.config.cli validate config.yaml
```

## Validation

**Always validate changes with these steps:**

1. **Basic syntax checking (always works):**
   ```bash
   python -m py_compile refunc/__init__.py
   find refunc -name "*.py" | head -5 | xargs python -m py_compile
   ```

2. **Environment setup validation (always works):**
   ```bash
   python3 -m venv test_venv
   source test_venv/bin/activate
   python --version  # Should show Python 3.7+
   ```

3. **Setup script validation (always works):**
   ```bash
   python scripts/setup_venv.py --help  # Should show usage
   ```

4. **If dependencies are installed successfully (rarely works due to network):**
   ```bash
   python -c "import refunc; print('Success: refunc module loaded')"
   pytest -x  # Stop on first failure
   pre-commit run --all-files
   ```

**Manual Test Scenarios (when dependencies available):**
- Import and use core decorators: `from refunc import time_it, memory_profile`
- Create and configure ML logger: `from refunc import MLLogger`
- Test file handling utilities: `from refunc.utils import FileHandler`
- Run configuration CLI: `python -m refunc.config.cli template --output test.yaml`

## Common Tasks and Timing Expectations

### Repository Structure Reference
```
refunc/
├── README.md
├── setup.py
├── pyproject.toml              # Main build configuration
├── .pre-commit-config.yaml     # Code quality hooks
├── scripts/
│   ├── setup_env.sh           # Simple setup (often fails)
│   ├── setup_env.bat          # Windows setup
│   ├── setup_env.ps1          # PowerShell setup
│   └── setup_venv.py          # Advanced setup (interactive)
├── requirements/
│   ├── base.txt               # Core dependencies
│   ├── dev.txt                # Development tools
│   └── test.txt               # Testing dependencies
├── refunc/                    # Main package
│   ├── __init__.py
│   ├── utils/                 # File handling utilities
│   ├── logging/               # ML-specific logging
│   ├── exceptions/            # Error handling framework
│   ├── decorators/            # Performance monitoring
│   ├── config/                # Configuration management
│   ├── data_science/          # Data science utilities
│   ├── math_stats/            # Mathematical utilities
│   └── ml/                    # Machine learning utilities
├── examples/                  # Usage examples
├── experiments/               # Experimental code
├── logs/                      # Log output directory
└── tests/                     # Test suite
```

### Expected Timing and Timeout Recommendations
- **Virtual environment creation:** 3-5 seconds - NEVER CANCEL
- **Dependency installation:** 10-15 minutes (often fails) - NEVER CANCEL: Set timeout to 600+ seconds
- **Pre-commit hooks:** 2-3 minutes - NEVER CANCEL: Set timeout to 300+ seconds
- **Test suite:** 5-10 minutes - NEVER CANCEL: Set timeout to 600+ seconds
- **Basic syntax checking:** 1-5 seconds
- **Setup script (interactive):** 5-10 minutes - NEVER CANCEL: Set timeout to 600+ seconds

### Network Limitations
**CRITICAL: Document when commands fail due to network issues:**
- `pip install` commands frequently timeout with `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out`
- This is a known limitation of the environment
- Always attempt installation with high timeouts before concluding failure
- Basic Python functionality and syntax checking work without network access

### Python Version Requirements
- **Minimum:** Python 3.7
- **Tested with:** Python 3.12.3
- **Multiple installations detected:** Scripts auto-detect available Python versions

### Dependencies Summary
**Core dependencies (from requirements/base.txt):**
- pandas>=1.3.0, numpy>=1.21.0, psutil>=5.8.0
- colorama>=0.4.4, tqdm>=4.62.0, pyyaml>=5.4.0
- pyarrow>=5.0.0, openpyxl>=3.0.0, h5py>=3.0.0

**Development tools (from requirements/dev.txt):**
- black>=22.0.0, isort>=5.10.0, flake8>=4.0.0
- mypy>=0.900, pre-commit>=2.15.0
- pytest>=6.2.0, pytest-cov>=2.12.0

**Import Dependencies Required:**
- Most refunc modules require psutil and other external dependencies
- Basic Python modules work without external dependencies
- Configuration CLI requires full dependency stack

## Troubleshooting

### Common Issues
1. **"ModuleNotFoundError: No module named 'psutil'"**
   - Dependencies not installed due to network timeouts
   - Run syntax checking instead: `python -m py_compile file.py`

2. **"ReadTimeoutError: HTTPSConnectionPool timeout"**
   - Normal network limitation in this environment
   - Increase timeout values and wait for completion
   - Use alternative testing approaches that don't require external dependencies

3. **"No such file or directory: pre-commit"**
   - Development dependencies not installed
   - Skip pre-commit validation if dependencies unavailable

### Working Without Full Dependencies
- Use `python -m py_compile` for syntax validation
- Test individual Python files rather than full module imports
- Focus on code structure and logic rather than runtime testing
- Review configuration files and documentation for validation