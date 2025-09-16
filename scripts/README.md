# REFUNC Scripts

This directory contains utility scripts for the REFUNC project.

## Virtual Environment Setup

The virtual environment setup script helps you quickly set up a development environment for REFUNC by:

1. **Scanning for Python installations** across your system
2. **Letting you choose** which Python version to use
3. **Creating a virtual environment** with your chosen Python
4. **Installing dependencies** from `pyproject.toml` or requirements files
5. **Providing activation instructions** for your platform

### Usage Options

#### Python Script (Cross-platform)

```bash
python scripts/setup_venv.py [options]
```

#### PowerShell (Windows)

```powershell
.\scripts\setup_venv.ps1 [options]
```

#### Batch File (Windows Command Prompt)

```cmd
scripts\setup_venv.bat [options]
```

### Available Options

| Option | Description |
|--------|-------------|
| `--venv-name NAME` | Name for the virtual environment (default: `venv`) |
| `--dev` | Install development dependencies |
| `--test` | Install test dependencies |
| `--all` | Install all optional dependencies (dev, test, gpu, docs) |
| `--python PATH` | Use specific Python executable path |
| `--force` | Force recreate if venv already exists |

### Examples

```bash
# Basic setup with test dependencies
python scripts/setup_venv.py --test

# Development setup with all dependencies
python scripts/setup_venv.py --dev --all

# Custom virtual environment name
python scripts/setup_venv.py --venv-name myproject --dev --test

# Force recreate existing environment
python scripts/setup_venv.py --force --all

# Use specific Python version
python scripts/setup_venv.py --python C:\Python39\python.exe --dev
```

### PowerShell Examples

```powershell
# Basic setup
.\scripts\setup_venv.ps1 -Test

# Full development setup
.\scripts\setup_venv.ps1 -Dev -All

# Custom environment name with force recreation
.\scripts\setup_venv.ps1 -VenvName "testenv" -Force -Test
```

## Features

### Intelligent Python Discovery

The script automatically finds Python installations from common locations:

- **Windows**: Registry entries, Program Files, AppData, conda environments
- **macOS**: Homebrew, pyenv, system locations, conda environments  
- **Linux**: System packages, pyenv, snap packages, conda environments

### Dependency Management

The script intelligently handles dependencies:

1. **pyproject.toml** (preferred): Installs the project in editable mode with optional dependencies
2. **requirements.txt**: Falls back to traditional requirements files
3. **Multiple requirements files**: Supports `requirements-dev.txt`, `requirements-test.txt`, etc.

### Cross-Platform Support

- Works on Windows, macOS, and Linux
- Provides platform-specific activation instructions
- Handles different shell environments (PowerShell, Command Prompt, bash, zsh)

### Error Handling

- Validates Python installations before use
- Provides clear error messages
- Offers suggestions for common issues
- Graceful handling of missing dependencies

## Activation Instructions

After the virtual environment is created, activate it using:

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
venv\Scripts\activate.bat
```

### Linux/macOS

```bash
source venv/bin/activate
```

### Deactivation

```bash
deactivate
```

## Troubleshooting

### No Python Found

If the script can't find Python:

1. Ensure Python is installed
2. Add Python to your PATH environment variable
3. Use the `--python` option to specify the full path

### Permission Errors

On Windows, you might need to:

1. Run as Administrator
2. Enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Virtual Environment Already Exists

Use the `--force` option to recreate an existing environment:

```bash
python scripts/setup_venv.py --force
```

## Dependencies

The setup script requires:

- Python 3.7+ (any version found on system)
- No additional dependencies (uses only standard library)

Optional enhancements:

- `winreg` (Windows registry access, included in Python on Windows)
- `colorama` (colored output, will be installed if available)

## Contributing

To improve the setup script:

1. Test on different platforms and Python versions
2. Add support for additional package managers
3. Improve error messages and user experience
4. Add more intelligent dependency detection
