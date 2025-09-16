@echo off
REM Batch wrapper for REFUNC virtual environment setup
REM
REM Usage:
REM   setup_venv.bat [options]
REM
REM Options will be passed to the Python script:
REM   --venv-name NAME    Name for the virtual environment
REM   --dev              Install development dependencies
REM   --test             Install test dependencies  
REM   --all              Install all optional dependencies
REM   --python PATH      Use specific Python executable
REM   --force            Force recreate if venv already exists
REM
REM Examples:
REM   setup_venv.bat --dev --test
REM   setup_venv.bat --venv-name myenv --all --force

setlocal

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "SETUP_SCRIPT=%SCRIPT_DIR%setup_venv.py"

echo REFUNC Virtual Environment Setup
echo Project Root: %PROJECT_ROOT%
echo.

REM Change to project root
pushd "%PROJECT_ROOT%"

REM Try to find Python
set "PYTHON_CMD="

REM Try different Python commands
for %%p in (python python3 py) do (
    %%p --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=%%p"
        for /f "tokens=*" %%v in ('%%p --version 2^>^&1') do echo Using Python: %%p ^(%%v^)
        goto :found_python
    )
)

:found_python
if "%PYTHON_CMD%"=="" (
    echo Error: No Python interpreter found in PATH
    echo Please install Python and ensure it's available in your PATH
    popd
    exit /b 1
)

REM Run the setup script with all passed arguments
echo Running setup script...
"%PYTHON_CMD%" "%SETUP_SCRIPT%" %*

if %errorlevel% equ 0 (
    echo.
    echo Setup completed successfully!
    echo.
    echo To activate the virtual environment:
    echo   venv\Scripts\activate.bat
) else (
    echo.
    echo Setup failed with exit code: %errorlevel%
    popd
    exit /b %errorlevel%
)

popd
endlocal