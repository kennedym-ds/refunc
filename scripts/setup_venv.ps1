#!/usr/bin/env pwsh
<#
.SYNOPSIS
    PowerShell wrapper for REFUNC virtual environment setup

.DESCRIPTION
    This script provides a convenient way to run the Python venv setup script
    on Windows systems with proper argument handling.

.PARAMETER VenvName
    Name for the virtual environment (default: venv)

.PARAMETER Dev
    Install development dependencies

.PARAMETER Test
    Install test dependencies

.PARAMETER All
    Install all optional dependencies

.PARAMETER Python
    Use specific Python executable path

.PARAMETER Force
    Force recreate if venv already exists

.EXAMPLE
    .\setup_venv.ps1 -Dev -Test
    
.EXAMPLE
    .\setup_venv.ps1 -VenvName "myenv" -All -Force
#>

param(
    [string]$VenvName = "venv",
    [switch]$Dev,
    [switch]$Test,
    [switch]$All,
    [string]$Python,
    [switch]$Force
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

# Path to the Python setup script
$SetupScript = Join-Path $ScriptDir "setup_venv.py"

# Build arguments array
$Arguments = @($SetupScript)

if ($VenvName -ne "venv") {
    $Arguments += "--venv-name"
    $Arguments += $VenvName
}

if ($Dev) {
    $Arguments += "--dev"
}

if ($Test) {
    $Arguments += "--test"
}

if ($All) {
    $Arguments += "--all"
}

if ($Python) {
    $Arguments += "--python"
    $Arguments += $Python
}

if ($Force) {
    $Arguments += "--force"
}

Write-Host "REFUNC Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Green
Write-Host ""

# Change to project root
Push-Location $ProjectRoot

try {
    # Try to find Python
    $PythonCmd = $null
    
    # Try different Python commands
    $PythonCandidates = @("python", "python3", "py")
    
    foreach ($candidate in $PythonCandidates) {
        try {
            $version = & $candidate --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $PythonCmd = $candidate
                Write-Host "Using Python: $candidate ($version)" -ForegroundColor Green
                break
            }
        }
        catch {
            continue
        }
    }
    
    if (-not $PythonCmd) {
        Write-Host "Error: No Python interpreter found in PATH" -ForegroundColor Red
        Write-Host "Please install Python and ensure it's available in your PATH" -ForegroundColor Yellow
        exit 1
    }
    
    # Run the setup script
    Write-Host "Running setup script..." -ForegroundColor Yellow
    & $PythonCmd @Arguments
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSetup completed successfully!" -ForegroundColor Green
        
        # Provide activation instructions
        Write-Host "`nTo activate the virtual environment in PowerShell:" -ForegroundColor Cyan
        Write-Host "  .\$VenvName\Scripts\Activate.ps1" -ForegroundColor Yellow
        Write-Host "`nTo activate in Command Prompt:" -ForegroundColor Cyan
        Write-Host "  .\$VenvName\Scripts\activate.bat" -ForegroundColor Yellow
    }
    else {
        Write-Host "`nSetup failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
catch {
    Write-Host "Error running setup script: $_" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}