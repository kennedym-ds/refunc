#!/usr/bin/env python3
"""
Virtual Environment Setup Helper for REFUNC

This script helps set up a virtual environment for the REFUNC project by:
1. Scanning the system for available Python installations
2. Allowing user to select a Python version
3. Creating a virtual environment
4. Installing dependencies from pyproject.toml or requirements files
5. Optionally installing development and test dependencies

Usage:
    python scripts/setup_venv.py [options]

Options:
    --venv-name NAME    Name for the virtual environment (default: venv)
    --dev              Install development dependencies
    --test             Install test dependencies
    --all              Install all optional dependencies
    --python PATH      Use specific Python executable
    --force            Force recreate if venv already exists
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PythonInstallation:
    """Represents a Python installation."""
    
    def __init__(self, executable: Path, version: str, architecture: str = ""):
        self.executable = executable
        self.version = version
        self.architecture = architecture
        
    def __str__(self):
        arch_str = f" ({self.architecture})" if self.architecture else ""
        return f"Python {self.version}{arch_str} - {self.executable}"
    
    @property
    def version_tuple(self) -> Tuple[int, ...]:
        """Get version as tuple for comparison."""
        try:
            return tuple(map(int, self.version.split('.')[:3]))
        except:
            return (0, 0, 0)


class VenvSetup:
    """Virtual environment setup manager."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.python_installations: List[PythonInstallation] = []
        
    def print_colored(self, message: str, color: str = Colors.ENDC):
        """Print colored message."""
        print(f"{color}{message}{Colors.ENDC}")
        
    def print_header(self, message: str):
        """Print header message."""
        self.print_colored(f"\n{Colors.BOLD}{'='*60}", Colors.HEADER)
        self.print_colored(f"{message}", Colors.HEADER + Colors.BOLD)
        self.print_colored(f"{'='*60}{Colors.ENDC}", Colors.HEADER)
        
    def find_python_installations(self) -> List[PythonInstallation]:
        """Find all Python installations on the system."""
        self.print_header("Scanning for Python Installations")
        
        installations = []
        system = platform.system().lower()
        
        # Common Python executable names
        python_names = ['python', 'python3', 'py']
        for i in range(20):  # python3.7, python3.8, etc.
            python_names.extend([f'python3.{i}', f'python{3}.{i}'])
        
        # Platform-specific search paths
        if system == 'windows':
            search_paths = self._get_windows_python_paths()
        elif system == 'darwin':  # macOS
            search_paths = self._get_macos_python_paths()
        else:  # Linux and others
            search_paths = self._get_linux_python_paths()
            
        # Also check PATH
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        search_paths.extend([Path(p) for p in path_dirs if p])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in search_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        
        print(f"{Colors.OKBLUE}Searching in {len(unique_paths)} directories...{Colors.ENDC}")
        
        # Search for Python executables
        for search_path in unique_paths:
            if not search_path.exists():
                continue
                
            for python_name in python_names:
                if system == 'windows' and not python_name.endswith('.exe'):
                    python_name += '.exe'
                    
                python_path = search_path / python_name
                if python_path.is_file():
                    installation = self._check_python_executable(python_path)
                    if installation:
                        installations.append(installation)
        
        # Remove duplicates based on version and path
        unique_installations = []
        seen_versions = set()
        
        for installation in sorted(installations, key=lambda x: x.version_tuple, reverse=True):
            key = (installation.version, str(installation.executable))
            if key not in seen_versions:
                seen_versions.add(key)
                unique_installations.append(installation)
        
        self.python_installations = unique_installations
        
        if unique_installations:
            self.print_colored(f"Found {len(unique_installations)} Python installation(s):", Colors.OKGREEN)
            for i, installation in enumerate(unique_installations, 1):
                print(f"  {i}. {installation}")
        else:
            self.print_colored("No Python installations found!", Colors.FAIL)
            
        return unique_installations
    
    def _get_windows_python_paths(self) -> List[Path]:
        """Get Windows-specific Python search paths."""
        paths = []
        
        # Common Windows Python locations
        common_paths = [
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python',
            Path('C:/Python*'),
            Path('C:/Program Files/Python*'),
            Path('C:/Program Files (x86)/Python*'),
            Path(os.environ.get('APPDATA', '')) / 'Python',
        ]
        
        for path_pattern in common_paths:
            if '*' in str(path_pattern):
                try:
                    import glob
                    paths.extend([Path(p) for p in glob.glob(str(path_pattern))])
                except:
                    pass
            else:
                paths.append(path_pattern)
        
        # Check conda environments
        conda_paths = self._get_conda_python_paths()
        paths.extend(conda_paths)
        
        # Windows Registry search
        try:
            import winreg
            python_paths = self._search_windows_registry()
            paths.extend(python_paths)
        except ImportError:
            pass
            
        return paths
    
    def _get_macos_python_paths(self) -> List[Path]:
        """Get macOS-specific Python search paths."""
        paths = [
            Path('/usr/bin'),
            Path('/usr/local/bin'),
            Path('/opt/homebrew/bin'),
            Path('/System/Library/Frameworks/Python.framework/Versions'),
            Path('/Library/Frameworks/Python.framework/Versions'),
            Path(os.path.expanduser('~/.pyenv/versions')),
        ]
        
        # Check conda environments
        conda_paths = self._get_conda_python_paths()
        paths.extend(conda_paths)
        
        return paths
    
    def _get_linux_python_paths(self) -> List[Path]:
        """Get Linux-specific Python search paths."""
        paths = [
            Path('/usr/bin'),
            Path('/usr/local/bin'),
            Path('/opt/python'),
            Path(os.path.expanduser('~/.pyenv/versions')),
            Path('/snap/bin'),  # Snap packages
        ]
        
        # Check conda environments
        conda_paths = self._get_conda_python_paths()
        paths.extend(conda_paths)
        
        return paths
    
    def _get_conda_python_paths(self) -> List[Path]:
        """Get conda environment Python paths."""
        paths = []
        
        # Try to find conda installations
        possible_conda_roots = [
            Path(os.path.expanduser('~/anaconda3')),
            Path(os.path.expanduser('~/miniconda3')),
            Path(os.path.expanduser('~/mambaforge')),
            Path(os.path.expanduser('~/miniforge3')),
            Path('C:/Anaconda3'),
            Path('C:/Miniconda3'),
            Path('C:/ProgramData/Anaconda3'),
            Path('C:/ProgramData/Miniconda3'),
        ]
        
        for conda_root in possible_conda_roots:
            if conda_root.exists():
                # Base environment
                paths.append(conda_root)
                paths.append(conda_root / 'Scripts')  # Windows
                paths.append(conda_root / 'bin')     # Unix
                
                # Environment directories
                envs_dir = conda_root / 'envs'
                if envs_dir.exists():
                    for env_dir in envs_dir.iterdir():
                        if env_dir.is_dir():
                            paths.append(env_dir)
                            paths.append(env_dir / 'Scripts')  # Windows
                            paths.append(env_dir / 'bin')     # Unix
        
        return paths
    
    def _search_windows_registry(self) -> List[Path]:
        """Search Windows registry for Python installations."""
        paths = []
        try:
            import winreg
            
            # Python launcher registry keys
            registry_keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python\PythonCore"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python\PythonCore"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Python\PythonCore"),
            ]
            
            for hkey, subkey in registry_keys:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        i = 0
                        while True:
                            try:
                                version = winreg.EnumKey(key, i)
                                install_path_key = f"{subkey}\\{version}\\InstallPath"
                                with winreg.OpenKey(hkey, install_path_key) as install_key:
                                    install_path, _ = winreg.QueryValueEx(install_key, "")
                                    paths.append(Path(install_path))
                                i += 1
                            except OSError:
                                break
                except OSError:
                    continue
                    
        except ImportError:
            pass
            
        return paths
    
    def _check_python_executable(self, python_path: Path) -> Optional[PythonInstallation]:
        """Check if a path is a valid Python executable and get its info."""
        try:
            # Run python --version to get version info
            result = subprocess.run(
                [str(python_path), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version_output = result.stdout.strip() or result.stderr.strip()
                # Extract version number (e.g., "Python 3.9.7" -> "3.9.7")
                version = version_output.replace('Python ', '').strip()
                
                # Get architecture info
                arch_result = subprocess.run(
                    [str(python_path), '-c', 'import platform; print(platform.architecture()[0])'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                architecture = ""
                if arch_result.returncode == 0:
                    architecture = arch_result.stdout.strip()
                
                return PythonInstallation(python_path, version, architecture)
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return None
    
    def select_python_version(self, python_path: Optional[str] = None) -> PythonInstallation:
        """Let user select a Python version or use specified path."""
        if python_path:
            # Use specified Python path
            python_path_obj = Path(python_path)
            if not python_path_obj.exists():
                raise ValueError(f"Specified Python path does not exist: {python_path}")
            
            installation = self._check_python_executable(python_path_obj)
            if not installation:
                raise ValueError(f"Invalid Python executable: {python_path}")
            
            self.print_colored(f"Using specified Python: {installation}", Colors.OKGREEN)
            return installation
        
        # Find available installations
        installations = self.find_python_installations()
        
        if not installations:
            raise RuntimeError("No Python installations found on the system")
        
        if len(installations) == 1:
            installation = installations[0]
            self.print_colored(f"Using only available Python: {installation}", Colors.OKGREEN)
            return installation
        
        # Let user choose
        print(f"\n{Colors.OKBLUE}Select a Python version:{Colors.ENDC}")
        for i, installation in enumerate(installations, 1):
            print(f"  {i}. {installation}")
        
        while True:
            try:
                choice = input(f"\n{Colors.OKCYAN}Enter choice (1-{len(installations)}): {Colors.ENDC}")
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(installations):
                    selected = installations[choice_idx]
                    self.print_colored(f"Selected: {selected}", Colors.OKGREEN)
                    return selected
                else:
                    self.print_colored(f"Invalid choice. Please enter 1-{len(installations)}", Colors.WARNING)
                    
            except (ValueError, KeyboardInterrupt):
                self.print_colored("Invalid input or cancelled.", Colors.FAIL)
                sys.exit(1)
    
    def create_virtual_environment(self, python_installation: PythonInstallation, venv_name: str, force: bool = False) -> Path:
        """Create a virtual environment."""
        self.print_header(f"Creating Virtual Environment: {venv_name}")
        
        venv_path = self.project_root / venv_name
        
        if venv_path.exists():
            if force:
                self.print_colored(f"Removing existing virtual environment...", Colors.WARNING)
                shutil.rmtree(venv_path)
            else:
                raise RuntimeError(f"Virtual environment already exists: {venv_path}\nUse --force to recreate")
        
        # Create virtual environment
        self.print_colored(f"Creating virtual environment at: {venv_path}", Colors.OKBLUE)
        
        try:
            subprocess.run(
                [str(python_installation.executable), '-m', 'venv', str(venv_path)],
                check=True,
                cwd=self.project_root
            )
            
            self.print_colored("Virtual environment created successfully!", Colors.OKGREEN)
            return venv_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create virtual environment: {e}")
    
    def get_venv_python(self, venv_path: Path) -> Path:
        """Get the Python executable in the virtual environment."""
        if platform.system().lower() == 'windows':
            return venv_path / 'Scripts' / 'python.exe'
        else:
            return venv_path / 'bin' / 'python'
    
    def get_venv_pip(self, venv_path: Path) -> Path:
        """Get the pip executable in the virtual environment."""
        if platform.system().lower() == 'windows':
            return venv_path / 'Scripts' / 'pip.exe'
        else:
            return venv_path / 'bin' / 'pip'
    
    def install_dependencies(self, venv_path: Path, dev: bool = False, test: bool = False, all_deps: bool = False):
        """Install dependencies from pyproject.toml or requirements files."""
        self.print_header("Installing Dependencies")
        
        pip_exe = self.get_venv_pip(venv_path)
        python_exe = self.get_venv_python(venv_path)
        
        # Upgrade pip first using python -m pip to avoid conflicts
        self.print_colored("Checking pip version...", Colors.OKBLUE)
        try:
            # Check current pip version
            pip_version_result = subprocess.run([str(python_exe), '-m', 'pip', '--version'], 
                                              capture_output=True, text=True, check=True)
            current_version = pip_version_result.stdout.strip()
            self.print_colored(f"Current pip: {current_version}", Colors.OKCYAN)
            
            # Try to upgrade pip
            self.print_colored("Upgrading pip if needed...", Colors.OKBLUE)
            upgrade_result = subprocess.run([str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                          capture_output=True, text=True, timeout=60)
            
            if upgrade_result.returncode == 0:
                if "Requirement already satisfied" in upgrade_result.stdout:
                    self.print_colored("Pip is already up to date!", Colors.OKGREEN)
                else:
                    self.print_colored("Pip upgraded successfully!", Colors.OKGREEN)
            else:
                self.print_colored(f"Warning: Could not upgrade pip: {upgrade_result.stderr}", Colors.WARNING)
                self.print_colored("Continuing with existing pip version...", Colors.WARNING)
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.print_colored(f"Warning: Could not check/upgrade pip: {e}", Colors.WARNING)
            self.print_colored("Continuing with existing pip version...", Colors.WARNING)
        
        # Check for pyproject.toml
        pyproject_path = self.project_root / 'pyproject.toml'
        
        if pyproject_path.exists():
            self.print_colored("Found pyproject.toml, installing project in development mode...", Colors.OKBLUE)
            
            # Install the project itself in editable mode
            install_cmd = [str(python_exe), '-m', 'pip', 'install', '-e', '.']
            
            # Add optional dependencies
            extras = []
            if dev:
                extras.append('dev')
            if test:
                extras.append('test')
            if all_deps:
                extras.extend(['gpu', 'docs'])  # Add all known extras
            
            if extras:
                install_cmd[-1] = f".[{','.join(extras)}]"
            
            try:
                subprocess.run(install_cmd, cwd=self.project_root, check=True)
                self.print_colored("Project installed successfully!", Colors.OKGREEN)
            except subprocess.CalledProcessError as e:
                self.print_colored(f"Error installing project: {e}", Colors.FAIL)
                raise
            
        else:
            # Fallback to requirements files
            requirements_files = [
                'requirements.txt',
                'requirements/base.txt',
                'requirements/requirements.txt'
            ]
            
            if dev:
                requirements_files.extend([
                    'requirements-dev.txt',
                    'requirements/dev.txt',
                    'requirements/development.txt'
                ])
            
            if test:
                requirements_files.extend([
                    'requirements-test.txt',
                    'requirements/test.txt',
                    'requirements/testing.txt'
                ])
            
            installed_any = False
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    self.print_colored(f"Installing from {req_file}...", Colors.OKBLUE)
                    try:
                        subprocess.run([str(python_exe), '-m', 'pip', 'install', '-r', str(req_path)], check=True)
                        installed_any = True
                    except subprocess.CalledProcessError as e:
                        self.print_colored(f"Error installing from {req_file}: {e}", Colors.WARNING)
            
            if not installed_any:
                self.print_colored("No requirements files found. Installing basic packages...", Colors.WARNING)
                # Install some basic packages for testing
                basic_packages = ['pytest', 'pytest-cov', 'black', 'isort', 'flake8']
                try:
                    subprocess.run([str(python_exe), '-m', 'pip', 'install'] + basic_packages, check=True)
                except subprocess.CalledProcessError as e:
                    self.print_colored(f"Error installing basic packages: {e}", Colors.WARNING)
        
        self.print_colored("Dependencies installed successfully!", Colors.OKGREEN)
    
    def print_activation_instructions(self, venv_path: Path):
        """Print instructions for activating the virtual environment."""
        self.print_header("Virtual Environment Ready!")
        
        system = platform.system().lower()
        
        self.print_colored("To activate the virtual environment:", Colors.OKBLUE)
        
        if system == 'windows':
            if 'powershell' in os.environ.get('PSMODULEPATH', '').lower():
                activate_cmd = f"{venv_path}\\Scripts\\Activate.ps1"
            else:
                activate_cmd = f"{venv_path}\\Scripts\\activate.bat"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        self.print_colored(f"  {activate_cmd}", Colors.OKGREEN + Colors.BOLD)
        
        self.print_colored("\nTo deactivate when done:", Colors.OKBLUE)
        self.print_colored("  deactivate", Colors.OKGREEN + Colors.BOLD)
        
        # Print Python and pip paths
        python_exe = self.get_venv_python(venv_path)
        pip_exe = self.get_venv_pip(venv_path)
        
        print(f"\n{Colors.OKCYAN}Virtual Environment Details:{Colors.ENDC}")
        print(f"  Location: {venv_path}")
        print(f"  Python:   {python_exe}")
        print(f"  Pip:      {pip_exe}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up virtual environment for REFUNC project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--venv-name',
        default='venv',
        help='Name for the virtual environment (default: venv)'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Install development dependencies'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Install test dependencies'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Install all optional dependencies'
    )
    
    parser.add_argument(
        '--python',
        help='Use specific Python executable path'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreate if venv already exists'
    )
    
    args = parser.parse_args()
    
    try:
        # Find project root
        current_dir = Path.cwd()
        project_root = current_dir
        
        # Look for pyproject.toml or setup.py to find project root
        while project_root.parent != project_root:
            if (project_root / 'pyproject.toml').exists() or (project_root / 'setup.py').exists():
                break
            project_root = project_root.parent
        
        if not ((project_root / 'pyproject.toml').exists() or (project_root / 'setup.py').exists()):
            project_root = current_dir  # Fallback to current directory
        
        venv_setup = VenvSetup(project_root)
        
        venv_setup.print_colored(
            f"REFUNC Virtual Environment Setup", 
            Colors.HEADER + Colors.BOLD
        )
        venv_setup.print_colored(f"Project root: {project_root}", Colors.OKCYAN)
        
        # Select Python version
        python_installation = venv_setup.select_python_version(args.python)
        
        # Create virtual environment
        venv_path = venv_setup.create_virtual_environment(
            python_installation, 
            args.venv_name, 
            args.force
        )
        
        # Install dependencies
        venv_setup.install_dependencies(
            venv_path, 
            dev=args.dev or args.all, 
            test=args.test or args.all,
            all_deps=args.all
        )
        
        # Print activation instructions
        venv_setup.print_activation_instructions(venv_path)
        
    except KeyboardInterrupt:
        print(f"{Colors.WARNING}\nSetup cancelled by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.FAIL}\nError: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == '__main__':
    main()