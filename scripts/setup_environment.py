#!/usr/bin/env python3
"""
Environment Setup Script for Federated Fraud Detection System

This script sets up the Python virtual environment and installs dependencies.
Ensures Python 3.10+ compatibility for all ML libraries.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Current version: {version.major}.{version.minor}")
        print("Please install Python 3.10 or higher and try again.")
        sys.exit(1)
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")


def create_virtual_environment():
    """Create Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    print("📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(1)


def get_activation_command():
    """Get the appropriate activation command for the current OS."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Get the appropriate pip executable
    if platform.system() == "Windows":
        pip_executable = "venv\\Scripts\\pip"
        python_executable = "venv\\Scripts\\python"
    else:
        pip_executable = "venv/bin/pip"
        python_executable = "venv/bin/python"
    
    try:
        # Try to upgrade pip using python -m pip (more reliable on Windows)
        try:
            subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("⚠️  Pip upgrade failed, continuing with existing version...")
        
        # Install requirements
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
        
        # Install package in development mode
        subprocess.run([pip_executable, "install", "-e", "."], check=True)
        
        print("✅ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/splits", 
        "models",
        "logs",
        "results",
        "notebooks"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")


def setup_git_hooks():
    """Set up pre-commit hooks if git repository exists."""
    if Path(".git").exists():
        try:
            # Get the appropriate pre-commit executable
            if platform.system() == "Windows":
                precommit_executable = "venv\\Scripts\\pre-commit"
            else:
                precommit_executable = "venv/bin/pre-commit"
            
            subprocess.run([precommit_executable, "install"], check=True)
            print("✅ Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Pre-commit hooks not installed (optional)")


def main():
    """Main setup function."""
    print("🚀 Setting up Federated Fraud Detection System Environment")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Setup git hooks
    setup_git_hooks()
    
    print("\n" + "=" * 60)
    print("🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate virtual environment: {get_activation_command()}")
    print("2. Place IEEE-CIS dataset files in data/raw/")
    print("3. Run: python -c 'from src.config.config_manager import get_config; print(get_config())'")
    print("4. Start with Task 2: Data preprocessing")


if __name__ == "__main__":
    main()