#!/usr/bin/env python3
"""
Simple verification script for Task 1 completion.
"""

import sys
from pathlib import Path


def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        "data/raw", "data/splits", "src/data", "src/model", "src/federated",
        "src/privacy", "src/explainability", "src/monitoring", "src/config",
        "src/utils", "config", "scripts"
    ]
    
    missing = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing.append(directory)
    
    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    
    print("✅ All required directories exist")
    return True


def check_files():
    """Check if key files exist."""
    required_files = [
        "requirements.txt", "requirements-minimal.txt", "setup.py", "README.md",
        "config/config.yaml", "src/config/config_manager.py", 
        "src/utils/logging_setup.py", "verify_setup.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All required files exist")
    return True


def check_imports():
    """Check if core packages can be imported."""
    core_packages = ["yaml", "pandas", "numpy", "pytest", "loguru"]
    
    missing = []
    for package in core_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing core packages: {missing}")
        return False
    
    print("✅ Core packages available")
    
    # Check optional ML packages
    ml_packages = ["torch", "sklearn", "opacus", "shap", "mlflow", "matplotlib"]
    available = []
    
    for package in ml_packages:
        try:
            __import__(package)
            available.append(package)
        except ImportError:
            pass
    
    if available:
        print(f"✅ ML packages available: {available}")
    else:
        print("⚠️  No ML packages found - install full requirements.txt when ready")
    
    return True


def main():
    """Main verification function."""
    print("🔍 Verifying Task 1 setup...")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directories),
        ("Required Files", check_files),
        ("Python Imports", check_imports),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"Checking {check_name}...")
        if not check_func():
            all_passed = False
        print()
    
    print("=" * 50)
    if all_passed:
        print("🎉 Task 1 completed successfully!")
        print("✅ Project structure and core dependencies are ready")
        print("📋 Next steps:")
        print("  1. Place IEEE-CIS dataset in data/raw/")
        print("  2. Install full requirements: pip install -r requirements.txt")
        print("  3. Start Task 2: Data preprocessing")
        return 0
    else:
        print("❌ Task 1 verification failed")
        print("Please fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())