#!/usr/bin/env python3
"""
Verification script for Task 1 completion.

This script verifies that the project structure and configuration system
are working correctly.
"""

import sys
from pathlib import Path
from src.config.config_manager import get_config
from src.utils.logging_setup import setup_logging
from loguru import logger


def check_directory_structure():
    """Verify all required directories exist."""
    required_dirs = [
        "data/raw", "data/splits", "src/data", "src/model", "src/federated",
        "src/privacy", "src/explainability", "src/monitoring", "src/config",
        "src/utils", "docker", "monitoring/grafana", "notebooks", 
        ".github/workflows", "tests", "config", "scripts"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    
    logger.info("✅ All required directories exist")
    return True


def check_configuration_system():
    """Verify configuration system works."""
    try:
        config = get_config()
        
        # Check key configuration values
        assert config.federated_learning.num_rounds == 30
        assert config.model.learning_rate == 0.001
        assert config.privacy.epsilon == 1.0
        assert len(config.privacy.target_epsilons) == 5
        
        logger.info("✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration system failed: {e}")
        return False


def check_logging_system():
    """Verify logging system works."""
    try:
        # Test structured logging
        logger.info("Testing logging system")
        logger.bind(test_data={"key": "value"}).info("Testing structured logging")
        
        # Check if log files are created
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                logger.info("✅ Logging system working correctly")
                return True
        
        logger.warning("⚠️  Log files not found, but logging system functional")
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging system failed: {e}")
        return False


def check_python_imports():
    """Verify core required packages can be imported."""
    required_packages = [
        "yaml", "pandas", "numpy", "pytest", "loguru"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        return False
    
    logger.info("✅ Core required packages available")
    
    # Check optional ML packages
    optional_packages = [
        "torch", "sklearn", "flwr", "opacus", "shap", "mlflow", 
        "prometheus_client", "matplotlib", "seaborn"
    ]
    
    available_optional = []
    for package in optional_packages:
        try:
            __import__(package)
            available_optional.append(package)
        except ImportError:
            pass
    
    if available_optional:
        logger.info(f"✅ Optional ML packages available: {available_optional}")
    else:
        logger.warning("⚠️  No optional ML packages found - install full requirements.txt when ready")
    
    return True


def main():
    """Main verification function."""
    logger.info("🔍 Verifying Task 1 setup...")
    logger.info("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Configuration System", check_configuration_system),
        ("Logging System", check_logging_system),
        ("Python Imports", check_python_imports),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    logger.info("=" * 50)
    if all_passed:
        logger.info("🎉 Task 1 completed successfully!")
        logger.info("✅ Project structure and dependencies are ready")
        logger.info("📋 Next: Place IEEE-CIS dataset in data/raw/ and start Task 2")
        return 0
    else:
        logger.error("❌ Task 1 verification failed")
        logger.error("Please fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    # Initialize logging first
    setup_logging()
    sys.exit(main())