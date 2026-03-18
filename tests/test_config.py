"""
Test configuration management system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.config.config_manager import ConfigManager, Config


def test_config_loading():
    """Test basic configuration loading."""
    config_manager = ConfigManager("config/config.yaml")
    config = config_manager.load_config()
    
    assert isinstance(config, Config)
    assert config.federated_learning.num_rounds == 30
    assert config.model.learning_rate == 0.001
    assert config.privacy.epsilon == 1.0


def test_config_validation():
    """Test configuration validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        invalid_config = {
            'federated_learning': {'num_rounds': -1},  # Invalid: negative rounds
            'model': {'learning_rate': -0.1},  # Invalid: negative learning rate
        }
        yaml.dump(invalid_config, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigManager(temp_path)
        with pytest.raises(ValueError):
            config_manager.load_config()
    finally:
        Path(temp_path).unlink()


def test_environment_overrides(monkeypatch):
    """Test environment variable overrides."""
    # Set environment variables
    monkeypatch.setenv("FFD_MODEL__LEARNING_RATE", "0.01")
    monkeypatch.setenv("FFD_PRIVACY__EPSILON", "2.0")
    
    config_manager = ConfigManager("config/config.yaml")
    config = config_manager.load_config()
    
    assert config.model.learning_rate == 0.01
    assert config.privacy.epsilon == 2.0


def test_config_save_load_roundtrip():
    """Test configuration save and load round-trip."""
    config_manager = ConfigManager("config/config.yaml")
    original_config = config_manager.load_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save and reload
        config_manager.save_config(original_config, temp_path)
        
        new_config_manager = ConfigManager(temp_path)
        reloaded_config = new_config_manager.load_config()
        
        # Compare key values
        assert reloaded_config.federated_learning.num_rounds == original_config.federated_learning.num_rounds
        assert reloaded_config.model.learning_rate == original_config.model.learning_rate
        assert reloaded_config.privacy.epsilon == original_config.privacy.epsilon
        
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])