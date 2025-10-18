"""
Tests for utility modules.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
import json

from ai_neuro_wrangler.utils.config_loader import (
    load_config,
    save_config,
    get_default_config
)
from ai_neuro_wrangler.utils.logger import setup_logger


class TestConfigLoader:
    """Tests for configuration loading utilities."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        
        assert isinstance(config, dict)
        assert "normalization" in config
        assert "quality_control" in config
        assert "outlier_detection" in config
        assert "label_encoding" in config
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        test_config = {
            "normalization": {"method": "zscore"},
            "test_key": "test_value"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            save_config(test_config, str(config_path))
            assert config_path.exists()
            
            loaded_config = load_config(str(config_path))
            assert loaded_config["normalization"]["method"] == "zscore"
            assert loaded_config["test_key"] == "test_value"
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        test_config = {
            "normalization": {"method": "minmax"},
            "test_key": "test_value"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            save_config(test_config, str(config_path))
            assert config_path.exists()
            
            loaded_config = load_config(str(config_path))
            assert loaded_config["normalization"]["method"] == "minmax"
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    def test_unsupported_format(self):
        """Test unsupported config format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.txt"
            config_path.write_text("test")
            
            with pytest.raises(ValueError):
                load_config(str(config_path))


class TestLogger:
    """Tests for logging utilities."""
    
    def test_setup_logger(self):
        """Test logger setup."""
        logger = setup_logger("test_logger")
        
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_logger_file", log_file=str(log_file))
            
            logger.info("Test message")
            
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
    
    def test_logger_singleton(self):
        """Test that logger returns same instance."""
        logger1 = setup_logger("singleton_test")
        logger2 = setup_logger("singleton_test")
        
        assert logger1 is logger2
