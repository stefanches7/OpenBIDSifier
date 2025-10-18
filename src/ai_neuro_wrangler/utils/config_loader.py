"""
Configuration loading utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "normalization": {
            "method": "zscore",
            "clip_percentiles": [1, 99]
        },
        "quality_control": {
            "min_snr": 10.0,
            "max_outlier_ratio": 0.05
        },
        "outlier_detection": {
            "method": "zscore",
            "threshold": 3.0
        },
        "label_encoding": {
            "encoding_type": "label",
            "categorical_columns": None,
            "ordinal_columns": None
        }
    }
