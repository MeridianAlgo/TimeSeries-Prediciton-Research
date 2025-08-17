"""Configuration management for the stock predictor system."""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigurationManager:
    """Manages system configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = self._get_default_config()
    
    def save_config(self) -> None:
        """Save current configuration to YAML file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'default_symbol': 'AAPL',
                'lookback_years': 5,
                'train_ratio': 0.7,
                'validation_ratio': 0.15,
                'test_ratio': 0.15,
                'features': {
                    'moving_averages': [5, 10, 20, 50],
                    'volatility_window': 20,
                    'lag_periods': [1, 2, 3, 5]
                }
            },
            'models': {
                'arima': {
                    'max_p': 5,
                    'max_d': 2,
                    'max_q': 5,
                    'seasonal': True
                },
                'lstm': {
                    'sequence_length': 60,
                    'units': [50, 50],
                    'dropout': 0.2,
                    'epochs': 100,
                    'batch_size': 32
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                }
            },
            'ensemble': {
                'weighting_method': 'inverse_error',
                'min_weight': 0.1,
                'confidence_level': 0.95,
                'dynamic_adjustment': True,
                'adjustment_window': 30
            }
        }