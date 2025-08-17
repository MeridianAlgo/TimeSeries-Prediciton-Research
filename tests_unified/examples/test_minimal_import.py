#!/usr/bin/env python3
"""Minimal test for basic imports and functionality."""

import sys
import traceback
from pathlib import Path
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_module_import():
    """Test that the main module can be imported."""
    try:
        import ultra_precision_predictor.predictor
        assert hasattr(ultra_precision_predictor.predictor, 'UltraPrecisionPredictor')
        print("✓ Module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")

def test_class_instantiation():
    """Test that the main class can be instantiated."""
    try:
        from ultra_precision_predictor.predictor import UltraPrecisionPredictor
        predictor = UltraPrecisionPredictor()
        assert predictor is not None
        print("✓ Class instantiated successfully")
    except Exception as e:
        pytest.fail(f"Failed to instantiate class: {e}")

def test_simple_predictor_import():
    """Test that the simple predictor can be imported."""
    try:
        from ultra_precision_predictor.simple_predictor import SimplePredictor
        predictor = SimplePredictor()
        assert predictor is not None
        print("✓ SimplePredictor imported and instantiated")
    except Exception as e:
        pytest.fail(f"Failed to import SimplePredictor: {e}")

def test_feature_engineering_import():
    """Test that feature engineering modules can be imported."""
    try:
        from ultra_precision_predictor.feature_engineering.extreme_feature_engineer import ExtremeFeatureEngineer
        engineer = ExtremeFeatureEngineer()
        assert engineer is not None
        print("✓ Feature engineering imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import feature engineering: {e}")

if __name__ == "__main__":
    print("Running minimal import tests...")
    
    test_module_import()
    test_class_instantiation()
    test_simple_predictor_import()
    test_feature_engineering_import()
    
    print("✓ All minimal tests passed!")