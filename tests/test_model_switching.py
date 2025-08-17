"""
Unit tests for model switching functionality.
Tests automatic model switching, backup systems, and graceful degradation.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_timeseries.monitoring.model_switching import (
    AdaptiveModelSwitcher, ModelSwitchingEngine, PerformanceEvaluator,
    ModelPerformanceSnapshot, SwitchingRule, SwitchingTrigger, SwitchingStrategy
)


class MockModel:
    """Mock model for testing."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name = model_id
    
    def predict(self, *args, **kwargs):
        """Mock prediction method."""
        return np.random.random()


class TestAdaptiveModelSwitcher(unittest.TestCase):
    """Test cases for AdaptiveModelSwitcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.switcher = AdaptiveModelSwitcher(cooldown_minutes=30)
        self.test_model1 = MockModel("model1")
        self.test_model2 = MockModel("model2")
        self.switcher.register_model("model1", self.test_model1)
        self.switcher.register_model("model2", self.test_model2)
    
    def test_initialization(self):
        """Test AdaptiveModelSwitcher initialization."""
        self.assertIsNotNone(self.switcher.engine)
        self.assertFalse(self.switcher.is_running)
    
    def test_register_model(self):
        """Test model registration."""
        model3 = MockModel("model3")
        self.switcher.register_model("model3", model3)
        
        # Test that model is registered
        active_model = self.switcher.get_active_model()
        # Should return the first registered model as active
        self.assertIsNotNone(active_model)
        self.assertEqual(active_model.model_id, "model1")
    
    def test_manual_switch(self):
        """Test manual model switching."""
        # First set an active model
        self.switcher.engine.active_model_id = "model1"
        
        # Switch to model2
        success = self.switcher.manual_switch("model2", "Test switch")
        
        self.assertTrue(success)
        self.assertEqual(self.switcher.get_active_model_id(), "model2")
    
    def test_get_active_model_id(self):
        """Test getting active model ID."""
        # Should return the first registered model as active
        self.assertEqual(self.switcher.get_active_model_id(), "model1")
        
        # Test switching to another model
        self.switcher.manual_switch("model2", "Test switch")
        self.assertEqual(self.switcher.get_active_model_id(), "model2")
    
    def test_add_prediction(self):
        """Test adding predictions."""
        self.switcher.add_prediction("model1", 100.0, 101.0, 0.8)
        
        # Check that prediction was added to evaluator
        snapshot = self.switcher.get_model_performance("model1")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.model_id, "model1")


class TestPerformanceEvaluator(unittest.TestCase):
    """Test cases for PerformanceEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = PerformanceEvaluator(evaluation_window=100)
    
    def test_initialization(self):
        """Test PerformanceEvaluator initialization."""
        self.assertEqual(self.evaluator.evaluation_window, 100)
        self.assertIsInstance(self.evaluator.model_predictions, dict)
        self.assertIsInstance(self.evaluator.model_actuals, dict)
    
    def test_add_prediction(self):
        """Test adding predictions."""
        self.evaluator.add_prediction("model1", 100.0, 101.0, 0.8)
        
        self.assertIn("model1", self.evaluator.model_predictions)
        self.assertEqual(len(self.evaluator.model_predictions["model1"]), 1)
        self.assertEqual(self.evaluator.model_predictions["model1"][0], 100.0)
    
    def test_get_performance_snapshot(self):
        """Test getting performance snapshot."""
        # Add some predictions
        for i in range(10):
            self.evaluator.add_prediction("model1", 100.0 + i, 101.0 + i, 0.8)
        
        snapshot = self.evaluator.get_performance_snapshot("model1")
        
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.model_id, "model1")
        self.assertGreater(snapshot.prediction_count, 0)
        self.assertIsInstance(snapshot.mae, float)
        self.assertIsInstance(snapshot.rmse, float)
    
    def test_get_all_performance_snapshots(self):
        """Test getting all performance snapshots."""
        # Add predictions for multiple models
        self.evaluator.add_prediction("model1", 100.0, 101.0, 0.8)
        self.evaluator.add_prediction("model2", 200.0, 201.0, 0.9)
        
        snapshots = self.evaluator.get_all_performance_snapshots()
        
        self.assertIn("model1", snapshots)
        self.assertIn("model2", snapshots)
        self.assertEqual(snapshots["model1"].model_id, "model1")
        self.assertEqual(snapshots["model2"].model_id, "model2")


class TestModelSwitchingEngine(unittest.TestCase):
    """Test cases for ModelSwitchingEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ModelSwitchingEngine(cooldown_minutes=30)
        self.test_model1 = MockModel("model1")
        self.test_model2 = MockModel("model2")
    
    def test_initialization(self):
        """Test ModelSwitchingEngine initialization."""
        self.assertEqual(self.engine.cooldown_minutes, 30)
        self.assertIsInstance(self.engine.switching_rules, dict)
        self.assertIsInstance(self.engine.performance_evaluator, PerformanceEvaluator)
        self.assertIsNone(self.engine.active_model_id)
    
    def test_register_model(self):
        """Test model registration."""
        self.engine.register_model("model1", self.test_model1)
        
        self.assertIn("model1", self.engine.available_models)
        self.assertEqual(self.engine.available_models["model1"], self.test_model1)
    
    def test_get_active_model(self):
        """Test getting active model."""
        # Initially no active model
        self.assertIsNone(self.engine.get_active_model())
        
        # Register and set active model
        self.engine.register_model("model1", self.test_model1)
        self.engine.active_model_id = "model1"
        
        active_model = self.engine.get_active_model()
        self.assertEqual(active_model, self.test_model1)
    
    def test_manual_switch(self):
        """Test manual model switching."""
        # Register models
        self.engine.register_model("model1", self.test_model1)
        self.engine.register_model("model2", self.test_model2)
        
        # Set initial active model
        self.engine.active_model_id = "model1"
        
        # Switch to model2
        success = self.engine.manual_switch("model2", "Test switch")
        
        self.assertTrue(success)
        self.assertEqual(self.engine.active_model_id, "model2")
    
    def test_add_prediction(self):
        """Test adding predictions."""
        self.engine.register_model("model1", self.test_model1)
        self.engine.add_prediction("model1", 100.0, 101.0, 0.8)
        
        # Check that prediction was added to evaluator
        snapshot = self.engine.get_model_performance("model1")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.model_id, "model1")


class TestModelPerformanceSnapshot(unittest.TestCase):
    """Test cases for ModelPerformanceSnapshot class."""
    
    def test_initialization(self):
        """Test ModelPerformanceSnapshot initialization."""
        timestamp = datetime.now()
        snapshot = ModelPerformanceSnapshot(
            model_id="test_model",
            timestamp=timestamp,
            mae=0.1,
            rmse=0.15,
            directional_accuracy=75.0,
            sharpe_ratio=1.2,
            confidence_score=0.8,
            prediction_count=100
        )
        
        self.assertEqual(snapshot.model_id, "test_model")
        self.assertEqual(snapshot.timestamp, timestamp)
        self.assertEqual(snapshot.mae, 0.1)
        self.assertEqual(snapshot.rmse, 0.15)
        self.assertEqual(snapshot.directional_accuracy, 75.0)
        self.assertEqual(snapshot.sharpe_ratio, 1.2)
        self.assertEqual(snapshot.confidence_score, 0.8)
        self.assertEqual(snapshot.prediction_count, 100)
    
    def test_to_dict(self):
        """Test converting snapshot to dictionary."""
        timestamp = datetime.now()
        snapshot = ModelPerformanceSnapshot(
            model_id="test_model",
            timestamp=timestamp,
            mae=0.1,
            rmse=0.15,
            directional_accuracy=75.0,
            sharpe_ratio=1.2,
            confidence_score=0.8,
            prediction_count=100
        )
        
        snapshot_dict = snapshot.to_dict()
        
        self.assertIsInstance(snapshot_dict, dict)
        self.assertEqual(snapshot_dict["model_id"], "test_model")
        self.assertEqual(snapshot_dict["mae"], 0.1)
        self.assertEqual(snapshot_dict["rmse"], 0.15)
        self.assertEqual(snapshot_dict["directional_accuracy"], 75.0)
        self.assertEqual(snapshot_dict["sharpe_ratio"], 1.2)
        self.assertEqual(snapshot_dict["confidence_score"], 0.8)
        self.assertEqual(snapshot_dict["prediction_count"], 100)


class TestSwitchingRule(unittest.TestCase):
    """Test cases for SwitchingRule class."""
    
    def test_initialization(self):
        """Test SwitchingRule initialization."""
        rule = SwitchingRule(
            rule_id="test_rule",
            name="Test Rule",
            trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
            strategy=SwitchingStrategy.BEST_PERFORMER,
            metric_name="mae",
            threshold_value=0.1,
            comparison_operator=">",
            evaluation_window_minutes=60,
            min_observations=10
        )
        
        self.assertEqual(rule.rule_id, "test_rule")
        self.assertEqual(rule.name, "Test Rule")
        self.assertEqual(rule.trigger, SwitchingTrigger.PERFORMANCE_DEGRADATION)
        self.assertEqual(rule.strategy, SwitchingStrategy.BEST_PERFORMER)
        self.assertEqual(rule.metric_name, "mae")
        self.assertEqual(rule.threshold_value, 0.1)
        self.assertEqual(rule.comparison_operator, ">")
        self.assertEqual(rule.evaluation_window_minutes, 60)
        self.assertEqual(rule.min_observations, 10)
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.priority, 1)
    
    def test_evaluate(self):
        """Test rule evaluation."""
        rule = SwitchingRule(
            rule_id="test_rule",
            name="Test Rule",
            trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
            strategy=SwitchingStrategy.BEST_PERFORMER,
            metric_name="mae",
            threshold_value=0.1,
            comparison_operator=">",
            evaluation_window_minutes=60,
            min_observations=10
        )
        
        # Create mock performance snapshots
        current_performance = {
            "model1": ModelPerformanceSnapshot(
                model_id="model1",
                timestamp=datetime.now(),
                mae=0.15,  # Above threshold
                rmse=0.2,
                directional_accuracy=70.0,
                sharpe_ratio=1.0,
                confidence_score=0.8,
                prediction_count=20
            ),
            "model2": ModelPerformanceSnapshot(
                model_id="model2",
                timestamp=datetime.now(),
                mae=0.05,  # Below threshold
                rmse=0.1,
                directional_accuracy=80.0,
                sharpe_ratio=1.5,
                confidence_score=0.9,
                prediction_count=20
            )
        }
        
        should_switch, recommended_model = rule.evaluate(current_performance, "model1")
        
        # Should switch because model1 MAE (0.15) > threshold (0.1)
        self.assertTrue(should_switch)
        # Should recommend model2 as it has better MAE
        self.assertEqual(recommended_model, "model2")


if __name__ == "__main__":
    unittest.main()
