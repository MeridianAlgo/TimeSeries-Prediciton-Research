"""
Unit tests for ensemble prediction framework.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from enhanced_timeseries.ensemble.ensemble_framework import (
    BaseModel, ModelWrapper, PerformanceTracker, WeightingStrategy,
    PerformanceBasedWeighting, TrendAwareWeighting, EnsembleMethod,
    WeightedAveraging, StackingEnsemble, EnsemblePredictor,
    create_ensemble_predictor, evaluate_ensemble_performance
)


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, name: str, prediction_value: float = 1.0, uncertainty_value: float = 0.1):
        self.name = name
        self.prediction_value = prediction_value
        self.uncertainty_value = uncertainty_value
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.full((batch_size, 1), self.prediction_value)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        pred = self.predict(x)
        uncertainty = torch.full_like(pred, self.uncertainty_value)
        return pred, uncertainty
    
    def get_model_name(self) -> str:
        return self.name


class SimpleNN(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


class TestModelWrapper(unittest.TestCase):
    """Test ModelWrapper class."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = SimpleNN(input_dim=5, output_dim=1)
        self.wrapper = ModelWrapper(self.model, "test_model")
    
    def test_wrapper_creation(self):
        """Test model wrapper creation."""
        self.assertEqual(self.wrapper.get_model_name(), "test_model")
        self.assertEqual(self.wrapper.uncertainty_method, "dropout")
        self.assertEqual(self.wrapper.n_samples, 100)
    
    def test_predict(self):
        """Test prediction functionality."""
        batch_size = 4
        x = torch.randn(batch_size, 5)
        
        pred = self.wrapper.predict(x)
        
        self.assertEqual(pred.shape, (batch_size, 1))
        self.assertIsInstance(pred, torch.Tensor)
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        batch_size = 2
        x = torch.randn(batch_size, 5)
        
        pred, uncertainty = self.wrapper.predict_with_uncertainty(x)
        
        self.assertEqual(pred.shape, (batch_size, 1))
        self.assertEqual(uncertainty.shape, (batch_size, 1))
        self.assertTrue(torch.all(uncertainty >= 0))  # Uncertainty should be non-negative
    
    def test_monte_carlo_dropout(self):
        """Test Monte Carlo dropout uncertainty estimation."""
        wrapper = ModelWrapper(self.model, "test_model", n_samples=10)
        
        batch_size = 2
        x = torch.randn(batch_size, 5)
        
        pred, uncertainty = wrapper.predict_with_uncertainty(x)
        
        self.assertEqual(pred.shape, (batch_size, 1))
        self.assertEqual(uncertainty.shape, (batch_size, 1))


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.tracker = PerformanceTracker(window_size=10, metrics=['mse', 'mae'])
    
    def test_tracker_creation(self):
        """Test performance tracker creation."""
        self.assertEqual(self.tracker.window_size, 10)
        self.assertEqual(self.tracker.metrics, ['mse', 'mae'])
    
    def test_update_performance(self):
        """Test performance update."""
        predictions = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[1.1], [1.9], [3.2]])
        
        metrics = self.tracker.update_performance("model1", predictions, targets)
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_get_recent_performance(self):
        """Test getting recent performance."""
        # Add some performance data
        for i in range(5):
            pred = torch.tensor([[float(i)]])
            target = torch.tensor([[float(i) + 0.1]])
            self.tracker.update_performance("model1", pred, target)
        
        recent_mse = self.tracker.get_recent_performance("model1", "mse", n_recent=3)
        
        self.assertIsInstance(recent_mse, float)
        self.assertGreater(recent_mse, 0)
    
    def test_get_performance_trend(self):
        """Test performance trend calculation."""
        # Add performance data with improving trend
        for i in range(10):
            error = 1.0 - i * 0.1  # Decreasing error
            pred = torch.tensor([[0.0]])
            target = torch.tensor([[error]])
            self.tracker.update_performance("model1", pred, target)
        
        trend = self.tracker.get_performance_trend("model1", "mse")
        
        self.assertIsInstance(trend, float)
        # Trend should be negative (improving)
        self.assertLess(trend, 0)
    
    def test_unknown_model_performance(self):
        """Test performance for unknown model."""
        perf = self.tracker.get_recent_performance("unknown_model", "mse")
        self.assertEqual(perf, float('inf'))


class TestPerformanceBasedWeighting(unittest.TestCase):
    """Test PerformanceBasedWeighting class."""
    
    def setUp(self):
        """Set up test environment."""
        self.weighting = PerformanceBasedWeighting(metric='mse', temperature=1.0)
        self.tracker = PerformanceTracker()
        
        # Create mock models
        self.models = [
            MockModel("model1", 1.0),
            MockModel("model2", 2.0),
            MockModel("model3", 3.0)
        ]
    
    def test_weighting_creation(self):
        """Test weighting strategy creation."""
        self.assertEqual(self.weighting.metric, 'mse')
        self.assertEqual(self.weighting.temperature, 1.0)
        self.assertEqual(self.weighting.min_weight, 0.01)
    
    def test_compute_weights_no_history(self):
        """Test weight computation with no performance history."""
        weights = self.weighting.compute_weights(self.models, self.tracker)
        
        # Should return equal weights
        expected_weight = 1.0 / len(self.models)
        for model in self.models:
            self.assertAlmostEqual(weights[model.get_model_name()], expected_weight, places=2)
    
    def test_compute_weights_with_history(self):
        """Test weight computation with performance history."""
        # Add performance data (model1 performs better)
        x = torch.randn(5, 10)
        
        # Model 1: low error
        pred1 = torch.ones(5, 1) * 1.0
        target = torch.ones(5, 1) * 1.05  # Small error
        self.tracker.update_performance("model1", pred1, target)
        
        # Model 2: high error
        pred2 = torch.ones(5, 1) * 2.0
        target = torch.ones(5, 1) * 1.0  # Large error
        self.tracker.update_performance("model2", pred2, target)
        
        weights = self.weighting.compute_weights(self.models, self.tracker)
        
        # Model 1 should have higher weight
        self.assertGreater(weights["model1"], weights["model2"])
        
        # Weights should sum to 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_minimum_weight_constraint(self):
        """Test minimum weight constraint."""
        weighting = PerformanceBasedWeighting(min_weight=0.1)
        
        # Add very different performance data
        pred_good = torch.ones(5, 1) * 1.0
        pred_bad = torch.ones(5, 1) * 10.0
        target = torch.ones(5, 1) * 1.0
        
        self.tracker.update_performance("model1", pred_good, target)
        self.tracker.update_performance("model2", pred_bad, target)
        
        weights = weighting.compute_weights(self.models[:2], self.tracker)
        
        # All weights should be at least min_weight
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.1)


class TestTrendAwareWeighting(unittest.TestCase):
    """Test TrendAwareWeighting class."""
    
    def setUp(self):
        """Set up test environment."""
        self.weighting = TrendAwareWeighting(trend_weight=0.3)
        self.tracker = PerformanceTracker()
        self.models = [MockModel("model1"), MockModel("model2")]
    
    def test_weighting_creation(self):
        """Test trend-aware weighting creation."""
        self.assertEqual(self.weighting.trend_weight, 0.3)
        self.assertEqual(self.weighting.metric, 'mse')
    
    def test_compute_weights_with_trends(self):
        """Test weight computation considering trends."""
        # Add performance data with different trends
        
        # Model 1: stable performance
        for i in range(10):
            pred = torch.ones(1, 1) * 1.0
            target = torch.ones(1, 1) * 1.1
            self.tracker.update_performance("model1", pred, target)
        
        # Model 2: improving performance
        for i in range(10):
            error = 1.0 - i * 0.05  # Improving
            pred = torch.ones(1, 1) * 0.0
            target = torch.ones(1, 1) * error
            self.tracker.update_performance("model2", pred, target)
        
        weights = self.weighting.compute_weights(self.models, self.tracker)
        
        # Model 2 should have higher weight due to improving trend
        self.assertGreater(weights["model2"], weights["model1"])


class TestWeightedAveraging(unittest.TestCase):
    """Test WeightedAveraging ensemble method."""
    
    def setUp(self):
        """Set up test environment."""
        self.ensemble_method = WeightedAveraging()
    
    def test_combine_predictions(self):
        """Test prediction combination."""
        predictions = {
            "model1": torch.tensor([[1.0], [2.0]]),
            "model2": torch.tensor([[2.0], [3.0]]),
            "model3": torch.tensor([[3.0], [4.0]])
        }
        
        weights = {"model1": 0.5, "model2": 0.3, "model3": 0.2}
        
        combined = self.ensemble_method.combine_predictions(predictions, weights)
        
        # Check shape
        self.assertEqual(combined.shape, (2, 1))
        
        # Check values (manual calculation)
        expected_0 = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0  # 1.7
        expected_1 = 0.5 * 2.0 + 0.3 * 3.0 + 0.2 * 4.0  # 2.7
        
        self.assertAlmostEqual(combined[0, 0].item(), expected_0, places=5)
        self.assertAlmostEqual(combined[1, 0].item(), expected_1, places=5)
    
    def test_combine_uncertainties(self):
        """Test uncertainty combination."""
        predictions = {
            "model1": torch.tensor([[1.0], [2.0]]),
            "model2": torch.tensor([[1.1], [2.1]])
        }
        
        uncertainties = {
            "model1": torch.tensor([[0.1], [0.1]]),
            "model2": torch.tensor([[0.2], [0.2]])
        }
        
        weights = {"model1": 0.6, "model2": 0.4}
        
        combined_unc = self.ensemble_method.combine_uncertainties(
            predictions, uncertainties, weights
        )
        
        # Check shape
        self.assertEqual(combined_unc.shape, (2, 1))
        
        # Uncertainty should be non-negative
        self.assertTrue(torch.all(combined_unc >= 0))


class TestStackingEnsemble(unittest.TestCase):
    """Test StackingEnsemble method."""
    
    def setUp(self):
        """Set up test environment."""
        self.ensemble_method = StackingEnsemble()
    
    def test_stacking_creation(self):
        """Test stacking ensemble creation."""
        self.assertIsNone(self.ensemble_method.meta_learner)
        self.assertFalse(self.ensemble_method.is_trained)
    
    def test_train_meta_learner(self):
        """Test meta-learner training."""
        predictions = {
            "model1": torch.randn(10, 1),
            "model2": torch.randn(10, 1),
            "model3": torch.randn(10, 1)
        }
        
        targets = torch.randn(10, 1)
        
        self.ensemble_method.train_meta_learner(predictions, targets)
        
        self.assertTrue(self.ensemble_method.is_trained)
        self.assertIsNotNone(self.ensemble_method.meta_learner)
    
    def test_combine_predictions_untrained(self):
        """Test prediction combination without training (fallback)."""
        predictions = {
            "model1": torch.tensor([[1.0]]),
            "model2": torch.tensor([[2.0]])
        }
        
        weights = {"model1": 0.6, "model2": 0.4}
        
        combined = self.ensemble_method.combine_predictions(predictions, weights)
        
        # Should fallback to weighted averaging
        expected = 0.6 * 1.0 + 0.4 * 2.0
        self.assertAlmostEqual(combined[0, 0].item(), expected, places=5)
    
    def test_combine_predictions_trained(self):
        """Test prediction combination with trained meta-learner."""
        # Train meta-learner first
        train_predictions = {
            "model1": torch.randn(20, 1),
            "model2": torch.randn(20, 1)
        }
        train_targets = torch.randn(20, 1)
        
        self.ensemble_method.train_meta_learner(train_predictions, train_targets)
        
        # Test prediction
        test_predictions = {
            "model1": torch.tensor([[1.0]]),
            "model2": torch.tensor([[2.0]])
        }
        
        weights = {"model1": 0.5, "model2": 0.5}
        
        combined = self.ensemble_method.combine_predictions(test_predictions, weights)
        
        self.assertEqual(combined.shape, (1, 1))


class TestEnsemblePredictor(unittest.TestCase):
    """Test EnsemblePredictor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.ensemble = EnsemblePredictor()
        
        # Add mock models
        self.models = [
            MockModel("model1", 1.0, 0.1),
            MockModel("model2", 2.0, 0.2),
            MockModel("model3", 3.0, 0.3)
        ]
        
        for model in self.models:
            self.ensemble.add_model(model)
    
    def test_ensemble_creation(self):
        """Test ensemble predictor creation."""
        self.assertEqual(len(self.ensemble.models), 3)
        self.assertIsNotNone(self.ensemble.weighting_strategy)
        self.assertIsNotNone(self.ensemble.ensemble_method)
        self.assertIsNotNone(self.ensemble.performance_tracker)
    
    def test_add_remove_models(self):
        """Test adding and removing models."""
        # Test adding
        new_model = MockModel("model4", 4.0)
        self.ensemble.add_model(new_model)
        self.assertEqual(len(self.ensemble.models), 4)
        
        # Test removing
        self.ensemble.remove_model("model2")
        self.assertEqual(len(self.ensemble.models), 3)
        
        # Check that model2 is gone
        model_names = [m.get_model_name() for m in self.ensemble.models]
        self.assertNotIn("model2", model_names)
    
    def test_predict(self):
        """Test ensemble prediction."""
        x = torch.randn(5, 10)
        
        pred = self.ensemble.predict(x)
        
        self.assertEqual(pred.shape, (5, 1))
        self.assertIsInstance(pred, torch.Tensor)
    
    def test_predict_with_individual(self):
        """Test prediction with individual model results."""
        x = torch.randn(3, 10)
        
        pred, individual = self.ensemble.predict(x, return_individual=True)
        
        self.assertEqual(pred.shape, (3, 1))
        self.assertIn('predictions', individual)
        self.assertIn('weights', individual)
        
        # Check individual predictions
        self.assertEqual(len(individual['predictions']), 3)
        self.assertEqual(len(individual['weights']), 3)
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        x = torch.randn(2, 10)
        
        pred, uncertainty = self.ensemble.predict_with_uncertainty(x)
        
        self.assertEqual(pred.shape, (2, 1))
        self.assertEqual(uncertainty.shape, (2, 1))
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_predict_with_uncertainty_individual(self):
        """Test uncertainty prediction with individual results."""
        x = torch.randn(2, 10)
        
        pred, uncertainty, individual = self.ensemble.predict_with_uncertainty(
            x, return_individual=True
        )
        
        self.assertEqual(pred.shape, (2, 1))
        self.assertEqual(uncertainty.shape, (2, 1))
        
        self.assertIn('predictions', individual)
        self.assertIn('uncertainties', individual)
        self.assertIn('weights', individual)
    
    def test_update_performance(self):
        """Test performance update."""
        x = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        
        # Should not raise any errors
        self.ensemble.update_performance(x, targets)
        
        # Check that performance was recorded
        summary = self.ensemble.get_performance_summary()
        self.assertEqual(len(summary), 3)
        
        for model_name in ["model1", "model2", "model3"]:
            self.assertIn(model_name, summary)
    
    def test_get_model_weights(self):
        """Test getting model weights."""
        weights = self.ensemble.get_model_weights()
        
        self.assertEqual(len(weights), 3)
        
        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_get_performance_summary(self):
        """Test performance summary."""
        # Add some performance data
        x = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        self.ensemble.update_performance(x, targets)
        
        summary = self.ensemble.get_performance_summary()
        
        self.assertEqual(len(summary), 3)
        
        for model_name, metrics in summary.items():
            self.assertIn('recent_mse', metrics)
            self.assertIn('recent_mae', metrics)
            self.assertIn('trend', metrics)
    
    def test_empty_ensemble_error(self):
        """Test error handling for empty ensemble."""
        empty_ensemble = EnsemblePredictor()
        x = torch.randn(2, 10)
        
        with self.assertRaises(ValueError):
            empty_ensemble.predict(x)
        
        with self.assertRaises(ValueError):
            empty_ensemble.predict_with_uncertainty(x)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_ensemble_predictor(self):
        """Test ensemble predictor creation utility."""
        models = [
            MockModel("model1", 1.0),
            MockModel("model2", 2.0)
        ]
        
        ensemble = create_ensemble_predictor(
            models, 
            weighting_method='performance',
            ensemble_method='averaging'
        )
        
        self.assertEqual(len(ensemble.models), 2)
        self.assertIsInstance(ensemble.weighting_strategy, PerformanceBasedWeighting)
        self.assertIsInstance(ensemble.ensemble_method, WeightedAveraging)
    
    def test_create_ensemble_predictor_trend_aware(self):
        """Test ensemble creation with trend-aware weighting."""
        models = [MockModel("model1")]
        
        ensemble = create_ensemble_predictor(
            models,
            weighting_method='trend_aware',
            ensemble_method='stacking'
        )
        
        self.assertIsInstance(ensemble.weighting_strategy, TrendAwareWeighting)
        self.assertIsInstance(ensemble.ensemble_method, StackingEnsemble)
    
    def test_create_ensemble_predictor_invalid_method(self):
        """Test error handling for invalid methods."""
        models = [MockModel("model1")]
        
        with self.assertRaises(ValueError):
            create_ensemble_predictor(models, weighting_method='invalid')
        
        with self.assertRaises(ValueError):
            create_ensemble_predictor(models, ensemble_method='invalid')
    
    def test_evaluate_ensemble_performance(self):
        """Test ensemble performance evaluation."""
        models = [
            MockModel("model1", 1.0),
            MockModel("model2", 1.1)
        ]
        
        ensemble = create_ensemble_predictor(models)
        
        # Create test data
        test_data = torch.randn(10, 5)
        test_targets = torch.ones(10, 1)  # Target value of 1.0
        
        results = evaluate_ensemble_performance(
            ensemble, test_data, test_targets, 
            metrics=['mse', 'mae', 'rmse']
        )
        
        self.assertIn('mse', results)
        self.assertIn('mae', results)
        self.assertIn('rmse', results)
        
        # All metrics should be positive
        for metric_value in results.values():
            self.assertGreater(metric_value, 0)
    
    def test_evaluate_ensemble_performance_mape(self):
        """Test ensemble evaluation with MAPE metric."""
        models = [MockModel("model1", 2.0)]
        ensemble = create_ensemble_predictor(models)
        
        test_data = torch.randn(5, 3)
        test_targets = torch.ones(5, 1) * 2.0  # Non-zero targets for MAPE
        
        results = evaluate_ensemble_performance(
            ensemble, test_data, test_targets, 
            metrics=['mape']
        )
        
        self.assertIn('mape', results)
        self.assertGreaterEqual(results['mape'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for ensemble framework."""
    
    def test_full_ensemble_workflow(self):
        """Test complete ensemble workflow."""
        # Create models with different behaviors
        models = [
            MockModel("conservative", 0.9, 0.05),  # Slightly underestimates
            MockModel("aggressive", 1.1, 0.1),    # Slightly overestimates
            MockModel("volatile", 1.0, 0.2)       # Correct mean, high uncertainty
        ]
        
        # Create ensemble
        ensemble = create_ensemble_predictor(
            models,
            weighting_method='trend_aware',
            ensemble_method='averaging',
            trend_weight=0.2
        )
        
        # Simulate training data updates
        for i in range(10):
            x = torch.randn(5, 10)
            targets = torch.ones(5, 1)  # True value is 1.0
            
            # Update performance
            ensemble.update_performance(x, targets)
        
        # Make predictions
        test_x = torch.randn(3, 10)
        pred, uncertainty = ensemble.predict_with_uncertainty(test_x)
        
        # Verify results
        self.assertEqual(pred.shape, (3, 1))
        self.assertEqual(uncertainty.shape, (3, 1))
        
        # Get performance summary
        summary = ensemble.get_performance_summary()
        self.assertEqual(len(summary), 3)
        
        # Conservative model should perform best (closest to 1.0)
        weights = ensemble.get_model_weights()
        self.assertIn("conservative", weights)
        self.assertIn("aggressive", weights)
        self.assertIn("volatile", weights)
    
    def test_dynamic_weight_adjustment(self):
        """Test that weights adjust based on performance."""
        # Create models with different performance patterns
        good_model = MockModel("good", 1.0, 0.1)
        bad_model = MockModel("bad", 2.0, 0.1)  # Always wrong
        
        ensemble = create_ensemble_predictor([good_model, bad_model])
        
        # Initial weights should be equal
        initial_weights = ensemble.get_model_weights()
        self.assertAlmostEqual(initial_weights["good"], initial_weights["bad"], places=1)
        
        # Update with performance data
        for _ in range(20):
            x = torch.randn(5, 10)
            targets = torch.ones(5, 1)  # True value is 1.0
            ensemble.update_performance(x, targets)
        
        # After updates, good model should have higher weight
        final_weights = ensemble.get_model_weights()
        self.assertGreater(final_weights["good"], final_weights["bad"])
    
    def test_ensemble_uncertainty_quantification(self):
        """Test uncertainty quantification in ensemble."""
        # Create models with different uncertainty levels
        certain_model = MockModel("certain", 1.0, 0.01)
        uncertain_model = MockModel("uncertain", 1.0, 0.5)
        
        ensemble = create_ensemble_predictor([certain_model, uncertain_model])
        
        x = torch.randn(5, 10)
        pred, uncertainty = ensemble.predict_with_uncertainty(x)
        
        # Ensemble uncertainty should be between individual uncertainties
        self.assertTrue(torch.all(uncertainty > 0))
        self.assertTrue(torch.all(uncertainty < 0.5))  # Less than max individual uncertainty


if __name__ == '__main__':
    unittest.main()