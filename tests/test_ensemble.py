"""
Unit tests for ensemble prediction system.
"""

import unittest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from enhanced_timeseries.models.ensemble import (
    EnsemblePredictionSystem, EnsembleWeighting, StackingEnsemble, BlendingEnsemble,
    EnsembleConfig, create_ensemble_system
)
from enhanced_timeseries.core.interfaces import BaseModel, PredictionResult


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, input_dim: int, prediction_value: float = 0.0, uncertainty_value: float = 0.1):
        super().__init__(input_dim)
        self.prediction_value = prediction_value
        self.uncertainty_value = uncertainty_value
        self.is_trained = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.full((batch_size, 1), self.prediction_value)
    
    def predict_with_uncertainty(self, x) -> tuple:
        if isinstance(x, pd.DataFrame):
            batch_size = len(x)
        else:
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
        
        prediction = torch.full((batch_size, 1), self.prediction_value)
        uncertainty = torch.full((batch_size, 1), self.uncertainty_value)
        
        return prediction, uncertainty
    
    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            batch_size = len(x)
        else:
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
        
        return torch.full((batch_size, 1), self.prediction_value)
    
    def train(self, data, **kwargs):
        return {'mae': abs(self.prediction_value) * 0.1}


class TestEnsembleWeighting(unittest.TestCase):
    """Test ensemble weighting strategies."""
    
    def test_equal_weights(self):
        """Test equal weighting."""
        weights = EnsembleWeighting.equal_weights(5)
        
        self.assertEqual(len(weights), 5)
        np.testing.assert_array_almost_equal(weights, [0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(np.sum(weights), 1.0)
    
    def test_performance_weights_inverse_error(self):
        """Test performance-based weighting with inverse error."""
        performances = np.array([0.1, 0.2, 0.05, 0.3])
        weights = EnsembleWeighting.performance_weights(performances, method='inverse_error')
        
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        
        # Model with lowest error (0.05) should have highest weight
        best_model_idx = np.argmin(performances)
        self.assertEqual(np.argmax(weights), best_model_idx)
    
    def test_performance_weights_exponential(self):
        """Test exponential performance weighting."""
        performances = np.array([0.1, 0.2, 0.05])
        weights = EnsembleWeighting.performance_weights(performances, method='exponential')
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))
    
    def test_performance_weights_softmax(self):
        """Test softmax performance weighting."""
        performances = np.array([0.1, 0.2, 0.05])
        weights = EnsembleWeighting.performance_weights(performances, method='softmax')
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))
    
    def test_volatility_adjusted_weights(self):
        """Test volatility-adjusted weighting."""
        performances = np.array([0.1, 0.2, 0.05])
        volatilities = np.array([0.15, 0.25, 0.1])
        
        weights = EnsembleWeighting.volatility_adjusted_weights(performances, volatilities)
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))
    
    def test_time_decay_weights(self):
        """Test time-decaying weights."""
        performances = np.array([0.1, 0.2, 0.05, 0.15])
        weights = EnsembleWeighting.time_decay_weights(performances, decay_factor=0.9)
        
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))


class TestStackingEnsemble(unittest.TestCase):
    """Test stacking ensemble."""
    
    def test_linear_stacking(self):
        """Test linear stacking ensemble."""
        n_models = 3
        stacking = StackingEnsemble(n_models, meta_learner_type='linear')
        
        # Test input
        predictions = torch.randn(4, n_models)  # batch_size=4
        
        # Forward pass
        output = stacking(predictions)
        
        self.assertEqual(output.shape, (4, 1))
    
    def test_mlp_stacking(self):
        """Test MLP stacking ensemble."""
        n_models = 3
        stacking = StackingEnsemble(n_models, meta_learner_type='mlp', hidden_dim=32)
        
        # Test input
        predictions = torch.randn(4, n_models)
        
        # Forward pass
        output = stacking(predictions)
        
        self.assertEqual(output.shape, (4, 1))
    
    def test_invalid_meta_learner(self):
        """Test invalid meta-learner type."""
        with self.assertRaises(ValueError):
            StackingEnsemble(3, meta_learner_type='invalid')


class TestBlendingEnsemble(unittest.TestCase):
    """Test blending ensemble."""
    
    def test_linear_blending(self):
        """Test linear blending."""
        blending = BlendingEnsemble(blend_method='linear')
        
        # Create sample data
        predictions = np.random.randn(100, 3)  # 100 samples, 3 models
        targets = np.random.randn(100)
        
        # Fit and predict
        blending.fit(predictions, targets)
        
        test_predictions = np.random.randn(10, 3)
        output = blending.predict(test_predictions)
        
        self.assertEqual(len(output), 10)
        self.assertTrue(blending.is_fitted)
    
    def test_ridge_blending(self):
        """Test Ridge blending."""
        blending = BlendingEnsemble(blend_method='ridge')
        
        predictions = np.random.randn(50, 3)
        targets = np.random.randn(50)
        
        blending.fit(predictions, targets)
        
        test_predictions = np.random.randn(5, 3)
        output = blending.predict(test_predictions)
        
        self.assertEqual(len(output), 5)
    
    def test_predict_without_fit(self):
        """Test prediction without fitting."""
        blending = BlendingEnsemble()
        
        with self.assertRaises(ValueError):
            blending.predict(np.random.randn(5, 3))
    
    def test_invalid_blend_method(self):
        """Test invalid blend method."""
        with self.assertRaises(ValueError):
            blending = BlendingEnsemble(blend_method='invalid')
            blending.fit(np.random.randn(10, 3), np.random.randn(10))


class TestEnsemblePredictionSystem(unittest.TestCase):
    """Test ensemble prediction system."""
    
    def setUp(self):
        """Set up test models and data."""
        # Create mock models with different prediction values
        self.models = [
            MockModel(input_dim=10, prediction_value=0.1, uncertainty_value=0.05),
            MockModel(input_dim=10, prediction_value=0.2, uncertainty_value=0.08),
            MockModel(input_dim=10, prediction_value=0.15, uncertainty_value=0.06)
        ]
        
        # Create sample data
        self.data = pd.DataFrame(np.random.randn(100, 10))
        self.validation_data = pd.DataFrame(np.random.randn(50, 10))
    
    def test_ensemble_creation(self):
        """Test ensemble system creation."""
        ensemble = EnsemblePredictionSystem(
            models=self.models,
            weighting_method='performance',
            combination_method='weighted_average'
        )
        
        self.assertEqual(len(ensemble.models), 3)
        self.assertEqual(ensemble.weighting_method, 'performance')
        self.assertEqual(ensemble.combination_method, 'weighted_average')
        self.assertFalse(ensemble.is_trained)
    
    def test_ensemble_training(self):
        """Test ensemble training."""
        ensemble = EnsemblePredictionSystem(self.models)
        
        # Train ensemble
        metrics = ensemble.train(self.data)
        
        self.assertTrue(ensemble.is_trained)
        self.assertIn('ensemble_mae', metrics)
        self.assertIn('n_models', metrics)
        self.assertEqual(metrics['n_models'], 3)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        ensemble = EnsemblePredictionSystem(self.models)
        ensemble.train(self.data)
        
        # Make prediction
        result = ensemble.predict(self.data.head(1))
        
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.symbol, 'ensemble')
        self.assertIsInstance(result.prediction, float)
        self.assertIsInstance(result.uncertainty, float)
        self.assertTrue(result.uncertainty >= 0)
        
        # Check model contributions
        self.assertEqual(len(result.model_contributions), 3)
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        ensemble = EnsemblePredictionSystem(self.models)
        ensemble.train(self.data)
        
        # Create batch data
        data_dict = {
            'AAPL': self.data.head(10),
            'MSFT': self.data.head(10),
            'GOOGL': self.data.head(10)
        }
        
        # Make batch predictions
        results = ensemble.predict_batch(data_dict)
        
        self.assertEqual(len(results), 3)
        self.assertIn('AAPL', results)
        self.assertIn('MSFT', results)
        self.assertIn('GOOGL', results)
        
        for symbol, result in results.items():
            self.assertEqual(result.symbol, symbol)
            self.assertIsInstance(result.prediction, float)
    
    def test_performance_update(self):
        """Test performance tracking update."""
        ensemble = EnsemblePredictionSystem(self.models)
        ensemble.train(self.data)
        
        # Update performance
        predictions = [0.1, 0.2, 0.15]
        actual = 0.12
        
        ensemble.update_performance(predictions, actual)
        
        # Check that performance was recorded
        for i in range(3):
            self.assertEqual(len(ensemble.model_predictions_history[i]), 1)
        
        self.assertEqual(len(ensemble.actual_values_history), 1)
    
    def test_different_combination_methods(self):
        """Test different combination methods."""
        combination_methods = ['weighted_average', 'uncertainty_weighted']
        
        for method in combination_methods:
            ensemble = EnsemblePredictionSystem(
                models=self.models,
                combination_method=method
            )
            ensemble.train(self.data)
            
            result = ensemble.predict(self.data.head(1))
            
            self.assertIsInstance(result.prediction, float)
            self.assertIsInstance(result.uncertainty, float)
    
    def test_stacking_ensemble(self):
        """Test stacking combination method."""
        ensemble = EnsemblePredictionSystem(
            models=self.models,
            combination_method='stacking'
        )
        
        # Check that stacking ensemble was created
        self.assertIsNotNone(ensemble.stacking_ensemble)
        
        # Train with validation data
        ensemble.train(self.data, validation_data=self.validation_data)
        
        result = ensemble.predict(self.data.head(1))
        self.assertIsInstance(result.prediction, float)
    
    def test_blending_ensemble(self):
        """Test blending combination method."""
        ensemble = EnsemblePredictionSystem(
            models=self.models,
            combination_method='blending'
        )
        
        # Check that blending ensemble was created
        self.assertIsNotNone(ensemble.blending_ensemble)
        
        # Train with validation data
        ensemble.train(self.data, validation_data=self.validation_data)
        
        result = ensemble.predict(self.data.head(1))
        self.assertIsInstance(result.prediction, float)
    
    def test_weight_updates(self):
        """Test weight updates based on performance."""
        ensemble = EnsemblePredictionSystem(
            models=self.models,
            weighting_method='performance',
            min_samples=5
        )
        ensemble.train(self.data)
        
        # Initial weights should be equal
        initial_weights = ensemble.current_weights.copy()
        np.testing.assert_array_almost_equal(initial_weights, [1/3, 1/3, 1/3])
        
        # Update performance multiple times
        for i in range(10):
            # Make model 0 perform better
            predictions = [0.01, 0.1, 0.1]  # Model 0 has lower error
            actual = 0.0
            ensemble.update_performance(predictions, actual)
        
        # Weights should have changed (model 0 should have higher weight)
        updated_weights = ensemble.current_weights
        self.assertGreater(updated_weights[0], updated_weights[1])
        self.assertGreater(updated_weights[0], updated_weights[2])
    
    def test_ensemble_info(self):
        """Test ensemble information retrieval."""
        ensemble = EnsemblePredictionSystem(self.models)
        
        info = ensemble.get_ensemble_info()
        
        self.assertIn('n_models', info)
        self.assertIn('weighting_method', info)
        self.assertIn('combination_method', info)
        self.assertIn('current_weights', info)
        self.assertIn('model_types', info)
        
        self.assertEqual(info['n_models'], 3)
        self.assertEqual(len(info['model_types']), 3)
    
    def test_model_weights_retrieval(self):
        """Test model weights retrieval."""
        ensemble = EnsemblePredictionSystem(self.models)
        
        weights = ensemble.get_model_weights()
        
        self.assertEqual(len(weights), 3)
        self.assertIn('model_0', weights)
        self.assertIn('model_1', weights)
        self.assertIn('model_2', weights)
        
        # Weights should sum to 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0)
    
    def test_performance_history(self):
        """Test performance history retrieval."""
        ensemble = EnsemblePredictionSystem(self.models)
        
        history = ensemble.get_performance_history()
        
        self.assertEqual(len(history), 3)
        for i in range(3):
            self.assertIn(i, history)
            self.assertIsInstance(history[i], list)


class TestEnsembleConfig(unittest.TestCase):
    """Test ensemble configuration."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = EnsembleConfig(
            weighting_method='performance',
            combination_method='stacking',
            adaptation_window=200
        )
        
        self.assertEqual(config.weighting_method, 'performance')
        self.assertEqual(config.combination_method, 'stacking')
        self.assertEqual(config.adaptation_window, 200)
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = EnsembleConfig(
            weighting_method='exponential',
            combination_method='blending'
        )
        
        config_dict = config.to_dict()
        
        self.assertIn('weighting_method', config_dict)
        self.assertIn('combination_method', config_dict)
        self.assertEqual(config_dict['weighting_method'], 'exponential')
        self.assertEqual(config_dict['combination_method'], 'blending')
    
    def test_create_ensemble_system(self):
        """Test ensemble system creation from config."""
        config = EnsembleConfig(weighting_method='performance')
        models = [MockModel(10) for _ in range(2)]
        
        ensemble = create_ensemble_system(models, config)
        
        self.assertIsInstance(ensemble, EnsemblePredictionSystem)
        self.assertEqual(ensemble.weighting_method, 'performance')
        self.assertEqual(len(ensemble.models), 2)


if __name__ == '__main__':
    unittest.main()