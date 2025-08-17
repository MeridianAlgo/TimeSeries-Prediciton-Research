"""
Unit tests for Monte Carlo uncertainty quantification.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from enhanced_timeseries.uncertainty.monte_carlo import (
    MCDropout, ConcreteDropout, BayesianLinear, UncertaintyQuantifier,
    UncertaintyCalibration, UncertaintyMetrics, create_mc_dropout_model
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


class BayesianModel(nn.Module):
    """Bayesian model for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)
    
    def kl_loss(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()


class TestMCDropout(unittest.TestCase):
    """Test Monte Carlo Dropout."""
    
    def test_mc_dropout_forward(self):
        """Test MC Dropout forward pass."""
        dropout = MCDropout(p=0.5)
        x = torch.randn(4, 10)
        
        # Test in training mode (should apply dropout)
        dropout.train()
        output = dropout(x)
        self.assertEqual(output.shape, x.shape)
        
        # Test in eval mode (should still apply dropout for MC)
        dropout.eval()
        output = dropout(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_mc_dropout_stochasticity(self):
        """Test that MC Dropout produces different outputs."""
        dropout = MCDropout(p=0.5)
        x = torch.randn(4, 10)
        
        # Generate multiple outputs
        outputs = []
        for _ in range(10):
            output = dropout(x)
            outputs.append(output)
        
        # Check that outputs are different (with high probability)
        outputs = torch.stack(outputs)
        variance = torch.var(outputs, dim=0)
        
        # Should have some variance due to dropout
        self.assertTrue(torch.any(variance > 0))


class TestConcreteDropout(unittest.TestCase):
    """Test Concrete Dropout."""
    
    def test_concrete_dropout_forward(self):
        """Test Concrete Dropout forward pass."""
        base_layer = nn.Linear(10, 5)
        concrete_dropout = ConcreteDropout(base_layer, input_dim=10)
        
        x = torch.randn(4, 10)
        output = concrete_dropout(x)
        
        self.assertEqual(output.shape, (4, 5))
    
    def test_regularization_loss(self):
        """Test regularization loss calculation."""
        base_layer = nn.Linear(10, 5)
        concrete_dropout = ConcreteDropout(base_layer, input_dim=10)
        
        reg_loss = concrete_dropout.regularization_loss()
        
        self.assertIsInstance(reg_loss, torch.Tensor)
        self.assertEqual(reg_loss.shape, ())  # Scalar
        self.assertTrue(reg_loss.item() >= 0)  # Should be non-negative


class TestBayesianLinear(unittest.TestCase):
    """Test Bayesian Linear layer."""
    
    def test_bayesian_linear_forward(self):
        """Test Bayesian Linear forward pass."""
        layer = BayesianLinear(10, 5)
        x = torch.randn(4, 10)
        
        output = layer(x)
        self.assertEqual(output.shape, (4, 5))
    
    def test_bayesian_linear_stochasticity(self):
        """Test that Bayesian Linear produces different outputs."""
        layer = BayesianLinear(10, 5)
        x = torch.randn(4, 10)
        
        # Generate multiple outputs
        outputs = []
        for _ in range(10):
            output = layer(x)
            outputs.append(output)
        
        # Check that outputs are different
        outputs = torch.stack(outputs)
        variance = torch.var(outputs, dim=0)
        
        # Should have variance due to weight sampling
        self.assertTrue(torch.any(variance > 0))
    
    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        layer = BayesianLinear(10, 5)
        
        kl_div = layer.kl_divergence()
        
        self.assertIsInstance(kl_div, torch.Tensor)
        self.assertEqual(kl_div.shape, ())  # Scalar
        self.assertTrue(kl_div.item() >= 0)  # Should be non-negative


class TestUncertaintyQuantifier(unittest.TestCase):
    """Test Uncertainty Quantifier."""
    
    def setUp(self):
        """Set up test models and data."""
        self.model = SimpleModel(input_dim=10)
        self.bayesian_model = BayesianModel(input_dim=10)
        self.x = torch.randn(4, 10)
        self.quantifier = UncertaintyQuantifier(self.model)
    
    def test_monte_carlo_predict(self):
        """Test Monte Carlo prediction."""
        mean_pred, var_pred = self.quantifier.monte_carlo_predict(self.x, n_samples=20)
        
        self.assertEqual(mean_pred.shape, (4, 1))
        self.assertEqual(var_pred.shape, (4, 1))
        self.assertTrue(torch.all(var_pred >= 0))  # Variance should be non-negative
    
    def test_deep_ensemble_predict(self):
        """Test deep ensemble prediction."""
        # Create multiple models
        models = [SimpleModel(input_dim=10) for _ in range(3)]
        
        mean_pred, var_pred = self.quantifier.deep_ensemble_predict(self.x, models)
        
        self.assertEqual(mean_pred.shape, (4, 1))
        self.assertEqual(var_pred.shape, (4, 1))
        self.assertTrue(torch.all(var_pred >= 0))
    
    def test_bayesian_predict(self):
        """Test Bayesian prediction."""
        bayesian_quantifier = UncertaintyQuantifier(self.bayesian_model)
        
        mean_pred, var_pred = bayesian_quantifier.bayesian_predict(self.x, n_samples=20)
        
        self.assertEqual(mean_pred.shape, (4, 1))
        self.assertEqual(var_pred.shape, (4, 1))
        self.assertTrue(torch.all(var_pred >= 0))
    
    def test_temperature_scaling_predict(self):
        """Test temperature scaling prediction."""
        pred, uncertainty = self.quantifier.temperature_scaling_predict(self.x, temperature=1.5)
        
        self.assertEqual(pred.shape, (4, 1))
        self.assertEqual(uncertainty.shape, (4, 1))
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_quantile_predict(self):
        """Test quantile prediction."""
        quantiles = [0.1, 0.5, 0.9]
        results = self.quantifier.quantile_predict(self.x, quantiles=quantiles)
        
        self.assertEqual(len(results), len(quantiles))
        
        for q in quantiles:
            key = f'quantile_{q}'
            self.assertIn(key, results)
            self.assertEqual(results[key].shape, (4, 1))
        
        # Check ordering: lower quantile <= median <= upper quantile
        q10 = results['quantile_0.1']
        q50 = results['quantile_0.5']
        q90 = results['quantile_0.9']
        
        self.assertTrue(torch.all(q10 <= q50))
        self.assertTrue(torch.all(q50 <= q90))
    
    def test_epistemic_aleatoric_split(self):
        """Test epistemic/aleatoric uncertainty split."""
        mean_pred, epistemic, aleatoric = self.quantifier.epistemic_aleatoric_split(self.x, n_samples=20)
        
        self.assertEqual(mean_pred.shape, (4, 1))
        self.assertEqual(epistemic.shape, (4, 1))
        self.assertEqual(aleatoric.shape, (4, 1))
        
        self.assertTrue(torch.all(epistemic >= 0))
        self.assertTrue(torch.all(aleatoric >= 0))


class TestUncertaintyCalibration(unittest.TestCase):
    """Test Uncertainty Calibration."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.uncertainties = np.random.uniform(0, 1, 100)
        self.errors = np.random.uniform(0, 1, 100)
        self.calibration = UncertaintyCalibration()
    
    def test_isotonic_calibration(self):
        """Test isotonic calibration."""
        self.calibration.fit_calibration(self.uncertainties, self.errors, method='isotonic')
        
        self.assertTrue(self.calibration.is_calibrated)
        
        # Test calibration
        calibrated = self.calibration.calibrate_uncertainty(self.uncertainties)
        
        self.assertEqual(len(calibrated), len(self.uncertainties))
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
    
    def test_platt_calibration(self):
        """Test Platt scaling calibration."""
        self.calibration.fit_calibration(self.uncertainties, self.errors, method='platt')
        
        self.assertTrue(self.calibration.is_calibrated)
        
        # Test calibration
        calibrated = self.calibration.calibrate_uncertainty(self.uncertainties)
        
        self.assertEqual(len(calibrated), len(self.uncertainties))
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
    
    def test_calibration_without_fit(self):
        """Test calibration without fitting."""
        with self.assertRaises(ValueError):
            self.calibration.calibrate_uncertainty(self.uncertainties)
    
    def test_invalid_calibration_method(self):
        """Test invalid calibration method."""
        with self.assertRaises(ValueError):
            self.calibration.fit_calibration(self.uncertainties, self.errors, method='invalid')


class TestUncertaintyMetrics(unittest.TestCase):
    """Test Uncertainty Metrics."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.uncertainties = np.random.uniform(0, 1, 100)
        self.errors = np.random.uniform(0, 1, 100)
        self.predictions = np.random.randn(100)
        self.actuals = self.predictions + np.random.randn(100) * 0.1
    
    def test_calibration_error(self):
        """Test Expected Calibration Error."""
        ece = UncertaintyMetrics.calibration_error(self.uncertainties, self.errors)
        
        self.assertIsInstance(ece, float)
        self.assertTrue(0 <= ece <= 1)
    
    def test_reliability_diagram(self):
        """Test reliability diagram generation."""
        bin_centers, accuracies, confidences = UncertaintyMetrics.reliability_diagram(
            self.uncertainties, self.errors, n_bins=5
        )
        
        self.assertEqual(len(bin_centers), 5)
        self.assertEqual(len(accuracies), 5)
        self.assertEqual(len(confidences), 5)
        
        self.assertTrue(np.all(accuracies >= 0))
        self.assertTrue(np.all(accuracies <= 1))
        self.assertTrue(np.all(confidences >= 0))
        self.assertTrue(np.all(confidences <= 1))
    
    def test_sharpness(self):
        """Test sharpness calculation."""
        sharpness = UncertaintyMetrics.sharpness(self.uncertainties)
        
        self.assertIsInstance(sharpness, float)
        self.assertTrue(sharpness >= 0)
        
        # Should be approximately the mean
        expected_sharpness = np.mean(self.uncertainties)
        self.assertAlmostEqual(sharpness, expected_sharpness, places=6)
    
    def test_coverage_probability(self):
        """Test coverage probability calculation."""
        coverage = UncertaintyMetrics.coverage_probability(
            self.predictions, self.uncertainties, self.actuals, confidence_level=0.95
        )
        
        self.assertIsInstance(coverage, float)
        self.assertTrue(0 <= coverage <= 1)


class TestCreateMCDropoutModel(unittest.TestCase):
    """Test MC Dropout model creation."""
    
    def test_create_mc_dropout_model(self):
        """Test creating MC Dropout model from regular model."""
        base_model = SimpleModel(input_dim=10)
        mc_model = create_mc_dropout_model(base_model, dropout_rate=0.3)
        
        # Check that model structure is preserved
        self.assertEqual(mc_model.input_dim, base_model.input_dim)
        
        # Test forward pass
        x = torch.randn(4, 10)
        
        base_output = base_model(x)
        mc_output = mc_model(x)
        
        self.assertEqual(base_output.shape, mc_output.shape)
        
        # Check that MC model produces different outputs (stochastic)
        mc_outputs = []
        for _ in range(10):
            output = mc_model(x)
            mc_outputs.append(output)
        
        mc_outputs = torch.stack(mc_outputs)
        variance = torch.var(mc_outputs, dim=0)
        
        # Should have some variance due to MC dropout
        self.assertTrue(torch.any(variance > 0))


if __name__ == '__main__':
    unittest.main()