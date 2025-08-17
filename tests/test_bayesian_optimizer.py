"""
Unit tests for Bayesian optimization framework.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from enhanced_timeseries.optimization.bayesian_optimizer import (
    Parameter, Trial, ExpectedImprovement, UpperConfidenceBound,
    ProbabilityOfImprovement, GaussianProcess, BayesianOptimizer,
    MultiObjectiveBayesianOptimizer, HyperparameterTuner,
    create_parameter_space, optimize_model_hyperparameters
)


class TestParameter(unittest.TestCase):
    """Test Parameter class."""
    
    def test_continuous_parameter(self):
        """Test continuous parameter creation."""
        param = Parameter(
            name="learning_rate",
            param_type="continuous",
            bounds=(0.001, 0.1),
            log_scale=True
        )
        
        self.assertEqual(param.name, "learning_rate")
        self.assertEqual(param.param_type, "continuous")
        self.assertEqual(param.bounds, (0.001, 0.1))
        self.assertTrue(param.log_scale)
    
    def test_discrete_parameter(self):
        """Test discrete parameter creation."""
        param = Parameter(
            name="num_layers",
            param_type="discrete",
            choices=[1, 2, 3, 4, 5]
        )
        
        self.assertEqual(param.name, "num_layers")
        self.assertEqual(param.param_type, "discrete")
        self.assertEqual(param.choices, [1, 2, 3, 4, 5])
    
    def test_categorical_parameter(self):
        """Test categorical parameter creation."""
        param = Parameter(
            name="optimizer",
            param_type="categorical",
            choices=["adam", "sgd", "rmsprop"]
        )
        
        self.assertEqual(param.name, "optimizer")
        self.assertEqual(param.param_type, "categorical")
        self.assertEqual(param.choices, ["adam", "sgd", "rmsprop"])
    
    def test_invalid_continuous_parameter(self):
        """Test invalid continuous parameter (no bounds)."""
        with self.assertRaises(ValueError):
            Parameter(
                name="test",
                param_type="continuous"
                # Missing bounds
            )
    
    def test_invalid_discrete_parameter(self):
        """Test invalid discrete parameter (no choices)."""
        with self.assertRaises(ValueError):
            Parameter(
                name="test",
                param_type="discrete"
                # Missing choices
            )


class TestTrial(unittest.TestCase):
    """Test Trial class."""
    
    def test_trial_creation(self):
        """Test trial creation."""
        trial = Trial(
            trial_id=1,
            parameters={"lr": 0.01, "batch_size": 32},
            objectives={"mse": 0.1, "mae": 0.05},
            duration=120.5,
            status="completed"
        )
        
        self.assertEqual(trial.trial_id, 1)
        self.assertEqual(trial.parameters["lr"], 0.01)
        self.assertEqual(trial.objectives["mse"], 0.1)
        self.assertEqual(trial.duration, 120.5)
        self.assertEqual(trial.status, "completed")


class TestAcquisitionFunctions(unittest.TestCase):
    """Test acquisition functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.x = np.array([[1.0], [2.0], [3.0]])
        self.gp_mean = np.array([0.5, 1.0, 1.5])
        self.gp_std = np.array([0.1, 0.2, 0.3])
        self.best_value = 0.8
    
    def test_expected_improvement(self):
        """Test Expected Improvement acquisition function."""
        ei = ExpectedImprovement(xi=0.01)
        
        values = ei(self.x, self.gp_mean, self.gp_std, self.best_value)
        
        self.assertEqual(len(values), 3)
        self.assertTrue(np.all(values >= 0))  # EI should be non-negative
        self.assertIsInstance(values, np.ndarray)
    
    def test_upper_confidence_bound(self):
        """Test Upper Confidence Bound acquisition function."""
        ucb = UpperConfidenceBound(kappa=2.0)
        
        values = ucb(self.x, self.gp_mean, self.gp_std, self.best_value)
        
        self.assertEqual(len(values), 3)
        # UCB should be mean + kappa * std
        expected = self.gp_mean + 2.0 * self.gp_std
        np.testing.assert_array_almost_equal(values, expected)
    
    def test_probability_of_improvement(self):
        """Test Probability of Improvement acquisition function."""
        pi = ProbabilityOfImprovement(xi=0.01)
        
        values = pi(self.x, self.gp_mean, self.gp_std, self.best_value)
        
        self.assertEqual(len(values), 3)
        self.assertTrue(np.all(values >= 0))  # PI should be non-negative
        self.assertTrue(np.all(values <= 1))  # PI should be <= 1


class TestGaussianProcess(unittest.TestCase):
    """Test GaussianProcess class."""
    
    def setUp(self):
        """Set up test environment."""
        self.gp = GaussianProcess(kernel='rbf', length_scale=1.0)
    
    def test_gp_creation(self):
        """Test GP creation."""
        self.assertEqual(self.gp.kernel, 'rbf')
        self.assertEqual(self.gp.length_scale, 1.0)
        self.assertIsNone(self.gp.X_train)
    
    def test_rbf_kernel(self):
        """Test RBF kernel computation."""
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[0.0], [1.0], [2.0]])
        
        K = self.gp._rbf_kernel(X1, X2)
        
        self.assertEqual(K.shape, (2, 3))
        # Diagonal elements should be 1 (same points)
        self.assertAlmostEqual(K[0, 0], 1.0, places=6)
        self.assertAlmostEqual(K[1, 1], 1.0, places=6)
    
    def test_matern_kernel(self):
        """Test Matern kernel computation."""
        gp_matern = GaussianProcess(kernel='matern')
        
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[0.0], [1.0]])
        
        K = gp_matern._matern_kernel(X1, X2)
        
        self.assertEqual(K.shape, (2, 2))
        # Diagonal elements should be 1
        self.assertAlmostEqual(K[0, 0], 1.0, places=6)
        self.assertAlmostEqual(K[1, 1], 1.0, places=6)
    
    def test_fit_and_predict(self):
        """Test GP fitting and prediction."""
        # Generate simple training data
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([0.0, 1.0, 4.0])  # y = x^2
        
        self.gp.fit(X_train, y_train)
        
        # Test prediction
        X_test = np.array([[0.5], [1.5]])
        mean, std = self.gp.predict(X_test)
        
        self.assertEqual(len(mean), 2)
        self.assertEqual(len(std), 2)
        self.assertTrue(np.all(std > 0))  # Uncertainty should be positive
    
    def test_predict_without_fit(self):
        """Test prediction without fitting (should raise error)."""
        X_test = np.array([[0.5]])
        
        with self.assertRaises(ValueError):
            self.gp.predict(X_test)


class TestBayesianOptimizer(unittest.TestCase):
    """Test BayesianOptimizer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.parameters = [
            Parameter("x1", "continuous", bounds=(-5.0, 5.0)),
            Parameter("x2", "continuous", bounds=(-5.0, 5.0)),
            Parameter("optimizer", "categorical", choices=["adam", "sgd"])
        ]
        
        self.optimizer = BayesianOptimizer(
            parameters=self.parameters,
            n_initial_points=3,
            random_state=42
        )
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        self.assertEqual(len(self.optimizer.parameters), 3)
        self.assertEqual(self.optimizer.n_initial_points, 3)
        self.assertEqual(self.optimizer.n_dims, 3)
        self.assertFalse(self.optimizer.is_fitted)
    
    def test_parameter_encoding_decoding(self):
        """Test parameter encoding and decoding."""
        params = {"x1": 1.5, "x2": -2.0, "optimizer": "adam"}
        
        # Encode
        encoded = self.optimizer._encode_parameters(params)
        self.assertEqual(len(encoded), 3)
        
        # Decode
        decoded = self.optimizer._decode_parameters(encoded)
        
        self.assertAlmostEqual(decoded["x1"], 1.5, places=6)
        self.assertAlmostEqual(decoded["x2"], -2.0, places=6)
        self.assertEqual(decoded["optimizer"], "adam")
    
    def test_initial_point_generation(self):
        """Test initial point generation."""
        points = self.optimizer._generate_initial_points()
        
        self.assertEqual(len(points), 3)
        
        for point in points:
            self.assertIn("x1", point)
            self.assertIn("x2", point)
            self.assertIn("optimizer", point)
            
            # Check bounds
            self.assertGreaterEqual(point["x1"], -5.0)
            self.assertLessEqual(point["x1"], 5.0)
            self.assertIn(point["optimizer"], ["adam", "sgd"])
    
    def test_suggest_initial_points(self):
        """Test suggesting initial points."""
        for i in range(3):
            params = self.optimizer.suggest()
            
            self.assertIsInstance(params, dict)
            self.assertIn("x1", params)
            self.assertIn("x2", params)
            self.assertIn("optimizer", params)
    
    def test_tell_and_suggest_bayesian(self):
        """Test telling results and Bayesian suggestion."""
        # Provide initial evaluations
        for i in range(3):
            params = self.optimizer.suggest()
            # Simple quadratic objective
            objective = params["x1"]**2 + params["x2"]**2
            self.optimizer.tell(params, objective)
        
        # Now should use Bayesian optimization
        params = self.optimizer.suggest()
        
        self.assertIsInstance(params, dict)
        self.assertTrue(self.optimizer.is_fitted)
    
    def test_get_best_trial(self):
        """Test getting best trial."""
        # Initially no trials
        self.assertIsNone(self.optimizer.get_best_trial())
        
        # Add some trials
        params1 = {"x1": 1.0, "x2": 1.0, "optimizer": "adam"}
        params2 = {"x1": 0.0, "x2": 0.0, "optimizer": "sgd"}
        
        self.optimizer.tell(params1, 2.0)  # x1^2 + x2^2 = 2.0
        self.optimizer.tell(params2, 0.0)  # x1^2 + x2^2 = 0.0
        
        best_trial = self.optimizer.get_best_trial()
        
        self.assertIsNotNone(best_trial)
        self.assertEqual(best_trial.objectives["objective"], 0.0)
        self.assertEqual(best_trial.parameters, params2)
    
    def test_optimization_history(self):
        """Test getting optimization history."""
        # Add some trials
        params1 = {"x1": 1.0, "x2": 1.0, "optimizer": "adam"}
        params2 = {"x1": 0.0, "x2": 0.0, "optimizer": "sgd"}
        
        self.optimizer.tell(params1, 2.0, duration=10.0)
        self.optimizer.tell(params2, 0.0, duration=15.0)
        
        history = self.optimizer.get_optimization_history()
        
        self.assertIsInstance(history, pd.DataFrame)
        self.assertEqual(len(history), 2)
        self.assertIn("trial_id", history.columns)
        self.assertIn("objective", history.columns)
        self.assertIn("param_x1", history.columns)
    
    def test_save_load_state(self):
        """Test saving and loading optimizer state."""
        # Add some trials
        params = {"x1": 1.0, "x2": 1.0, "optimizer": "adam"}
        self.optimizer.tell(params, 2.0)
        
        # Save state
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self.optimizer.save_state(tmp_file.name)
            
            # Create new optimizer and load state
            new_optimizer = BayesianOptimizer(
                parameters=self.parameters,
                random_state=42
            )
            new_optimizer.load_state(tmp_file.name)
            
            # Check that state was loaded correctly
            self.assertEqual(len(new_optimizer.trials), 1)
            self.assertEqual(new_optimizer.trial_counter, 1)
            
            # Clean up
            Path(tmp_file.name).unlink()


class TestMultiObjectiveBayesianOptimizer(unittest.TestCase):
    """Test MultiObjectiveBayesianOptimizer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.parameters = [
            Parameter("x", "continuous", bounds=(0.0, 1.0))
        ]
        
        self.optimizer = MultiObjectiveBayesianOptimizer(
            parameters=self.parameters,
            objectives=["obj1", "obj2"],
            n_initial_points=2,
            random_state=42
        )
    
    def test_multi_objective_creation(self):
        """Test multi-objective optimizer creation."""
        self.assertEqual(self.optimizer.objectives, ["obj1", "obj2"])
        self.assertEqual(len(self.optimizer.gps), 2)
    
    def test_multi_objective_tell(self):
        """Test telling multi-objective results."""
        params = {"x": 0.5}
        objectives = {"obj1": 0.25, "obj2": 0.75}
        
        self.optimizer.tell(params, objectives)
        
        self.assertEqual(len(self.optimizer.trials), 1)
        self.assertEqual(self.optimizer.trials[0].objectives, objectives)
    
    def test_pareto_front(self):
        """Test Pareto front calculation."""
        # Add some trials
        self.optimizer.tell({"x": 0.1}, {"obj1": 0.1, "obj2": 0.9})  # Good obj1, bad obj2
        self.optimizer.tell({"x": 0.9}, {"obj1": 0.9, "obj2": 0.1})  # Bad obj1, good obj2
        self.optimizer.tell({"x": 0.5}, {"obj1": 0.5, "obj2": 0.5})  # Dominated by both
        
        pareto_front = self.optimizer.get_pareto_front()
        
        # Should have 2 non-dominated solutions
        self.assertEqual(len(pareto_front), 2)
        
        # Check that the dominated solution is not in Pareto front
        pareto_x_values = [trial.parameters["x"] for trial in pareto_front]
        self.assertNotIn(0.5, pareto_x_values)


class TestHyperparameterTuner(unittest.TestCase):
    """Test HyperparameterTuner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.parameter_space = {
            "lr": {"type": "continuous", "low": 0.001, "high": 0.1, "log": True},
            "batch_size": {"type": "discrete", "choices": [16, 32, 64, 128]},
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]}
        }
        
        # Simple objective function (minimize lr + batch_size/100)
        def objective_func(params):
            return params["lr"] + params["batch_size"] / 100.0
        
        self.tuner = HyperparameterTuner(
            parameter_space=self.parameter_space,
            objective_function=objective_func,
            n_trials=5,
            random_state=42
        )
    
    def test_tuner_creation(self):
        """Test tuner creation."""
        self.assertEqual(len(self.tuner.parameters), 3)
        self.assertEqual(self.tuner.n_trials, 5)
    
    def test_parameter_space_parsing(self):
        """Test parameter space parsing."""
        params = self.tuner.parameters
        
        # Check continuous parameter
        lr_param = next(p for p in params if p.name == "lr")
        self.assertEqual(lr_param.param_type, "continuous")
        self.assertTrue(lr_param.log_scale)
        
        # Check discrete parameter
        batch_param = next(p for p in params if p.name == "batch_size")
        self.assertEqual(batch_param.param_type, "discrete")
        self.assertEqual(batch_param.choices, [16, 32, 64, 128])
        
        # Check categorical parameter
        opt_param = next(p for p in params if p.name == "optimizer")
        self.assertEqual(opt_param.param_type, "categorical")
    
    def test_optimization(self):
        """Test running optimization."""
        best_params = self.tuner.optimize()
        
        self.assertIsInstance(best_params, dict)
        self.assertIn("lr", best_params)
        self.assertIn("batch_size", best_params)
        self.assertIn("optimizer", best_params)
        
        # Should find relatively good parameters (low lr and batch_size)
        self.assertLess(best_params["lr"], 0.05)  # Should find low learning rate
    
    def test_simple_bounds_specification(self):
        """Test simple bounds specification."""
        simple_space = {
            "x": (0.0, 1.0),  # Simple tuple bounds
            "y": {"type": "continuous", "low": -1.0, "high": 1.0}
        }
        
        def simple_objective(params):
            return params["x"]**2 + params["y"]**2
        
        tuner = HyperparameterTuner(
            parameter_space=simple_space,
            objective_function=simple_objective,
            n_trials=3
        )
        
        self.assertEqual(len(tuner.parameters), 2)
        
        # Both should be continuous parameters
        for param in tuner.parameters:
            self.assertEqual(param.param_type, "continuous")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_parameter_space(self):
        """Test parameter space creation utility."""
        config = {
            "lr": {"type": "continuous", "low": 0.001, "high": 0.1, "log": True},
            "num_layers": {"type": "discrete", "choices": [1, 2, 3]},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]}
        }
        
        parameters = create_parameter_space(config)
        
        self.assertEqual(len(parameters), 3)
        
        # Check each parameter
        lr_param = next(p for p in parameters if p.name == "lr")
        self.assertEqual(lr_param.param_type, "continuous")
        self.assertTrue(lr_param.log_scale)
        
        layers_param = next(p for p in parameters if p.name == "num_layers")
        self.assertEqual(layers_param.param_type, "discrete")
        
        activation_param = next(p for p in parameters if p.name == "activation")
        self.assertEqual(activation_param.param_type, "categorical")
    
    def test_optimize_model_hyperparameters(self):
        """Test model hyperparameter optimization utility."""
        # Mock model class
        class MockModel:
            def __init__(self, lr=0.01, hidden_dim=64):
                self.lr = lr
                self.hidden_dim = hidden_dim
            
            def train(self):
                pass
        
        # Mock data
        train_data = None
        val_data = None
        
        parameter_space = {
            "lr": {"type": "continuous", "low": 0.001, "high": 0.1},
            "hidden_dim": {"type": "discrete", "choices": [32, 64, 128]}
        }
        
        # This will use random validation loss, so just test that it runs
        best_params = optimize_model_hyperparameters(
            MockModel, train_data, val_data, parameter_space, n_trials=3
        )
        
        self.assertIsInstance(best_params, dict)
        # May be empty if all trials failed, which is OK for this test


class TestIntegration(unittest.TestCase):
    """Integration tests for Bayesian optimization."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Define a simple optimization problem: minimize (x-2)^2 + (y+1)^2
        def objective(params):
            x, y = params["x"], params["y"]
            return (x - 2)**2 + (y + 1)**2
        
        parameter_space = {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0)
        }
        
        tuner = HyperparameterTuner(
            parameter_space=parameter_space,
            objective_function=objective,
            n_trials=20,
            acquisition_function='ei',
            random_state=42
        )
        
        best_params = tuner.optimize()
        
        # Should find parameters close to optimal (x=2, y=-1)
        self.assertLess(abs(best_params["x"] - 2.0), 1.0)
        self.assertLess(abs(best_params["y"] - (-1.0)), 1.0)
        
        # Check optimization history
        history = tuner.get_optimization_history()
        self.assertEqual(len(history), 20)
        
        # Objective should generally improve over time
        objectives = history["objective"].values
        # Best objective should be better than first few
        best_objective = min(objectives)
        initial_avg = np.mean(objectives[:5])
        self.assertLess(best_objective, initial_avg)
    
    def test_mixed_parameter_types(self):
        """Test optimization with mixed parameter types."""
        def objective(params):
            # Minimize based on continuous and discrete parameters
            x = params["x"]
            n = params["n_layers"]
            activation_penalty = 0.1 if params["activation"] == "relu" else 0.0
            
            return x**2 + n * 0.1 + activation_penalty
        
        parameter_space = {
            "x": {"type": "continuous", "low": -2.0, "high": 2.0},
            "n_layers": {"type": "discrete", "choices": [1, 2, 3, 4]},
            "activation": {"type": "categorical", "choices": ["relu", "tanh", "sigmoid"]}
        }
        
        tuner = HyperparameterTuner(
            parameter_space=parameter_space,
            objective_function=objective,
            n_trials=15,
            random_state=42
        )
        
        best_params = tuner.optimize()
        
        # Should find x close to 0, n_layers = 1, activation != "relu"
        self.assertLess(abs(best_params["x"]), 0.5)
        self.assertEqual(best_params["n_layers"], 1)
        self.assertNotEqual(best_params["activation"], "relu")


if __name__ == '__main__':
    unittest.main()