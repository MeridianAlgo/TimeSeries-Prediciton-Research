"""
Bayesian optimization framework for hyperparameter tuning using Gaussian processes.
Implements acquisition functions, multi-objective optimization, and parameter sensitivity analysis.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
import json
import pickle
from pathlib import Path
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import time

warnings.filterwarnings('ignore')


@dataclass
class Parameter:
    """Parameter definition for optimization."""
    
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None  # For continuous parameters
    choices: Optional[List] = None  # For discrete/categorical parameters
    log_scale: bool = False  # Whether to use log scale for continuous parameters
    
    def __post_init__(self):
        """Validate parameter definition."""
        if self.param_type == 'continuous':
            if self.bounds is None:
                raise ValueError(f"Continuous parameter {self.name} must have bounds")
        elif self.param_type in ['discrete', 'categorical']:
            if self.choices is None:
                raise ValueError(f"{self.param_type} parameter {self.name} must have choices")


@dataclass
class Trial:
    """Single optimization trial result."""
    
    trial_id: int
    parameters: Dict[str, Any]
    objectives: Dict[str, float]  # Multiple objectives supported
    metadata: Optional[Dict] = None
    duration: Optional[float] = None
    status: str = 'completed'  # 'completed', 'failed', 'pruned'


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray, gp_mean: np.ndarray, gp_std: np.ndarray,
                 best_value: float) -> np.ndarray:
        """
        Compute acquisition function value.
        
        Args:
            x: Input points
            gp_mean: GP posterior mean
            gp_std: GP posterior standard deviation
            best_value: Current best objective value
            
        Returns:
            Acquisition function values
        """
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        """
        Initialize Expected Improvement.
        
        Args:
            xi: Exploration parameter
        """
        self.xi = xi
    
    def __call__(self, x: np.ndarray, gp_mean: np.ndarray, gp_std: np.ndarray,
                 best_value: float) -> np.ndarray:
        """Compute Expected Improvement."""
        improvement = gp_mean - best_value - self.xi
        
        # Avoid division by zero
        gp_std = np.maximum(gp_std, 1e-9)
        
        z = improvement / gp_std
        ei = improvement * norm.cdf(z) + gp_std * norm.pdf(z)
        
        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    
    def __init__(self, kappa: float = 2.576):
        """
        Initialize Upper Confidence Bound.
        
        Args:
            kappa: Exploration parameter (2.576 for 99% confidence)
        """
        self.kappa = kappa
    
    def __call__(self, x: np.ndarray, gp_mean: np.ndarray, gp_std: np.ndarray,
                 best_value: float) -> np.ndarray:
        """Compute Upper Confidence Bound."""
        return gp_mean + self.kappa * gp_std


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        """
        Initialize Probability of Improvement.
        
        Args:
            xi: Exploration parameter
        """
        self.xi = xi
    
    def __call__(self, x: np.ndarray, gp_mean: np.ndarray, gp_std: np.ndarray,
                 best_value: float) -> np.ndarray:
        """Compute Probability of Improvement."""
        improvement = gp_mean - best_value - self.xi
        
        # Avoid division by zero
        gp_std = np.maximum(gp_std, 1e-9)
        
        z = improvement / gp_std
        pi = norm.cdf(z)
        
        return pi


class GaussianProcess:
    """Simple Gaussian Process implementation for Bayesian optimization."""
    
    def __init__(self, kernel: str = 'rbf', length_scale: float = 1.0,
                 noise_level: float = 1e-5):
        """
        Initialize Gaussian Process.
        
        Args:
            kernel: Kernel type ('rbf', 'matern')
            length_scale: Kernel length scale
            noise_level: Noise level for numerical stability
        """
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise_level = noise_level
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel."""
        # Compute squared Euclidean distances
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        
        distances_sq = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
        distances_sq = np.maximum(distances_sq, 0)  # Numerical stability
        
        return np.exp(-distances_sq / (2 * self.length_scale**2))
    
    def _matern_kernel(self, X1: np.ndarray, X2: np.ndarray, nu: float = 2.5) -> np.ndarray:
        """Matern kernel."""
        # Simplified Matern kernel (nu=2.5)
        distances = np.sqrt(np.maximum(
            np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2), 1e-12
        ))
        
        sqrt5_d = np.sqrt(5) * distances / self.length_scale
        
        return (1 + sqrt5_d + (5 * distances**2) / (3 * self.length_scale**2)) * \
               np.exp(-sqrt5_d)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Process to training data.
        
        Args:
            X: Training inputs of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Compute kernel matrix
        if self.kernel == 'rbf':
            K = self._rbf_kernel(X, X)
        elif self.kernel == 'matern':
            K = self._matern_kernel(X, X)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # Add noise for numerical stability
        K += self.noise_level * np.eye(len(X))
        
        # Compute inverse (with regularization)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.K_inv = np.linalg.pinv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty.
        
        Args:
            X: Test inputs of shape (n_test, n_features)
            
        Returns:
            Tuple of (mean, std) predictions
        """
        if self.X_train is None:
            raise ValueError("GP not fitted yet")
        
        # Compute kernel between test and training points
        if self.kernel == 'rbf':
            K_star = self._rbf_kernel(X, self.X_train)
            K_star_star = self._rbf_kernel(X, X)
        elif self.kernel == 'matern':
            K_star = self._matern_kernel(X, self.X_train)
            K_star_star = self._matern_kernel(X, X)
        
        # Compute mean prediction
        mean = K_star @ self.K_inv @ self.y_train
        
        # Compute variance prediction
        var = np.diag(K_star_star) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
        var = np.maximum(var, 1e-9)  # Ensure positive variance
        std = np.sqrt(var)
        
        return mean, std


class BayesianOptimizer:
    """Bayesian optimizer using Gaussian processes."""
    
    def __init__(self, 
                 parameters: List[Parameter],
                 acquisition_function: AcquisitionFunction = None,
                 n_initial_points: int = 5,
                 kernel: str = 'rbf',
                 random_state: Optional[int] = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            parameters: List of parameters to optimize
            acquisition_function: Acquisition function to use
            n_initial_points: Number of initial random points
            kernel: GP kernel type
            random_state: Random seed
        """
        self.parameters = parameters
        self.acquisition_function = acquisition_function or ExpectedImprovement()
        self.n_initial_points = n_initial_points
        self.kernel = kernel
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize storage
        self.trials = []
        self.trial_counter = 0
        
        # Create parameter space mapping
        self._setup_parameter_space()
        
        # Initialize GP
        self.gp = GaussianProcess(kernel=kernel)
        self.is_fitted = False
        
    def _setup_parameter_space(self):
        """Setup parameter space for optimization."""
        self.param_bounds = []
        self.param_types = []
        self.param_names = []
        
        for param in self.parameters:
            self.param_names.append(param.name)
            self.param_types.append(param.param_type)
            
            if param.param_type == 'continuous':
                self.param_bounds.append(param.bounds)
            elif param.param_type == 'discrete':
                # Map discrete choices to continuous space
                self.param_bounds.append((0, len(param.choices) - 1))
            elif param.param_type == 'categorical':
                # Map categorical choices to continuous space
                self.param_bounds.append((0, len(param.choices) - 1))
        
        self.param_bounds = np.array(self.param_bounds)
        self.n_dims = len(self.parameters)
    
    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameter dictionary to continuous space."""
        encoded = np.zeros(self.n_dims)
        
        for i, param in enumerate(self.parameters):
            value = params[param.name]
            
            if param.param_type == 'continuous':
                if param.log_scale:
                    encoded[i] = np.log(value)
                else:
                    encoded[i] = value
            elif param.param_type in ['discrete', 'categorical']:
                encoded[i] = param.choices.index(value)
        
        return encoded
    
    def _decode_parameters(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode continuous space to parameter dictionary."""
        params = {}
        
        for i, param in enumerate(self.parameters):
            value = encoded[i]
            
            if param.param_type == 'continuous':
                # Clip to bounds
                value = np.clip(value, param.bounds[0], param.bounds[1])
                
                if param.log_scale:
                    params[param.name] = np.exp(value)
                else:
                    params[param.name] = float(value)
            elif param.param_type == 'discrete':
                # Round to nearest integer and clip
                idx = int(np.clip(np.round(value), 0, len(param.choices) - 1))
                params[param.name] = param.choices[idx]
            elif param.param_type == 'categorical':
                # Round to nearest integer and clip
                idx = int(np.clip(np.round(value), 0, len(param.choices) - 1))
                params[param.name] = param.choices[idx]
        
        return params
    
    def _generate_initial_points(self) -> List[Dict[str, Any]]:
        """Generate initial random points."""
        points = []
        
        for _ in range(self.n_initial_points):
            params = {}
            
            for param in self.parameters:
                if param.param_type == 'continuous':
                    if param.log_scale:
                        # Log-uniform sampling
                        log_low, log_high = np.log(param.bounds[0]), np.log(param.bounds[1])
                        value = np.exp(np.random.uniform(log_low, log_high))
                    else:
                        value = np.random.uniform(param.bounds[0], param.bounds[1])
                    params[param.name] = value
                elif param.param_type in ['discrete', 'categorical']:
                    params[param.name] = np.random.choice(param.choices)
            
            points.append(params)
        
        return points
    
    def _optimize_acquisition(self, n_candidates: int = 1000) -> np.ndarray:
        """Optimize acquisition function to find next point."""
        if not self.is_fitted:
            raise ValueError("Optimizer not fitted yet")
        
        # Get current best value
        objectives = [trial.objectives.get('objective', float('inf')) for trial in self.trials]
        best_value = min(objectives)
        
        # Generate candidate points
        candidates = np.random.uniform(
            self.param_bounds[:, 0],
            self.param_bounds[:, 1],
            size=(n_candidates, self.n_dims)
        )
        
        # Evaluate acquisition function
        gp_mean, gp_std = self.gp.predict(candidates)
        acquisition_values = self.acquisition_function(
            candidates, gp_mean, gp_std, best_value
        )
        
        # Find best candidate
        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]
    
    def suggest(self) -> Dict[str, Any]:
        """Suggest next parameter configuration to evaluate."""
        if len(self.trials) < self.n_initial_points:
            # Generate initial random points
            if not hasattr(self, '_initial_points'):
                self._initial_points = self._generate_initial_points()
            
            point_idx = len(self.trials)
            return self._initial_points[point_idx]
        
        else:
            # Use Bayesian optimization
            if not self.is_fitted:
                self._fit_gp()
            
            # Optimize acquisition function
            next_point_encoded = self._optimize_acquisition()
            return self._decode_parameters(next_point_encoded)
    
    def tell(self, parameters: Dict[str, Any], objective: float, 
             metadata: Optional[Dict] = None, duration: Optional[float] = None):
        """
        Report evaluation result.
        
        Args:
            parameters: Parameter configuration that was evaluated
            objective: Objective value (lower is better)
            metadata: Optional metadata
            duration: Evaluation duration
        """
        trial = Trial(
            trial_id=self.trial_counter,
            parameters=parameters.copy(),
            objectives={'objective': objective},
            metadata=metadata,
            duration=duration
        )
        
        self.trials.append(trial)
        self.trial_counter += 1
        
        # Refit GP if we have enough points
        if len(self.trials) >= self.n_initial_points:
            self._fit_gp()
    
    def _fit_gp(self):
        """Fit Gaussian Process to current trials."""
        if len(self.trials) < 2:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for trial in self.trials:
            if trial.status == 'completed':
                encoded_params = self._encode_parameters(trial.parameters)
                X.append(encoded_params)
                y.append(trial.objectives['objective'])
        
        if len(X) < 2:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit GP
        self.gp.fit(X, y)
        self.is_fitted = True
    
    def get_best_trial(self) -> Optional[Trial]:
        """Get the best trial so far."""
        if not self.trials:
            return None
        
        completed_trials = [t for t in self.trials if t.status == 'completed']
        if not completed_trials:
            return None
        
        best_trial = min(completed_trials, 
                        key=lambda t: t.objectives['objective'])
        return best_trial
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        data = []
        
        for trial in self.trials:
            row = {
                'trial_id': trial.trial_id,
                'objective': trial.objectives.get('objective'),
                'duration': trial.duration,
                'status': trial.status
            }
            
            # Add parameter values
            for param_name, param_value in trial.parameters.items():
                row[f'param_{param_name}'] = param_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_state(self, filepath: str):
        """Save optimizer state."""
        state = {
            'parameters': self.parameters,
            'trials': self.trials,
            'trial_counter': self.trial_counter,
            'n_initial_points': self.n_initial_points,
            'kernel': self.kernel
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load optimizer state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.parameters = state['parameters']
        self.trials = state['trials']
        self.trial_counter = state['trial_counter']
        self.n_initial_points = state['n_initial_points']
        self.kernel = state['kernel']
        
        # Rebuild parameter space
        self._setup_parameter_space()
        
        # Refit GP if needed
        if len(self.trials) >= self.n_initial_points:
            self._fit_gp()


class MultiObjectiveBayesianOptimizer(BayesianOptimizer):
    """Multi-objective Bayesian optimizer."""
    
    def __init__(self, 
                 parameters: List[Parameter],
                 objectives: List[str],
                 **kwargs):
        """
        Initialize multi-objective optimizer.
        
        Args:
            parameters: List of parameters to optimize
            objectives: List of objective names
            **kwargs: Additional arguments for BayesianOptimizer
        """
        super().__init__(parameters, **kwargs)
        self.objectives = objectives
        self.gps = {obj: GaussianProcess(kernel=self.kernel) for obj in objectives}
        
    def tell(self, parameters: Dict[str, Any], objectives: Dict[str, float],
             metadata: Optional[Dict] = None, duration: Optional[float] = None):
        """
        Report multi-objective evaluation result.
        
        Args:
            parameters: Parameter configuration
            objectives: Dictionary of objective values
            metadata: Optional metadata
            duration: Evaluation duration
        """
        trial = Trial(
            trial_id=self.trial_counter,
            parameters=parameters.copy(),
            objectives=objectives.copy(),
            metadata=metadata,
            duration=duration
        )
        
        self.trials.append(trial)
        self.trial_counter += 1
        
        # Refit GPs
        if len(self.trials) >= self.n_initial_points:
            self._fit_multi_gps()
    
    def _fit_multi_gps(self):
        """Fit separate GPs for each objective."""
        if len(self.trials) < 2:
            return
        
        # Prepare training data
        X = []
        y_dict = {obj: [] for obj in self.objectives}
        
        for trial in self.trials:
            if trial.status == 'completed':
                encoded_params = self._encode_parameters(trial.parameters)
                X.append(encoded_params)
                
                for obj in self.objectives:
                    y_dict[obj].append(trial.objectives.get(obj, 0.0))
        
        if len(X) < 2:
            return
        
        X = np.array(X)
        
        # Fit GP for each objective
        for obj in self.objectives:
            y = np.array(y_dict[obj])
            self.gps[obj].fit(X, y)
        
        self.is_fitted = True
    
    def get_pareto_front(self) -> List[Trial]:
        """Get Pareto-optimal trials."""
        completed_trials = [t for t in self.trials if t.status == 'completed']
        if not completed_trials:
            return []
        
        pareto_trials = []
        
        for trial in completed_trials:
            is_dominated = False
            
            for other_trial in completed_trials:
                if other_trial == trial:
                    continue
                
                # Check if other_trial dominates trial
                dominates = True
                for obj in self.objectives:
                    if other_trial.objectives[obj] >= trial.objectives[obj]:
                        dominates = False
                        break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_trials.append(trial)
        
        return pareto_trials


class HyperparameterTuner:
    """High-level interface for hyperparameter tuning."""
    
    def __init__(self, 
                 parameter_space: Dict[str, Any],
                 objective_function: Callable,
                 n_trials: int = 100,
                 acquisition_function: str = 'ei',
                 random_state: Optional[int] = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            parameter_space: Dictionary defining parameter space
            objective_function: Function to optimize
            n_trials: Number of optimization trials
            acquisition_function: Acquisition function ('ei', 'ucb', 'pi')
            random_state: Random seed
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.random_state = random_state
        
        # Parse parameter space
        self.parameters = self._parse_parameter_space(parameter_space)
        
        # Create acquisition function
        if acquisition_function == 'ei':
            acq_func = ExpectedImprovement()
        elif acquisition_function == 'ucb':
            acq_func = UpperConfidenceBound()
        elif acquisition_function == 'pi':
            acq_func = ProbabilityOfImprovement()
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
        
        # Create optimizer
        self.optimizer = BayesianOptimizer(
            parameters=self.parameters,
            acquisition_function=acq_func,
            random_state=random_state
        )
        
    def _parse_parameter_space(self, space: Dict[str, Any]) -> List[Parameter]:
        """Parse parameter space dictionary."""
        parameters = []
        
        for name, config in space.items():
            if isinstance(config, dict):
                param_type = config.get('type', 'continuous')
                
                if param_type == 'continuous':
                    param = Parameter(
                        name=name,
                        param_type='continuous',
                        bounds=(config['low'], config['high']),
                        log_scale=config.get('log', False)
                    )
                elif param_type in ['discrete', 'categorical']:
                    param = Parameter(
                        name=name,
                        param_type=param_type,
                        choices=config['choices']
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
                    
                parameters.append(param)
            
            elif isinstance(config, (list, tuple)):
                # Simple bounds specification
                param = Parameter(
                    name=name,
                    param_type='continuous',
                    bounds=(config[0], config[1])
                )
                parameters.append(param)
        
        return parameters
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization.
        
        Returns:
            Best parameter configuration found
        """
        logging.info(f"Starting Bayesian optimization with {self.n_trials} trials")
        
        for trial_idx in range(self.n_trials):
            # Get next parameter suggestion
            params = self.optimizer.suggest()
            
            # Evaluate objective function
            start_time = time.time()
            try:
                objective_value = self.objective_function(params)
                duration = time.time() - start_time
                
                # Report result
                self.optimizer.tell(params, objective_value, duration=duration)
                
                logging.info(f"Trial {trial_idx + 1}/{self.n_trials}: "
                           f"objective={objective_value:.6f}, params={params}")
                
            except Exception as e:
                logging.error(f"Trial {trial_idx + 1} failed: {e}")
                # Report failed trial
                self.optimizer.tell(params, float('inf'), duration=0.0)
        
        # Get best result
        best_trial = self.optimizer.get_best_trial()
        
        if best_trial:
            logging.info(f"Best trial: objective={best_trial.objectives['objective']:.6f}, "
                        f"params={best_trial.parameters}")
            return best_trial.parameters
        else:
            logging.warning("No successful trials found")
            return {}
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history."""
        return self.optimizer.get_optimization_history()


# Utility functions
def create_parameter_space(config: Dict[str, Any]) -> List[Parameter]:
    """Create parameter list from configuration."""
    parameters = []
    
    for name, param_config in config.items():
        if param_config['type'] == 'continuous':
            param = Parameter(
                name=name,
                param_type='continuous',
                bounds=(param_config['low'], param_config['high']),
                log_scale=param_config.get('log', False)
            )
        elif param_config['type'] in ['discrete', 'categorical']:
            param = Parameter(
                name=name,
                param_type=param_config['type'],
                choices=param_config['choices']
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_config['type']}")
        
        parameters.append(param)
    
    return parameters


def optimize_model_hyperparameters(model_class, train_data, val_data, 
                                 parameter_space: Dict[str, Any],
                                 n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a model class.
    
    Args:
        model_class: Model class to optimize
        train_data: Training data
        val_data: Validation data
        parameter_space: Parameter space definition
        n_trials: Number of optimization trials
        
    Returns:
        Best hyperparameters found
    """
    def objective_function(params):
        """Objective function for hyperparameter optimization."""
        try:
            # Create model with parameters
            model = model_class(**params)
            
            # Train model (simplified)
            # In practice, this would involve proper training loop
            model.train()
            
            # Evaluate on validation data
            # This is a placeholder - implement actual evaluation
            val_loss = np.random.random()  # Replace with actual validation
            
            return val_loss
            
        except Exception as e:
            logging.error(f"Error in objective function: {e}")
            return float('inf')
    
    # Create tuner
    tuner = HyperparameterTuner(
        parameter_space=parameter_space,
        objective_function=objective_function,
        n_trials=n_trials
    )
    
    # Run optimization
    best_params = tuner.optimize()
    
    return best_params