"""Hyperparameter optimization system for stock prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection import TimeSeriesSplit
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import ModelTrainingError
from stock_predictor.evaluation.evaluator import ModelEvaluator


class HyperparameterOptimizer:
    """Optimizes hyperparameters for stock prediction models."""
    
    def __init__(self, n_jobs: int = -1, cv_folds: int = 3):
        self.logger = get_logger('evaluation.hyperparameter_optimizer')
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.cv_folds = cv_folds
        self.evaluator = ModelEvaluator()
        self.optimization_history = {}
    
    def grid_search(self, model_class, param_grid: Dict[str, List], 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   scoring: str = 'rmse') -> Dict[str, Any]:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            model_class: Model class to optimize
            param_grid: Dictionary of parameter names and values to try
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            scoring: Scoring metric ('rmse', 'mae', 'directional_accuracy')
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting grid search for {model_class.__name__}")
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Perform optimization
        results = self._optimize_parameters(
            model_class, param_combinations, X_train, y_train, X_val, y_val, scoring
        )
        
        # Store results
        self.optimization_history[model_class.__name__] = results
        
        return results
    
    def random_search(self, model_class, param_distributions: Dict[str, Any],
                     n_iter: int, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                     scoring: str = 'rmse', random_state: int = 42) -> Dict[str, Any]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            model_class: Model class to optimize
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of parameter combinations to try
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            scoring: Scoring metric
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting random search for {model_class.__name__}")
        
        # Generate random parameter combinations
        param_sampler = ParameterSampler(
            param_distributions, n_iter=n_iter, random_state=random_state
        )
        param_combinations = list(param_sampler)
        
        self.logger.info(f"Testing {len(param_combinations)} random parameter combinations")
        
        # Perform optimization
        results = self._optimize_parameters(
            model_class, param_combinations, X_train, y_train, X_val, y_val, scoring
        )
        
        # Store results
        self.optimization_history[model_class.__name__] = results
        
        return results
    
    def bayesian_optimization(self, model_class, param_space: Dict[str, Tuple],
                            n_calls: int, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                            scoring: str = 'rmse', random_state: int = 42) -> Dict[str, Any]:
        """
        Perform Bayesian optimization (requires scikit-optimize).
        
        Args:
            model_class: Model class to optimize
            param_space: Dictionary of parameter spaces (name: (low, high))
            n_calls: Number of optimization calls
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            scoring: Scoring metric
            random_state: Random state
            
        Returns:
            Dictionary with optimization results
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            self.logger.warning("scikit-optimize not available, falling back to random search")
            # Convert param_space to distributions for random search
            param_distributions = {}
            for name, (low, high) in param_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    param_distributions[name] = list(range(low, high + 1))
                else:
                    param_distributions[name] = [low + (high - low) * i / 10 for i in range(11)]
            
            return self.random_search(
                model_class, param_distributions, n_calls, 
                X_train, y_train, X_val, y_val, scoring, random_state
            )
        
        self.logger.info(f"Starting Bayesian optimization for {model_class.__name__}")
        
        # Define search space
        dimensions = []
        param_names = []
        
        for name, (low, high) in param_space.items():
            param_names.append(name)
            if isinstance(low, int) and isinstance(high, int):
                dimensions.append(Integer(low, high, name=name))
            else:
                dimensions.append(Real(low, high, name=name))
        
        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            try:
                # Create model with parameters
                model = model_class(f"{model_class.__name__}_bayesian")
                model.set_hyperparameters(params)
                
                # Train and evaluate
                model.train(X_train, y_train, X_val, y_val)
                
                if X_val is not None and y_val is not None:
                    predictions = model.predict(X_val)
                    score = self._calculate_score(y_val, predictions, scoring)
                else:
                    # Use cross-validation
                    cv_results = self.evaluator.cross_validation_evaluation(
                        model, X_train, y_train, cv_folds=self.cv_folds
                    )
                    score = cv_results.get(f'{scoring}_mean', float('inf'))
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Evaluation failed for params {params}: {str(e)}")
                return float('inf')
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI'  # Expected Improvement
        )
        
        # Format results
        best_params = dict(zip(param_names, result.x))
        
        optimization_results = {
            'best_params': best_params,
            'best_score': float(result.fun),
            'n_calls': len(result.func_vals),
            'optimization_method': 'bayesian',
            'scoring_metric': scoring,
            'convergence_trace': [float(val) for val in result.func_vals]
        }
        
        self.logger.info(f"Bayesian optimization completed. Best {scoring}: {result.fun:.6f}")
        
        return optimization_results
    
    def _optimize_parameters(self, model_class, param_combinations: List[Dict],
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                           scoring: str) -> Dict[str, Any]:
        """Internal method to optimize parameters."""
        
        best_score = float('inf') if scoring in ['rmse', 'mae'] else float('-inf')
        best_params = None
        all_results = []
        
        # Sequential evaluation (can be parallelized for production)
        for i, params in enumerate(param_combinations):
            try:
                self.logger.debug(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
                
                # Create and configure model
                model = model_class(f"{model_class.__name__}_opt_{i}")
                model.set_hyperparameters(params)
                
                # Train model
                model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate model
                if X_val is not None and y_val is not None:
                    predictions = model.predict(X_val)
                    score = self._calculate_score(y_val, predictions, scoring)
                else:
                    # Use cross-validation if no validation set
                    cv_results = self.evaluator.cross_validation_evaluation(
                        model, X_train, y_train, cv_folds=self.cv_folds
                    )
                    score = cv_results.get(f'{scoring}_mean', float('inf'))
                
                # Store result
                result = {
                    'params': params.copy(),
                    'score': float(score),
                    'model_name': model.name
                }
                all_results.append(result)
                
                # Update best if better
                is_better = (score < best_score if scoring in ['rmse', 'mae'] else score > best_score)
                if is_better:
                    best_score = score
                    best_params = params.copy()
                
            except Exception as e:
                self.logger.warning(f"Parameter combination {params} failed: {str(e)}")
                continue
        
        if best_params is None:
            raise ModelTrainingError("No valid parameter combinations found")
        
        # Sort results by score
        all_results.sort(key=lambda x: x['score'], reverse=(scoring not in ['rmse', 'mae']))
        
        optimization_results = {
            'best_params': best_params,
            'best_score': float(best_score),
            'n_combinations_tested': len(all_results),
            'optimization_method': 'grid_search',
            'scoring_metric': scoring,
            'all_results': all_results[:10],  # Top 10 results
            'improvement_over_default': self._calculate_improvement(all_results, scoring)
        }
        
        self.logger.info(f"Optimization completed. Best {scoring}: {best_score:.6f}")
        
        return optimization_results
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate score based on scoring metric."""
        if scoring == 'rmse':
            return self.evaluator.calculate_rmse(y_true, y_pred)
        elif scoring == 'mae':
            return self.evaluator.calculate_mae(y_true, y_pred)
        elif scoring == 'directional_accuracy':
            return self.evaluator.calculate_directional_accuracy(y_true, y_pred)
        elif scoring == 'r2_score':
            return self.evaluator.calculate_r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")
    
    def _calculate_improvement(self, results: List[Dict], scoring: str) -> Optional[float]:
        """Calculate improvement over baseline."""
        if len(results) < 2:
            return None
        
        best_score = results[0]['score']
        worst_score = results[-1]['score']
        
        if scoring in ['rmse', 'mae']:
            # Lower is better
            improvement = ((worst_score - best_score) / worst_score) * 100
        else:
            # Higher is better
            improvement = ((best_score - worst_score) / worst_score) * 100
        
        return float(improvement)
    
    def optimize_model_ensemble(self, model_configs: Dict[str, Dict], 
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for multiple models in ensemble.
        
        Args:
            model_configs: Dict of {model_name: {'class': ModelClass, 'param_grid': {...}}}
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with optimization results for all models
        """
        self.logger.info("Starting ensemble hyperparameter optimization")
        
        ensemble_results = {}
        
        for model_name, config in model_configs.items():
            try:
                self.logger.info(f"Optimizing {model_name}")
                
                model_class = config['class']
                param_grid = config['param_grid']
                method = config.get('method', 'grid')
                
                if method == 'grid':
                    results = self.grid_search(
                        model_class, param_grid, X_train, y_train, X_val, y_val
                    )
                elif method == 'random':
                    n_iter = config.get('n_iter', 50)
                    results = self.random_search(
                        model_class, param_grid, n_iter, X_train, y_train, X_val, y_val
                    )
                elif method == 'bayesian':
                    n_calls = config.get('n_calls', 50)
                    results = self.bayesian_optimization(
                        model_class, param_grid, n_calls, X_train, y_train, X_val, y_val
                    )
                else:
                    raise ValueError(f"Unknown optimization method: {method}")
                
                ensemble_results[model_name] = results
                
            except Exception as e:
                self.logger.error(f"Failed to optimize {model_name}: {str(e)}")
                continue
        
        return ensemble_results
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of all optimization runs."""
        if not self.optimization_history:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.optimization_history.items():
            summary_data.append({
                'Model': model_name,
                'Best Score': results['best_score'],
                'Scoring Metric': results['scoring_metric'],
                'Combinations Tested': results.get('n_combinations_tested', results.get('n_calls', 0)),
                'Optimization Method': results.get('optimization_method', 'unknown'),
                'Improvement (%)': results.get('improvement_over_default', 0)
            })
        
        return pd.DataFrame(summary_data)
    
    def suggest_parameters(self, model_class, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Suggest good starting parameters based on data characteristics.
        
        Args:
            model_class: Model class
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of suggested parameters
        """
        n_samples, n_features = X_train.shape
        
        suggestions = {}
        
        if 'ARIMA' in model_class.__name__:
            # ARIMA suggestions based on data size
            suggestions = {
                'max_p': min(5, n_samples // 50),
                'max_d': 2,
                'max_q': min(5, n_samples // 50),
                'seasonal': n_samples > 100
            }
        
        elif 'LSTM' in model_class.__name__:
            # LSTM suggestions based on data size and complexity
            suggestions = {
                'sequence_length': min(60, n_samples // 10),
                'units': [50, 50] if n_samples > 500 else [25, 25],
                'dropout': 0.2 if n_samples > 1000 else 0.1,
                'epochs': min(100, max(50, n_samples // 10)),
                'batch_size': min(32, max(8, n_samples // 50))
            }
        
        elif 'RandomForest' in model_class.__name__:
            # Random Forest suggestions
            suggestions = {
                'n_estimators': min(200, max(50, n_samples // 10)),
                'max_depth': min(20, max(5, int(np.log2(n_features)) + 1)),
                'min_samples_split': max(2, n_samples // 1000),
                'min_samples_leaf': max(1, n_samples // 2000),
                'max_features': 'sqrt' if n_features > 10 else None
            }
        
        self.logger.info(f"Parameter suggestions for {model_class.__name__}: {suggestions}")
        
        return suggestions