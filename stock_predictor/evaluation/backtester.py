"""Backtesting framework for stock price prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import warnings

from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import ModelTrainingError
from stock_predictor.evaluation.evaluator import ModelEvaluator


class Backtester:
    """Backtests stock prediction models using time series cross-validation."""
    
    def __init__(self, initial_train_size: int = None, test_size: int = None, 
                 step_size: int = 1, max_train_size: int = None):
        """
        Initialize backtester.
        
        Args:
            initial_train_size: Initial training window size
            test_size: Test window size for each fold
            step_size: Step size between folds
            max_train_size: Maximum training window size (expanding window if None)
        """
        self.logger = get_logger('evaluation.backtester')
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.max_train_size = max_train_size
        self.evaluator = ModelEvaluator()
        self.backtest_results = {}
    
    def time_series_cross_validation(self, data: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits.
        
        Args:
            data: DataFrame with time series data
            n_splits: Number of splits
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        
        if self.initial_train_size is None:
            self.initial_train_size = n_samples // (n_splits + 1)
        
        if self.test_size is None:
            self.test_size = n_samples // (n_splits * 2)
        
        splits = []
        
        for i in range(n_splits):
            # Calculate split boundaries
            train_start = 0
            train_end = self.initial_train_size + i * self.step_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)
            
            # Apply maximum training size if specified
            if self.max_train_size and (train_end - train_start) > self.max_train_size:
                train_start = train_end - self.max_train_size
            
            # Ensure we have enough data for both train and test
            if test_end <= test_start or train_end <= train_start:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        self.logger.info(f"Created {len(splits)} time series CV splits")
        return splits
    
    def walk_forward_validation(self, data: pd.DataFrame, window_size: int, 
                               step_size: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create walk-forward validation splits with fixed window size.
        
        Args:
            data: DataFrame with time series data
            window_size: Size of training window
            step_size: Step size between windows
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        splits = []
        
        start = 0
        while start + window_size < n_samples:
            train_end = start + window_size
            test_start = train_end
            test_end = min(test_start + step_size, n_samples)
            
            if test_end <= test_start:
                break
            
            train_indices = np.arange(start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            start += step_size
        
        self.logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def expanding_window_validation(self, data: pd.DataFrame, 
                                  min_train_size: int = None,
                                  test_size: int = None,
                                  n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create expanding window validation splits.
        
        Args:
            data: DataFrame with time series data
            min_train_size: Minimum training size
            test_size: Test size for each split
            n_splits: Number of splits
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        
        if min_train_size is None:
            min_train_size = n_samples // 3
        
        if test_size is None:
            test_size = (n_samples - min_train_size) // n_splits
        
        splits = []
        
        for i in range(n_splits):
            train_end = min_train_size + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)
            
            if test_end <= test_start or train_end <= 0:
                break
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        self.logger.info(f"Created {len(splits)} expanding window splits")
        return splits
    
    def backtest_model(self, model_class, data: pd.DataFrame, 
                      feature_columns: List[str], target_column: str,
                      cv_method: str = 'time_series', n_splits: int = 5,
                      model_params: Dict[str, Any] = None,
                      refit_frequency: int = 1) -> Dict[str, Any]:
        """
        Backtest a single model.
        
        Args:
            model_class: Model class to backtest
            data: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Target column name
            cv_method: Cross-validation method ('time_series', 'walk_forward', 'expanding')
            n_splits: Number of splits
            model_params: Model hyperparameters
            refit_frequency: How often to refit the model (1 = every split)
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest for {model_class.__name__}")
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Create CV splits
        if cv_method == 'time_series':
            splits = self.time_series_cross_validation(data, n_splits)
        elif cv_method == 'walk_forward':
            window_size = len(data) // (n_splits + 1)
            splits = self.walk_forward_validation(data, window_size)
        elif cv_method == 'expanding':
            splits = self.expanding_window_validation(data, n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        if not splits:
            raise ModelTrainingError("No valid CV splits created")
        
        # Initialize results storage
        fold_results = []
        all_predictions = []
        all_actuals = []
        model_instance = None
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            try:
                self.logger.debug(f"Processing fold {fold_idx + 1}/{len(splits)}")
                
                # Get fold data
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_test_fold = X[test_idx]
                y_test_fold = y[test_idx]
                
                # Create or reuse model
                if model_instance is None or fold_idx % refit_frequency == 0:
                    model_instance = model_class(f"{model_class.__name__}_backtest_fold_{fold_idx}")
                    if model_params:
                        model_instance.set_hyperparameters(model_params)
                    
                    # Train model
                    model_instance.train(X_train_fold, y_train_fold, feature_names=feature_columns)
                
                # Make predictions
                y_pred_fold = model_instance.predict(X_test_fold)
                
                # Calculate fold metrics
                fold_metrics = self.evaluator.calculate_comprehensive_metrics(
                    y_test_fold, y_pred_fold, f"fold_{fold_idx}"
                )
                
                fold_metrics.update({
                    'fold': fold_idx,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_start': int(train_idx[0]),
                    'train_end': int(train_idx[-1]),
                    'test_start': int(test_idx[0]),
                    'test_end': int(test_idx[-1])
                })
                
                fold_results.append(fold_metrics)
                all_predictions.extend(y_pred_fold)
                all_actuals.extend(y_test_fold)
                
            except Exception as e:
                self.logger.warning(f"Fold {fold_idx} failed: {str(e)}")
                continue
        
        if not fold_results:
            raise ModelTrainingError("All backtest folds failed")
        
        # Calculate overall metrics
        overall_metrics = self.evaluator.calculate_comprehensive_metrics(
            np.array(all_actuals), np.array(all_predictions), 
            f"{model_class.__name__}_overall"
        )
        
        # Calculate fold statistics
        fold_stats = self._calculate_fold_statistics(fold_results)
        
        backtest_results = {
            'model_name': model_class.__name__,
            'cv_method': cv_method,
            'n_splits': len(splits),
            'n_successful_folds': len(fold_results),
            'overall_metrics': overall_metrics,
            'fold_statistics': fold_stats,
            'fold_results': fold_results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'backtest_completed_at': datetime.now().isoformat()
        }
        
        # Store results
        self.backtest_results[model_class.__name__] = backtest_results
        
        self.logger.info(f"Backtest completed for {model_class.__name__}. "
                        f"Overall RMSE: {overall_metrics['rmse']:.4f}")
        
        return backtest_results
    
    def backtest_ensemble(self, models: Dict[str, Any], ensemble_builder,
                         data: pd.DataFrame, feature_columns: List[str], 
                         target_column: str, cv_method: str = 'time_series',
                         n_splits: int = 5) -> Dict[str, Any]:
        """
        Backtest an ensemble of models.
        
        Args:
            models: Dictionary of {model_name: model_instance}
            ensemble_builder: EnsembleBuilder instance
            data: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Target column name
            cv_method: Cross-validation method
            n_splits: Number of splits
            
        Returns:
            Dictionary with ensemble backtest results
        """
        self.logger.info("Starting ensemble backtest")
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Create CV splits
        if cv_method == 'time_series':
            splits = self.time_series_cross_validation(data, n_splits)
        elif cv_method == 'walk_forward':
            window_size = len(data) // (n_splits + 1)
            splits = self.walk_forward_validation(data, window_size)
        elif cv_method == 'expanding':
            splits = self.expanding_window_validation(data, n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        fold_results = []
        all_ensemble_predictions = []
        all_actuals = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            try:
                self.logger.debug(f"Processing ensemble fold {fold_idx + 1}/{len(splits)}")
                
                # Get fold data
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_test_fold = X[test_idx]
                y_test_fold = y[test_idx]
                
                # Train all models for this fold
                fold_models = {}
                fold_predictions = {}
                fold_performance = {}
                
                for model_name, model in models.items():
                    try:
                        # Create model copy for this fold
                        model_copy = type(model)(f"{model_name}_fold_{fold_idx}")
                        model_copy.set_hyperparameters(model.get_hyperparameters())
                        
                        # Train model
                        model_copy.train(X_train_fold, y_train_fold, feature_names=feature_columns)
                        
                        # Get predictions
                        pred = model_copy.predict(X_test_fold)
                        fold_predictions[model_name] = pred
                        
                        # Calculate performance for weighting
                        metrics = self.evaluator.calculate_comprehensive_metrics(
                            y_test_fold, pred, model_name
                        )
                        fold_performance[model_name] = metrics
                        
                        fold_models[model_name] = model_copy
                        
                    except Exception as e:
                        self.logger.warning(f"Model {model_name} failed in fold {fold_idx}: {str(e)}")
                        continue
                
                if not fold_predictions:
                    self.logger.warning(f"No models succeeded in fold {fold_idx}")
                    continue
                
                # Calculate ensemble weights for this fold
                ensemble_builder.calculate_weights(fold_performance)
                
                # Get ensemble prediction
                ensemble_pred = ensemble_builder.weighted_prediction(fold_predictions)
                
                # Calculate ensemble metrics
                ensemble_metrics = self.evaluator.calculate_comprehensive_metrics(
                    y_test_fold, ensemble_pred, f"ensemble_fold_{fold_idx}"
                )
                
                ensemble_metrics.update({
                    'fold': fold_idx,
                    'n_models_used': len(fold_predictions),
                    'model_weights': ensemble_builder.weights.copy(),
                    'individual_model_rmse': {name: perf['rmse'] for name, perf in fold_performance.items()}
                })
                
                fold_results.append(ensemble_metrics)
                all_ensemble_predictions.extend(ensemble_pred)
                all_actuals.extend(y_test_fold)
                
            except Exception as e:
                self.logger.warning(f"Ensemble fold {fold_idx} failed: {str(e)}")
                continue
        
        if not fold_results:
            raise ModelTrainingError("All ensemble backtest folds failed")
        
        # Calculate overall ensemble metrics
        overall_metrics = self.evaluator.calculate_comprehensive_metrics(
            np.array(all_actuals), np.array(all_ensemble_predictions), "ensemble_overall"
        )
        
        # Calculate fold statistics
        fold_stats = self._calculate_fold_statistics(fold_results)
        
        ensemble_results = {
            'model_name': 'ensemble',
            'cv_method': cv_method,
            'n_splits': len(splits),
            'n_successful_folds': len(fold_results),
            'overall_metrics': overall_metrics,
            'fold_statistics': fold_stats,
            'fold_results': fold_results,
            'predictions': all_ensemble_predictions,
            'actuals': all_actuals,
            'ensemble_composition': list(models.keys()),
            'backtest_completed_at': datetime.now().isoformat()
        }
        
        self.backtest_results['ensemble'] = ensemble_results
        
        self.logger.info(f"Ensemble backtest completed. Overall RMSE: {overall_metrics['rmse']:.4f}")
        
        return ensemble_results
    
    def calculate_backtest_metrics(self, predictions: np.ndarray, 
                                 actuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive backtest metrics.
        
        Args:
            predictions: Array of predictions
            actuals: Array of actual values
            
        Returns:
            Dictionary with backtest metrics
        """
        return self.evaluator.calculate_comprehensive_metrics(actuals, predictions, "backtest")
    
    def _calculate_fold_statistics(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics across folds."""
        if not fold_results:
            return {}
        
        metrics = ['rmse', 'mae', 'directional_accuracy', 'r2_score', 'correlation']
        stats = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                stats[f'{metric}_mean'] = float(np.mean(values))
                stats[f'{metric}_std'] = float(np.std(values))
                stats[f'{metric}_min'] = float(np.min(values))
                stats[f'{metric}_max'] = float(np.max(values))
        
        return stats
    
    def compare_backtest_results(self, model_names: List[str] = None) -> pd.DataFrame:
        """
        Compare backtest results across models.
        
        Args:
            model_names: List of model names to compare (all if None)
            
        Returns:
            DataFrame with comparison results
        """
        if model_names is None:
            model_names = list(self.backtest_results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.backtest_results:
                continue
            
            results = self.backtest_results[model_name]
            overall = results['overall_metrics']
            fold_stats = results['fold_statistics']
            
            comparison_data.append({
                'Model': model_name,
                'RMSE': overall['rmse'],
                'RMSE_Std': fold_stats.get('rmse_std', 0),
                'MAE': overall['mae'],
                'MAE_Std': fold_stats.get('mae_std', 0),
                'Directional_Accuracy': overall['directional_accuracy'],
                'Dir_Acc_Std': fold_stats.get('directional_accuracy_std', 0),
                'R2_Score': overall['r2_score'],
                'R2_Std': fold_stats.get('r2_score_std', 0),
                'Successful_Folds': results['n_successful_folds'],
                'Total_Folds': results['n_splits']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('RMSE')
        
        return comparison_df
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of all backtest results."""
        if not self.backtest_results:
            return {'message': 'No backtest results available'}
        
        summary = {
            'n_models_tested': len(self.backtest_results),
            'models': list(self.backtest_results.keys()),
            'best_model_by_rmse': min(
                self.backtest_results.items(), 
                key=lambda x: x[1]['overall_metrics']['rmse']
            )[0],
            'best_model_by_directional_accuracy': max(
                self.backtest_results.items(),
                key=lambda x: x[1]['overall_metrics']['directional_accuracy']
            )[0]
        }
        
        return summary
    
    def save_backtest_results(self, filepath: str) -> None:
        """Save backtest results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.backtest_results.items():
            serializable_results[model_name] = results.copy()
            if 'predictions' in serializable_results[model_name]:
                serializable_results[model_name]['predictions'] = [
                    float(x) for x in serializable_results[model_name]['predictions']
                ]
            if 'actuals' in serializable_results[model_name]:
                serializable_results[model_name]['actuals'] = [
                    float(x) for x in serializable_results[model_name]['actuals']
                ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Backtest results saved to {filepath}")
    
    def load_backtest_results(self, filepath: str) -> None:
        """Load backtest results from file."""
        import json
        
        with open(filepath, 'r') as f:
            self.backtest_results = json.load(f)
        
        self.logger.info(f"Backtest results loaded from {filepath}")