"""Model evaluation utilities for stock price prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings

from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import ModelTrainingError


class ModelEvaluator:
    """Evaluates and compares model performance using multiple metrics."""
    
    def __init__(self):
        self.logger = get_logger('evaluation.evaluator')
        self.evaluation_results = {}
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE score
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE score
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE score
        """
        return float(mean_squared_error(y_true, y_pred))
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct up/down predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy as percentage (0-100)
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted directions (up/down from previous value)
        actual_directions = np.diff(y_true) > 0
        predicted_directions = np.diff(y_pred) > 0
        
        # Calculate accuracy
        correct_directions = actual_directions == predicted_directions
        accuracy = np.mean(correct_directions) * 100
        
        return float(accuracy)
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE score as percentage
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)
    
    def calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R-squared score
        """
        return float(r2_score(y_true, y_pred))
    
    def calculate_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Correlation coefficient
        """
        correlation, _ = stats.pearsonr(y_true, y_pred)
        return float(correlation)
    
    def calculate_max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate maximum absolute error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Maximum absolute error
        """
        return float(np.max(np.abs(y_true - y_pred)))
    
    def calculate_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate prediction bias (mean error).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Bias (positive = overestimation, negative = underestimation)
        """
        return float(np.mean(y_pred - y_true))
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      model_name: str = "model") -> Dict[str, float]:
        """
        Calculate comprehensive set of evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'mae': self.calculate_mae(y_true, y_pred),
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mse': self.calculate_mse(y_true, y_pred),
            'directional_accuracy': self.calculate_directional_accuracy(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred),
            'r2_score': self.calculate_r2_score(y_true, y_pred),
            'correlation': self.calculate_correlation(y_true, y_pred),
            'max_error': self.calculate_max_error(y_true, y_pred),
            'bias': self.calculate_bias(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        self.logger.info(f"Evaluation completed for {model_name}: "
                        f"RMSE={metrics['rmse']:.4f}, "
                        f"MAE={metrics['mae']:.4f}, "
                        f"Dir_Acc={metrics['directional_accuracy']:.2f}%")
        
        return metrics
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = None) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name is None:
            model_name = getattr(model, 'name', 'unknown_model')
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, model_name)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed for {model_name}: {str(e)}")
            raise ModelTrainingError(f"Model evaluation failed: {str(e)}")
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models on the same test data.
        
        Args:
            models: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                metrics['model_name'] = model_name
                comparison_results.append(metrics)
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                continue
        
        if not comparison_results:
            raise ModelTrainingError("No models could be evaluated")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('rmse')
        
        self.logger.info(f"Model comparison completed for {len(comparison_results)} models")
        
        return comparison_df
    
    def generate_performance_report(self, models: Dict[str, Any] = None, 
                                  predictions: Dict[str, np.ndarray] = None,
                                  y_true: np.ndarray = None) -> pd.DataFrame:
        """
        Generate comprehensive performance report.
        
        Args:
            models: Dictionary of models (optional)
            predictions: Dictionary of {model_name: predictions} (optional)
            y_true: True values (required if predictions provided)
            
        Returns:
            DataFrame with performance report
        """
        if predictions is not None and y_true is not None:
            # Evaluate from predictions
            for model_name, y_pred in predictions.items():
                self.calculate_comprehensive_metrics(y_true, y_pred, model_name)
        
        if not self.evaluation_results:
            raise ModelTrainingError("No evaluation results available")
        
        # Create report DataFrame
        report_data = []
        for model_name, metrics in self.evaluation_results.items():
            report_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'R²': metrics['r2_score'],
                'Directional Accuracy (%)': metrics['directional_accuracy'],
                'Correlation': metrics['correlation'],
                'Bias': metrics['bias'],
                'Max Error': metrics['max_error'],
                'Samples': metrics['n_samples']
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by RMSE
        report_df = report_df.sort_values('RMSE')
        
        return report_df
    
    def statistical_significance_test(self, y_true: np.ndarray, 
                                    predictions1: np.ndarray, predictions2: np.ndarray,
                                    model1_name: str = "Model 1", 
                                    model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Test statistical significance of difference between two models.
        
        Args:
            y_true: True values
            predictions1: Predictions from first model
            predictions2: Predictions from second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary with test results
        """
        # Calculate squared errors for both models
        se1 = (y_true - predictions1) ** 2
        se2 = (y_true - predictions2) ** 2
        
        # Calculate difference in squared errors
        se_diff = se1 - se2
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(se1, se2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(se1) + np.var(se2)) / 2)
        cohens_d = np.mean(se_diff) / pooled_std if pooled_std > 0 else 0
        
        # Determine significance
        is_significant = p_value < 0.05
        better_model = model1_name if np.mean(se1) < np.mean(se2) else model2_name
        
        results = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_mse': float(np.mean(se1)),
            'model2_mse': float(np.mean(se2)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'is_significant': is_significant,
            'better_model': better_model,
            'interpretation': self._interpret_significance_test(p_value, cohens_d, better_model)
        }
        
        return results
    
    def _interpret_significance_test(self, p_value: float, cohens_d: float, 
                                   better_model: str) -> str:
        """Interpret statistical significance test results."""
        if p_value >= 0.05:
            return "No significant difference between models"
        
        effect_size = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        
        return f"{better_model} is significantly better (p={p_value:.4f}, {effect_size} effect size)"
    
    def cross_validation_evaluation(self, model, X: np.ndarray, y: np.ndarray, 
                                  cv_folds: int = 5, time_series: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            time_series: Whether to use time series CV (no shuffling)
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import TimeSeriesSplit, KFold
        
        # Choose CV strategy
        if time_series:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'directional_accuracy': [],
            'r2_score': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            try:
                # Split data
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train model (create a copy to avoid modifying original)
                model_copy = type(model)(model.name + f"_fold_{fold}")
                model_copy.set_hyperparameters(model.get_hyperparameters())
                model_copy.train(X_train_fold, y_train_fold)
                
                # Evaluate
                y_pred_fold = model_copy.predict(X_val_fold)
                
                # Calculate metrics
                cv_scores['rmse'].append(self.calculate_rmse(y_val_fold, y_pred_fold))
                cv_scores['mae'].append(self.calculate_mae(y_val_fold, y_pred_fold))
                cv_scores['directional_accuracy'].append(
                    self.calculate_directional_accuracy(y_val_fold, y_pred_fold)
                )
                cv_scores['r2_score'].append(self.calculate_r2_score(y_val_fold, y_pred_fold))
                
            except Exception as e:
                self.logger.warning(f"CV fold {fold} failed: {str(e)}")
                continue
        
        # Calculate summary statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            if scores:  # Only if we have valid scores
                cv_results[f'{metric}_mean'] = float(np.mean(scores))
                cv_results[f'{metric}_std'] = float(np.std(scores))
                cv_results[f'{metric}_scores'] = scores
        
        cv_results['n_folds_completed'] = len(cv_scores['rmse'])
        cv_results['cv_strategy'] = 'TimeSeriesSplit' if time_series else 'KFold'
        
        return cv_results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        if not self.evaluation_results:
            return {'message': 'No evaluations performed yet'}
        
        summary = {
            'n_models_evaluated': len(self.evaluation_results),
            'models': list(self.evaluation_results.keys()),
            'best_model_by_rmse': min(self.evaluation_results.items(), 
                                    key=lambda x: x[1]['rmse'])[0],
            'best_model_by_mae': min(self.evaluation_results.items(), 
                                   key=lambda x: x[1]['mae'])[0],
            'best_model_by_directional_accuracy': max(self.evaluation_results.items(), 
                                                    key=lambda x: x[1]['directional_accuracy'])[0]
        }
        
        return summary