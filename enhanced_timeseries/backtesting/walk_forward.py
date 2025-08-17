"""
Walk-forward analysis engine for time series backtesting.
Implements proper temporal validation with model retraining and performance tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
from ..core.interfaces import BasePredictor, BaseBacktester, PerformanceMetrics
from ..utils.math_utils import PerformanceUtils, StatisticalUtils
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtesting."""
    initial_train_size: int = 252  # Initial training window (1 year)
    retraining_frequency: int = 21  # Retrain every 21 days (monthly)
    test_size: int = 21  # Test window size
    min_train_size: int = 126  # Minimum training size (6 months)
    max_train_size: Optional[int] = None  # Maximum training size (None = unlimited)
    expanding_window: bool = True  # True for expanding, False for rolling window
    purge_days: int = 0  # Days to purge between train and test
    embargo_days: int = 0  # Days to embargo after test period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BacktestPeriod:
    """Single backtesting period information."""
    period_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_id': self.period_id,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'train_size': self.train_size,
            'test_size': self.test_size
        }


@dataclass
class BacktestResult:
    """Results from a single backtesting period."""
    period: BacktestPeriod
    predictions: List[float]
    actuals: List[float]
    timestamps: List[pd.Timestamp]
    metrics: PerformanceMetrics
    training_time: float
    prediction_time: float
    model_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period': self.period.to_dict(),
            'predictions': self.predictions,
            'actuals': self.actuals,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'metrics': asdict(self.metrics),
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'model_info': self.model_info
        }


class WalkForwardBacktester(BaseBacktester):
    """
    Walk-forward analysis engine with proper temporal validation.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = []
        self.summary_metrics = None
        
    def run_backtest(self, data: pd.DataFrame, predictor_factory: Callable,
                    target_column: str = 'target', **kwargs) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            data: Time series data with datetime index
            predictor_factory: Function that creates a new predictor instance
            target_column: Name of target column
            **kwargs: Additional arguments for predictor training
            
        Returns:
            Comprehensive backtest results
        """
        print("🚀 Starting Walk-Forward Backtesting...")
        
        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Sort data by date
        data = data.sort_index()
        
        # Generate backtesting periods
        periods = self._generate_periods(data)
        print(f"📊 Generated {len(periods)} backtesting periods")
        
        # Run backtest for each period
        self.results = []
        
        for i, period in enumerate(periods):
            print(f"\n🔄 Period {i+1}/{len(periods)}: {period.test_start.date()} to {period.test_end.date()}")
            
            try:
                result = self._run_single_period(data, period, predictor_factory, target_column, **kwargs)
                self.results.append(result)
                
                # Print period results
                print(f"   📈 MAE: {result.metrics.mae:.4f}")
                print(f"   🎯 Directional Accuracy: {result.metrics.directional_accuracy:.1f}%")
                print(f"   ⏱️  Training Time: {result.training_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Error in period {i+1}: {e}")
                continue
        
        # Calculate summary metrics
        self.summary_metrics = self._calculate_summary_metrics()
        
        print(f"\n📊 Backtest Complete!")
        print(f"   📈 Overall MAE: {self.summary_metrics['overall_mae']:.4f}")
        print(f"   🎯 Overall Directional Accuracy: {self.summary_metrics['overall_directional_accuracy']:.1f}%")
        print(f"   📊 Sharpe Ratio: {self.summary_metrics['sharpe_ratio']:.2f}")
        
        return {
            'config': self.config.to_dict(),
            'periods': [result.to_dict() for result in self.results],
            'summary_metrics': self.summary_metrics,
            'n_periods': len(self.results)
        }
    
    def _generate_periods(self, data: pd.DataFrame) -> List[BacktestPeriod]:
        """Generate backtesting periods."""
        periods = []
        period_id = 0
        
        # Start after initial training period
        current_idx = self.config.initial_train_size
        
        while current_idx + self.config.test_size <= len(data):
            # Training period
            if self.config.expanding_window:
                train_start_idx = 0
            else:
                # Rolling window
                if self.config.max_train_size:
                    train_start_idx = max(0, current_idx - self.config.max_train_size)
                else:
                    train_start_idx = max(0, current_idx - self.config.initial_train_size)
            
            train_end_idx = current_idx - 1
            
            # Apply purge
            if self.config.purge_days > 0:
                train_end_idx = max(train_start_idx, train_end_idx - self.config.purge_days)
            
            # Test period
            test_start_idx = current_idx
            test_end_idx = min(len(data) - 1, current_idx + self.config.test_size - 1)
            
            # Check minimum training size
            train_size = train_end_idx - train_start_idx + 1
            if train_size < self.config.min_train_size:
                current_idx += self.config.retraining_frequency
                continue
            
            # Create period
            period = BacktestPeriod(
                period_id=period_id,
                train_start=data.index[train_start_idx],
                train_end=data.index[train_end_idx],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx],
                train_size=train_size,
                test_size=test_end_idx - test_start_idx + 1
            )
            
            periods.append(period)
            period_id += 1
            
            # Move to next period
            current_idx += self.config.retraining_frequency
            
            # Apply embargo
            if self.config.embargo_days > 0:
                current_idx += self.config.embargo_days
        
        return periods
    
    def _run_single_period(self, data: pd.DataFrame, period: BacktestPeriod,
                          predictor_factory: Callable, target_column: str, **kwargs) -> BacktestResult:
        """Run backtest for a single period."""
        import time
        
        # Extract training and test data
        train_data = data.loc[period.train_start:period.train_end].copy()
        test_data = data.loc[period.test_start:period.test_end].copy()
        
        # Create predictor
        predictor = predictor_factory()
        
        # Train predictor
        start_time = time.time()
        
        try:
            training_metrics = predictor.train(train_data, **kwargs)
        except Exception as e:
            print(f"Training failed: {e}")
            training_metrics = {'mae': float('inf')}
        
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        
        predictions = []
        actuals = []
        timestamps = []
        
        for idx in test_data.index:
            try:
                # Get data up to current point (avoid look-ahead bias)
                current_data = data.loc[:idx].copy()
                
                # Make prediction
                if hasattr(predictor, 'predict'):
                    pred_result = predictor.predict(current_data.tail(1))
                    
                    if hasattr(pred_result, 'prediction'):
                        prediction = pred_result.prediction
                    else:
                        prediction = float(pred_result)
                else:
                    prediction = 0.0
                
                # Get actual value
                actual = test_data.loc[idx, target_column]
                
                predictions.append(prediction)
                actuals.append(actual)
                timestamps.append(idx)
                
            except Exception as e:
                print(f"Prediction failed for {idx}: {e}")
                continue
        
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        if len(predictions) > 0:
            metrics = self.calculate_performance_metrics(predictions, actuals)
        else:
            metrics = PerformanceMetrics(
                mae=float('inf'), rmse=float('inf'), directional_accuracy=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, hit_rate=0.0,
                profit_factor=0.0, regime_performance={}
            )
        
        # Get model info
        if hasattr(predictor, 'get_predictor_info'):
            model_info = predictor.get_predictor_info()
        else:
            model_info = {'type': type(predictor).__name__}
        
        return BacktestResult(
            period=period,
            predictions=predictions,
            actuals=actuals,
            timestamps=timestamps,
            metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            model_info=model_info
        )
    
    def calculate_performance_metrics(self, predictions: List[float], 
                                    actuals: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if len(predictions) == 0 or len(actuals) == 0:
            return PerformanceMetrics(
                mae=float('inf'), rmse=float('inf'), directional_accuracy=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, hit_rate=0.0,
                profit_factor=0.0, regime_performance={}
            )
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Basic metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Directional accuracy
        directional_accuracy = StatisticalUtils.calculate_hit_rate(predictions, actuals) * 100
        
        # Financial metrics (assuming predictions are returns)
        returns = predictions  # Treat predictions as expected returns
        
        # Sharpe ratio
        sharpe_ratio = StatisticalUtils.calculate_sharpe_ratio(returns)
        
        # Max drawdown
        max_drawdown = StatisticalUtils.calculate_max_drawdown(returns)
        
        # Hit rate (same as directional accuracy)
        hit_rate = directional_accuracy / 100
        
        # Profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) > 0 and np.sum(negative_returns) != 0:
            profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns))
        else:
            profit_factor = float('inf') if len(positive_returns) > 0 else 0.0
        
        return PerformanceMetrics(
            mae=mae,
            rmse=rmse,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            regime_performance={}  # Will be filled by regime analysis
        )
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics across all periods."""
        if not self.results:
            return {}
        
        # Aggregate predictions and actuals
        all_predictions = []
        all_actuals = []
        all_returns = []
        
        for result in self.results:
            all_predictions.extend(result.predictions)
            all_actuals.extend(result.actuals)
            all_returns.extend(result.predictions)  # Assuming predictions are returns
        
        # Overall metrics
        overall_metrics = self.calculate_performance_metrics(all_predictions, all_actuals)
        
        # Period-wise statistics
        period_maes = [result.metrics.mae for result in self.results if np.isfinite(result.metrics.mae)]
        period_accuracies = [result.metrics.directional_accuracy for result in self.results]
        period_sharpes = [result.metrics.sharpe_ratio for result in self.results if np.isfinite(result.metrics.sharpe_ratio)]
        
        # Training and prediction times
        total_training_time = sum(result.training_time for result in self.results)
        total_prediction_time = sum(result.prediction_time for result in self.results)
        
        # Consistency metrics
        mae_std = np.std(period_maes) if period_maes else 0.0
        accuracy_std = np.std(period_accuracies) if period_accuracies else 0.0
        
        # Win rate (percentage of profitable periods)
        profitable_periods = sum(1 for result in self.results 
                               if result.metrics.sharpe_ratio > 0 and np.isfinite(result.metrics.sharpe_ratio))
        win_rate = profitable_periods / len(self.results) if self.results else 0.0
        
        return {
            # Overall performance
            'overall_mae': overall_metrics.mae,
            'overall_rmse': overall_metrics.rmse,
            'overall_directional_accuracy': overall_metrics.directional_accuracy,
            'sharpe_ratio': overall_metrics.sharpe_ratio,
            'max_drawdown': overall_metrics.max_drawdown,
            'profit_factor': overall_metrics.profit_factor,
            
            # Period statistics
            'n_periods': len(self.results),
            'avg_period_mae': np.mean(period_maes) if period_maes else float('inf'),
            'median_period_mae': np.median(period_maes) if period_maes else float('inf'),
            'mae_std': mae_std,
            'avg_period_accuracy': np.mean(period_accuracies) if period_accuracies else 0.0,
            'accuracy_std': accuracy_std,
            'avg_period_sharpe': np.mean(period_sharpes) if period_sharpes else 0.0,
            
            # Consistency metrics
            'win_rate': win_rate,
            'consistency_score': 1.0 - (mae_std / np.mean(period_maes)) if period_maes and np.mean(period_maes) > 0 else 0.0,
            
            # Timing
            'total_training_time': total_training_time,
            'total_prediction_time': total_prediction_time,
            'avg_training_time_per_period': total_training_time / len(self.results) if self.results else 0.0,
            'avg_prediction_time_per_period': total_prediction_time / len(self.results) if self.results else 0.0,
            
            # Data coverage
            'total_predictions': len(all_predictions),
            'date_range': {
                'start': self.results[0].period.test_start.isoformat() if self.results else None,
                'end': self.results[-1].period.test_end.isoformat() if self.results else None
            }
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        records = []
        
        for result in self.results:
            for i, (pred, actual, timestamp) in enumerate(zip(
                result.predictions, result.actuals, result.timestamps
            )):
                records.append({
                    'period_id': result.period.period_id,
                    'timestamp': timestamp,
                    'prediction': pred,
                    'actual': actual,
                    'error': abs(pred - actual),
                    'squared_error': (pred - actual) ** 2,
                    'directional_correct': np.sign(pred) == np.sign(actual),
                    'train_start': result.period.train_start,
                    'train_end': result.period.train_end,
                    'train_size': result.period.train_size
                })
        
        return pd.DataFrame(records)
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = self.get_results_dataframe()
            
            if df.empty:
                print("No results to plot")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Predictions vs Actuals over time
            axes[0, 0].plot(df['timestamp'], df['prediction'], label='Predictions', alpha=0.7)
            axes[0, 0].plot(df['timestamp'], df['actual'], label='Actuals', alpha=0.7)
            axes[0, 0].set_title('Predictions vs Actuals Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Error distribution
            axes[0, 1].hist(df['error'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Prediction Error Distribution')
            axes[0, 1].set_xlabel('Absolute Error')
            axes[0, 1].set_ylabel('Frequency')
            
            # 3. Rolling MAE by period
            period_metrics = df.groupby('period_id').agg({
                'error': 'mean',
                'directional_correct': 'mean',
                'timestamp': 'first'
            }).reset_index()
            
            axes[1, 0].plot(period_metrics['timestamp'], period_metrics['error'])
            axes[1, 0].set_title('Rolling MAE by Period')
            axes[1, 0].set_xlabel('Period Start Date')
            axes[1, 0].set_ylabel('Mean Absolute Error')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Directional accuracy by period
            axes[1, 1].plot(period_metrics['timestamp'], period_metrics['directional_correct'] * 100)
            axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
            axes[1, 1].set_title('Directional Accuracy by Period')
            axes[1, 1].set_xlabel('Period Start Date')
            axes[1, 1].set_ylabel('Directional Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def export_results(self, filepath: str, format: str = 'csv') -> None:
        """Export results to file."""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No results to export")
            return
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {filepath}")


def create_walk_forward_backtester(config: Dict[str, Any]) -> WalkForwardBacktester:
    """Factory function to create walk-forward backtester."""
    backtest_config = BacktestConfig(**config)
    return WalkForwardBacktester(backtest_config)