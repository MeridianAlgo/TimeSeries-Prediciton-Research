"""
Mathematical utility functions for enhanced time series prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import torch
import torch.nn.functional as F


class StatisticalUtils:
    """Statistical utility functions."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    @staticmethod
    def calculate_hit_rate(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate hit rate (percentage of correct direction predictions)."""
        if len(predictions) != len(actuals) or len(predictions) == 0:
            return 0.0
            
        correct_directions = np.sum(np.sign(predictions) == np.sign(actuals))
        return correct_directions / len(predictions)
    
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return 0.0
            
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
            
        return np.mean(excess_returns) / tracking_error * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
            
        annual_return = np.mean(returns) * 252
        max_dd = StatisticalUtils.calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return annual_return / max_dd


class TechnicalAnalysisUtils:
    """Technical analysis utility functions."""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(gains))
        avg_losses = np.zeros(len(losses))
        
        # Initial averages
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        # Exponential moving averages
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend NaN for the first price (no delta)
        return np.concatenate([[50.0], rsi])
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, signal line, and histogram."""
        if len(prices) < slow:
            zeros = np.zeros(len(prices))
            return zeros, zeros, zeros
            
        ema_fast = TechnicalAnalysisUtils.ema(prices, fast)
        ema_slow = TechnicalAnalysisUtils.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalysisUtils.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) == 0:
            return np.array([])
            
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
        return ema
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            middle = np.full(len(prices), np.mean(prices))
            return middle, middle, middle
            
        middle = TechnicalAnalysisUtils.sma(prices, period)
        std = TechnicalAnalysisUtils.rolling_std(prices, period)
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.mean(prices))
            
        sma = np.zeros(len(prices))
        sma[:period-1] = np.mean(prices[:period])
        
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
    
    @staticmethod
    def rolling_std(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        if len(prices) < period:
            return np.full(len(prices), np.std(prices))
            
        rolling_std = np.zeros(len(prices))
        rolling_std[:period-1] = np.std(prices[:period])
        
        for i in range(period-1, len(prices)):
            rolling_std[i] = np.std(prices[i-period+1:i+1])
            
        return rolling_std
    
    @staticmethod
    def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator %K and %D."""
        if len(high) < k_period:
            k_values = np.full(len(close), 50.0)
            d_values = np.full(len(close), 50.0)
            return k_values, d_values
            
        k_values = np.zeros(len(close))
        
        for i in range(k_period-1, len(close)):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            
            if highest_high == lowest_low:
                k_values[i] = 50.0
            else:
                k_values[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Fill initial values
        k_values[:k_period-1] = k_values[k_period-1]
        
        # Calculate %D (moving average of %K)
        d_values = TechnicalAnalysisUtils.sma(k_values, d_period)
        
        return k_values, d_values


class OptimizationUtils:
    """Optimization utility functions."""
    
    @staticmethod
    def optimize_portfolio_weights(returns: np.ndarray, method: str = 'sharpe') -> np.ndarray:
        """Optimize portfolio weights."""
        n_assets = returns.shape[1]
        
        def objective(weights):
            portfolio_return = np.sum(returns.mean(axis=0) * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T) * 252, weights)))
            
            if method == 'sharpe':
                return -portfolio_return / portfolio_std if portfolio_std > 0 else 0
            elif method == 'min_variance':
                return portfolio_std
            else:
                raise ValueError(f"Unknown optimization method: {method}")
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0/n_assets] * n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_guess
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion for position sizing."""
        if avg_loss == 0:
            return 0.0
            
        b = avg_win / avg_loss  # Ratio of win to loss
        p = win_rate  # Probability of winning
        q = 1 - p  # Probability of losing
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 25% for safety
        return max(0, min(0.25, kelly_fraction))


class UncertaintyUtils:
    """Uncertainty quantification utilities."""
    
    @staticmethod
    def monte_carlo_dropout_prediction(model: torch.nn.Module, x: torch.Tensor, 
                                     n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo dropout for uncertainty estimation."""
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred
    
    @staticmethod
    def ensemble_uncertainty(predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate ensemble uncertainty from multiple model predictions."""
        if not predictions:
            raise ValueError("No predictions provided")
            
        stacked_preds = torch.stack(predictions)
        mean_pred = torch.mean(stacked_preds, dim=0)
        std_pred = torch.std(stacked_preds, dim=0)
        
        return mean_pred, std_pred
    
    @staticmethod
    def confidence_interval(mean: float, std: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std
        
        return mean - margin, mean + margin


class PerformanceUtils:
    """Performance measurement utilities."""
    
    @staticmethod
    def calculate_metrics_batch(predictions: np.ndarray, actuals: np.ndarray) -> dict:
        """Calculate comprehensive performance metrics."""
        if len(predictions) != len(actuals) or len(predictions) == 0:
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'directional_accuracy': 0.0,
                'hit_rate': 0.0,
                'correlation': 0.0
            }
        
        # Basic error metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / np.abs(actuals))) * 100
        
        # Directional accuracy
        directional_accuracy = StatisticalUtils.calculate_hit_rate(predictions, actuals)
        
        # Correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'hit_rate': directional_accuracy,
            'correlation': correlation
        }
    
    @staticmethod
    def rolling_performance(predictions: np.ndarray, actuals: np.ndarray, 
                          window: int = 50) -> dict:
        """Calculate rolling performance metrics."""
        if len(predictions) < window:
            return PerformanceUtils.calculate_metrics_batch(predictions, actuals)
        
        rolling_metrics = {
            'mae': [],
            'rmse': [],
            'directional_accuracy': []
        }
        
        for i in range(window, len(predictions) + 1):
            window_preds = predictions[i-window:i]
            window_actuals = actuals[i-window:i]
            
            metrics = PerformanceUtils.calculate_metrics_batch(window_preds, window_actuals)
            rolling_metrics['mae'].append(metrics['mae'])
            rolling_metrics['rmse'].append(metrics['rmse'])
            rolling_metrics['directional_accuracy'].append(metrics['directional_accuracy'])
        
        return rolling_metrics