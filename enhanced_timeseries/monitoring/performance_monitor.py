"""
Real-time performance monitoring system for time series prediction models.
Tracks accuracy, performance metrics, and model health with dashboard capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import threading
import time
import json
from collections import deque, defaultdict
from pathlib import Path
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics."""
    model_id: str
    timestamp: datetime
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float
    prediction_count: int
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'hit_rate': self.hit_rate,
            'profit_factor': self.profit_factor,
            'prediction_count': self.prediction_count,
            'confidence_score': self.confidence_score
        }


@dataclass
class EnsemblePerformanceMetrics:
    """Container for ensemble performance metrics."""
    timestamp: datetime
    ensemble_mae: float
    ensemble_rmse: float
    ensemble_accuracy: float
    model_agreement: float
    diversity_score: float
    best_model_id: str
    worst_model_id: str
    model_weights: Dict[str, float]
    prediction_variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ensemble_mae': self.ensemble_mae,
            'ensemble_rmse': self.ensemble_rmse,
            'ensemble_accuracy': self.ensemble_accuracy,
            'model_agreement': self.model_agreement,
            'diversity_score': self.diversity_score,
            'best_model_id': self.best_model_id,
            'worst_model_id': self.worst_model_id,
            'model_weights': self.model_weights,
            'prediction_variance': self.prediction_variance
        }


@dataclass
class SystemHealthMetrics:
    """Container for system health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    prediction_latency: float
    throughput: float  # predictions per second
    error_rate: float
    uptime: float  # hours
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'prediction_latency': self.prediction_latency,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'uptime': self.uptime
        }


class MetricsCalculator:
    """Calculate various performance metrics for model monitoring."""
    
    @staticmethod
    def calculate_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(predictions - actuals)))
    
    @staticmethod
    def calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    
    @staticmethod
    def calculate_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = actuals != 0
        if np.sum(mask) == 0:
            return 0.0
        return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100)
    
    @staticmethod
    def calculate_directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        if len(predictions) <= 1:
            return 0.0
        
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        
        correct_directions = np.sum(pred_direction == actual_direction)
        total_directions = len(pred_direction)
        
        return float(correct_directions / total_directions * 100) if total_directions > 0 else 0.0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(abs(np.min(drawdown)))
    
    @staticmethod
    def calculate_hit_rate(predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate hit rate (percentage of predictions within threshold)."""
        if len(predictions) == 0:
            return 0.0
        
        hits = np.abs(predictions - actuals) <= threshold
        return float(np.mean(hits) * 100)
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
        
        return float(gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0


class RealTimeMetricsTracker:
    """Real-time tracking of model performance metrics."""
    
    def __init__(self, max_history: int = 10000, update_interval: int = 60):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of historical records to keep
            update_interval: Update interval in seconds
        """
        self.max_history = max_history
        self.update_interval = update_interval
        
        # Storage for metrics
        self.model_metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.ensemble_metrics_history = deque(maxlen=max_history)
        self.system_health_history = deque(maxlen=max_history)
        
        # Current predictions and actuals for real-time calculation
        self.current_predictions = defaultdict(list)
        self.current_actuals = defaultdict(list)
        self.prediction_timestamps = defaultdict(list)
        
        # System monitoring
        self.start_time = datetime.now()
        self.prediction_count = 0
        self.error_count = 0
        self.latency_measurements = deque(maxlen=1000)
        
        # Threading
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = None
        
        self.calculator = MetricsCalculator()
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def add_prediction(self, model_id: str, prediction: float, actual: Optional[float] = None, 
                      confidence: float = 1.0, timestamp: Optional[datetime] = None):
        """
        Add a new prediction for tracking.
        
        Args:
            model_id: Identifier for the model
            prediction: Predicted value
            actual: Actual value (if available)
            confidence: Prediction confidence score
            timestamp: Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_predictions[model_id].append(prediction)
        self.prediction_timestamps[model_id].append(timestamp)
        
        if actual is not None:
            self.current_actuals[model_id].append(actual)
        
        self.prediction_count += 1
        
        # Clean old data
        self._cleanup_old_data(model_id)
    
    def add_actual(self, model_id: str, actual: float, timestamp: Optional[datetime] = None):
        """
        Add actual value for a previous prediction.
        
        Args:
            model_id: Identifier for the model
            actual: Actual value
            timestamp: Timestamp of the actual value
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_actuals[model_id].append(actual)
    
    def record_latency(self, latency_ms: float):
        """Record prediction latency."""
        self.latency_measurements.append(latency_ms)
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def calculate_model_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Calculate current metrics for a specific model."""
        if (model_id not in self.current_predictions or 
            model_id not in self.current_actuals or
            len(self.current_predictions[model_id]) == 0 or
            len(self.current_actuals[model_id]) == 0):
            return None
        
        predictions = np.array(self.current_predictions[model_id])
        actuals = np.array(self.current_actuals[model_id])
        
        # Align predictions and actuals (take minimum length)
        min_len = min(len(predictions), len(actuals))
        if min_len == 0:
            return None
        
        predictions = predictions[-min_len:]
        actuals = actuals[-min_len:]
        
        # Calculate metrics
        mae = self.calculator.calculate_mae(predictions, actuals)
        rmse = self.calculator.calculate_rmse(predictions, actuals)
        mape = self.calculator.calculate_mape(predictions, actuals)
        directional_accuracy = self.calculator.calculate_directional_accuracy(predictions, actuals)
        
        # Calculate returns for financial metrics
        returns = predictions  # Assuming predictions are returns
        sharpe_ratio = self.calculator.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculator.calculate_max_drawdown(returns)
        hit_rate = self.calculator.calculate_hit_rate(predictions, actuals, threshold=0.01)
        profit_factor = self.calculator.calculate_profit_factor(returns)
        
        # Average confidence (placeholder - would need to track confidence per prediction)
        confidence_score = 0.8  # Default confidence
        
        return ModelPerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            prediction_count=len(predictions),
            confidence_score=confidence_score
        )
    
    def calculate_ensemble_metrics(self, model_weights: Optional[Dict[str, float]] = None) -> Optional[EnsemblePerformanceMetrics]:
        """Calculate ensemble performance metrics."""
        if not self.current_predictions:
            return None
        
        model_ids = list(self.current_predictions.keys())
        if len(model_ids) < 2:
            return None
        
        # Calculate individual model metrics
        model_metrics = {}
        for model_id in model_ids:
            metrics = self.calculate_model_metrics(model_id)
            if metrics:
                model_metrics[model_id] = metrics
        
        if len(model_metrics) < 2:
            return None
        
        # Calculate ensemble metrics
        ensemble_mae = np.mean([m.mae for m in model_metrics.values()])
        ensemble_rmse = np.mean([m.rmse for m in model_metrics.values()])
        ensemble_accuracy = np.mean([m.directional_accuracy for m in model_metrics.values()])
        
        # Model agreement (correlation between predictions)
        model_agreement = self._calculate_model_agreement(model_ids)
        
        # Diversity score (inverse of agreement)
        diversity_score = 1.0 - model_agreement
        
        # Best and worst models
        best_model_id = min(model_metrics.keys(), key=lambda k: model_metrics[k].mae)
        worst_model_id = max(model_metrics.keys(), key=lambda k: model_metrics[k].mae)
        
        # Model weights (equal if not provided)
        if model_weights is None:
            model_weights = {model_id: 1.0 / len(model_ids) for model_id in model_ids}
        
        # Prediction variance
        prediction_variance = self._calculate_prediction_variance(model_ids)
        
        return EnsemblePerformanceMetrics(
            timestamp=datetime.now(),
            ensemble_mae=ensemble_mae,
            ensemble_rmse=ensemble_rmse,
            ensemble_accuracy=ensemble_accuracy,
            model_agreement=model_agreement,
            diversity_score=diversity_score,
            best_model_id=best_model_id,
            worst_model_id=worst_model_id,
            model_weights=model_weights,
            prediction_variance=prediction_variance
        )
    
    def calculate_system_health(self) -> SystemHealthMetrics:
        """Calculate system health metrics."""
        import psutil
        
        # CPU and memory usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # GPU usage (if available)
        gpu_usage = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except ImportError:
            pass
        
        # Prediction latency
        avg_latency = np.mean(self.latency_measurements) if self.latency_measurements else 0.0
        
        # Throughput (predictions per second)
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        throughput = self.prediction_count / uptime_seconds if uptime_seconds > 0 else 0.0
        
        # Error rate
        error_rate = (self.error_count / self.prediction_count * 100) if self.prediction_count > 0 else 0.0
        
        # Uptime in hours
        uptime_hours = uptime_seconds / 3600
        
        return SystemHealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            prediction_latency=avg_latency,
            throughput=throughput,
            error_rate=error_rate,
            uptime=uptime_hours
        )
    
    def _calculate_model_agreement(self, model_ids: List[str]) -> float:
        """Calculate agreement between models."""
        if len(model_ids) < 2:
            return 1.0
        
        correlations = []
        
        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids):
                if i < j:
                    pred1 = self.current_predictions[model1]
                    pred2 = self.current_predictions[model2]
                    
                    if len(pred1) > 1 and len(pred2) > 1:
                        min_len = min(len(pred1), len(pred2))
                        corr = np.corrcoef(pred1[-min_len:], pred2[-min_len:])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _calculate_prediction_variance(self, model_ids: List[str]) -> float:
        """Calculate variance in predictions across models."""
        if len(model_ids) < 2:
            return 0.0
        
        # Get latest predictions from each model
        latest_predictions = []
        for model_id in model_ids:
            if self.current_predictions[model_id]:
                latest_predictions.append(self.current_predictions[model_id][-1])
        
        return float(np.var(latest_predictions)) if len(latest_predictions) > 1 else 0.0
    
    def _cleanup_old_data(self, model_id: str):
        """Clean up old prediction data."""
        max_size = 1000  # Keep last 1000 predictions per model
        
        if len(self.current_predictions[model_id]) > max_size:
            self.current_predictions[model_id] = self.current_predictions[model_id][-max_size:]
        
        if len(self.current_actuals[model_id]) > max_size:
            self.current_actuals[model_id] = self.current_actuals[model_id][-max_size:]
        
        if len(self.prediction_timestamps[model_id]) > max_size:
            self.prediction_timestamps[model_id] = self.prediction_timestamps[model_id][-max_size:]
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Calculate and store metrics for all models
                for model_id in list(self.current_predictions.keys()):
                    metrics = self.calculate_model_metrics(model_id)
                    if metrics:
                        self.model_metrics_history[model_id].append(metrics)
                
                # Calculate and store ensemble metrics
                ensemble_metrics = self.calculate_ensemble_metrics()
                if ensemble_metrics:
                    self.ensemble_metrics_history.append(ensemble_metrics)
                
                # Calculate and store system health
                system_health = self.calculate_system_health()
                self.system_health_history.append(system_health)
                
                # Wait for next update
                self._stop_monitoring.wait(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.record_error()
    
    def get_recent_metrics(self, model_id: str, hours: int = 24) -> List[ModelPerformanceMetrics]:
        """Get recent metrics for a model."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = []
        for metrics in self.model_metrics_history[model_id]:
            if metrics.timestamp >= cutoff_time:
                recent_metrics.append(metrics)
        
        return recent_metrics
    
    def get_recent_ensemble_metrics(self, hours: int = 24) -> List[EnsemblePerformanceMetrics]:
        """Get recent ensemble metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = []
        for metrics in self.ensemble_metrics_history:
            if metrics.timestamp >= cutoff_time:
                recent_metrics.append(metrics)
        
        return recent_metrics
    
    def get_recent_system_health(self, hours: int = 24) -> List[SystemHealthMetrics]:
        """Get recent system health metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = []
        for metrics in self.system_health_history:
            if metrics.timestamp >= cutoff_time:
                recent_metrics.append(metrics)
        
        return recent_metrics


class PerformanceDashboard:
    """Web-based performance monitoring dashboard."""
    
    def __init__(self, metrics_tracker: RealTimeMetricsTracker, 
                 dashboard_port: int = 8050, auto_refresh: int = 30):
        """
        Initialize performance dashboard.
        
        Args:
            metrics_tracker: Real-time metrics tracker
            dashboard_port: Port for web dashboard
            auto_refresh: Auto-refresh interval in seconds
        """
        self.metrics_tracker = metrics_tracker
        self.dashboard_port = dashboard_port
        self.auto_refresh = auto_refresh
        
        # Dashboard data
        self.dashboard_data = {}
        self._update_dashboard_data()
    
    def _update_dashboard_data(self):
        """Update dashboard data from metrics tracker."""
        try:
            # Get recent metrics (last 24 hours)
            hours = 24
            
            # Model metrics
            model_metrics = {}
            for model_id in self.metrics_tracker.current_predictions.keys():
                recent_metrics = self.metrics_tracker.get_recent_metrics(model_id, hours)
                if recent_metrics:
                    model_metrics[model_id] = [m.to_dict() for m in recent_metrics]
            
            # Ensemble metrics
            ensemble_metrics = self.metrics_tracker.get_recent_ensemble_metrics(hours)
            ensemble_data = [m.to_dict() for m in ensemble_metrics]
            
            # System health
            system_health = self.metrics_tracker.get_recent_system_health(hours)
            health_data = [m.to_dict() for m in system_health]
            
            # Summary statistics
            summary = self._calculate_summary_stats(model_metrics, ensemble_data, health_data)
            
            self.dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'model_metrics': model_metrics,
                'ensemble_metrics': ensemble_data,
                'system_health': health_data,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def _calculate_summary_stats(self, model_metrics: Dict, ensemble_data: List, health_data: List) -> Dict:
        """Calculate summary statistics for dashboard."""
        summary = {
            'total_models': len(model_metrics),
            'total_predictions': self.metrics_tracker.prediction_count,
            'error_count': self.metrics_tracker.error_count,
            'uptime_hours': 0,
            'avg_latency': 0,
            'best_model': None,
            'worst_model': None,
            'ensemble_performance': {}
        }
        
        # System uptime and performance
        if health_data:
            latest_health = health_data[-1]
            summary['uptime_hours'] = latest_health['uptime']
            summary['avg_latency'] = latest_health['prediction_latency']
        
        # Model performance comparison
        if model_metrics:
            model_performance = {}
            
            for model_id, metrics_list in model_metrics.items():
                if metrics_list:
                    latest_metrics = metrics_list[-1]
                    model_performance[model_id] = latest_metrics['mae']
            
            if model_performance:
                best_model = min(model_performance.keys(), key=lambda k: model_performance[k])
                worst_model = max(model_performance.keys(), key=lambda k: model_performance[k])
                
                summary['best_model'] = {
                    'model_id': best_model,
                    'mae': model_performance[best_model]
                }
                summary['worst_model'] = {
                    'model_id': worst_model,
                    'mae': model_performance[worst_model]
                }
        
        # Ensemble performance
        if ensemble_data:
            latest_ensemble = ensemble_data[-1]
            summary['ensemble_performance'] = {
                'mae': latest_ensemble['ensemble_mae'],
                'accuracy': latest_ensemble['ensemble_accuracy'],
                'diversity': latest_ensemble['diversity_score']
            }
        
        return summary
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard."""
        self._update_dashboard_data()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Time Series Model Performance Dashboard</title>
            <meta http-equiv="refresh" content="{auto_refresh}">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                .card {{ background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 3px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
                .status-good {{ color: #27ae60; }}
                .status-warning {{ color: #f39c12; }}
                .status-error {{ color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Time Series Model Performance Dashboard</h1>
                <p>Last Updated: {timestamp}</p>
            </div>
            
            <div class="container">
                <!-- Summary Card -->
                <div class="card" style="flex: 1; min-width: 300px;">
                    <h2>System Summary</h2>
                    <div class="metric">
                        <div class="metric-value">{total_models}</div>
                        <div class="metric-label">Active Models</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{total_predictions}</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{uptime_hours:.1f}h</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{avg_latency:.1f}ms</div>
                        <div class="metric-label">Avg Latency</div>
                    </div>
                </div>
                
                <!-- Model Performance Card -->
                <div class="card" style="flex: 2; min-width: 400px;">
                    <h2>Model Performance</h2>
                    {model_performance_table}
                </div>
            </div>
            
            <div class="container">
                <!-- Ensemble Performance Card -->
                <div class="card" style="flex: 1; min-width: 300px;">
                    <h2>Ensemble Performance</h2>
                    {ensemble_performance_content}
                </div>
                
                <!-- System Health Card -->
                <div class="card" style="flex: 1; min-width: 300px;">
                    <h2>System Health</h2>
                    {system_health_content}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Generate model performance table
        model_table = self._generate_model_performance_table()
        
        # Generate ensemble performance content
        ensemble_content = self._generate_ensemble_performance_content()
        
        # Generate system health content
        health_content = self._generate_system_health_content()
        
        # Fill template
        summary = self.dashboard_data.get('summary', {})
        
        return html_template.format(
            auto_refresh=self.auto_refresh,
            timestamp=self.dashboard_data.get('timestamp', 'Unknown'),
            total_models=summary.get('total_models', 0),
            total_predictions=summary.get('total_predictions', 0),
            uptime_hours=summary.get('uptime_hours', 0),
            avg_latency=summary.get('avg_latency', 0),
            model_performance_table=model_table,
            ensemble_performance_content=ensemble_content,
            system_health_content=health_content
        )
    
    def _generate_model_performance_table(self) -> str:
        """Generate model performance table HTML."""
        model_metrics = self.dashboard_data.get('model_metrics', {})
        
        if not model_metrics:
            return "<p>No model data available</p>"
        
        table_html = """
        <table>
            <tr>
                <th>Model ID</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>Accuracy (%)</th>
                <th>Sharpe Ratio</th>
                <th>Predictions</th>
                <th>Status</th>
            </tr>
        """
        
        for model_id, metrics_list in model_metrics.items():
            if metrics_list:
                latest = metrics_list[-1]
                
                # Determine status based on performance
                mae = latest['mae']
                if mae < 0.02:
                    status_class = "status-good"
                    status_text = "Good"
                elif mae < 0.05:
                    status_class = "status-warning"
                    status_text = "Warning"
                else:
                    status_class = "status-error"
                    status_text = "Poor"
                
                table_html += f"""
                <tr>
                    <td>{model_id}</td>
                    <td>{latest['mae']:.4f}</td>
                    <td>{latest['rmse']:.4f}</td>
                    <td>{latest['directional_accuracy']:.1f}%</td>
                    <td>{latest['sharpe_ratio']:.2f}</td>
                    <td>{latest['prediction_count']}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
                """
        
        table_html += "</table>"
        return table_html
    
    def _generate_ensemble_performance_content(self) -> str:
        """Generate ensemble performance content HTML."""
        ensemble_metrics = self.dashboard_data.get('ensemble_metrics', [])
        
        if not ensemble_metrics:
            return "<p>No ensemble data available</p>"
        
        latest = ensemble_metrics[-1]
        
        return f"""
        <div class="metric">
            <div class="metric-value">{latest['ensemble_mae']:.4f}</div>
            <div class="metric-label">Ensemble MAE</div>
        </div>
        <div class="metric">
            <div class="metric-value">{latest['ensemble_accuracy']:.1f}%</div>
            <div class="metric-label">Ensemble Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{latest['diversity_score']:.2f}</div>
            <div class="metric-label">Model Diversity</div>
        </div>
        <div class="metric">
            <div class="metric-value">{latest['model_agreement']:.2f}</div>
            <div class="metric-label">Model Agreement</div>
        </div>
        <p><strong>Best Model:</strong> {latest['best_model_id']}</p>
        <p><strong>Worst Model:</strong> {latest['worst_model_id']}</p>
        """
    
    def _generate_system_health_content(self) -> str:
        """Generate system health content HTML."""
        health_data = self.dashboard_data.get('system_health', [])
        
        if not health_data:
            return "<p>No system health data available</p>"
        
        latest = health_data[-1]
        
        # Determine status colors
        cpu_class = "status-good" if latest['cpu_usage'] < 70 else "status-warning" if latest['cpu_usage'] < 90 else "status-error"
        memory_class = "status-good" if latest['memory_usage'] < 70 else "status-warning" if latest['memory_usage'] < 90 else "status-error"
        
        gpu_content = ""
        if latest['gpu_usage'] is not None:
            gpu_class = "status-good" if latest['gpu_usage'] < 70 else "status-warning" if latest['gpu_usage'] < 90 else "status-error"
            gpu_content = f"""
            <div class="metric">
                <div class="metric-value {gpu_class}">{latest['gpu_usage']:.1f}%</div>
                <div class="metric-label">GPU Usage</div>
            </div>
            """
        
        return f"""
        <div class="metric">
            <div class="metric-value {cpu_class}">{latest['cpu_usage']:.1f}%</div>
            <div class="metric-label">CPU Usage</div>
        </div>
        <div class="metric">
            <div class="metric-value {memory_class}">{latest['memory_usage']:.1f}%</div>
            <div class="metric-label">Memory Usage</div>
        </div>
        {gpu_content}
        <div class="metric">
            <div class="metric-value">{latest['throughput']:.1f}</div>
            <div class="metric-label">Predictions/sec</div>
        </div>
        <div class="metric">
            <div class="metric-value">{latest['error_rate']:.1f}%</div>
            <div class="metric-label">Error Rate</div>
        </div>
        """
    
    def save_dashboard(self, filepath: str):
        """Save dashboard HTML to file."""
        html_content = self.generate_html_dashboard()
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {filepath}")
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file."""
        self._update_dashboard_data()
        
        with open(filepath, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")


class PerformanceMonitor:
    """
    Main performance monitoring system integrating real-time tracking and dashboard.
    """
    
    def __init__(self, 
                 max_history: int = 10000,
                 update_interval: int = 60,
                 dashboard_port: int = 8050,
                 auto_refresh: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum historical records to keep
            update_interval: Metrics update interval in seconds
            dashboard_port: Port for web dashboard
            auto_refresh: Dashboard auto-refresh interval in seconds
        """
        self.metrics_tracker = RealTimeMetricsTracker(max_history, update_interval)
        self.dashboard = PerformanceDashboard(self.metrics_tracker, dashboard_port, auto_refresh)
        
        self.is_running = False
    
    def start(self):
        """Start performance monitoring."""
        if not self.is_running:
            self.metrics_tracker.start_monitoring()
            self.is_running = True
            logger.info("Performance monitor started")
    
    def stop(self):
        """Stop performance monitoring."""
        if self.is_running:
            self.metrics_tracker.stop_monitoring()
            self.is_running = False
            logger.info("Performance monitor stopped")
    
    def add_prediction(self, model_id: str, prediction: float, actual: Optional[float] = None, 
                      confidence: float = 1.0, timestamp: Optional[datetime] = None):
        """Add a prediction for monitoring."""
        self.metrics_tracker.add_prediction(model_id, prediction, actual, confidence, timestamp)
    
    def add_actual(self, model_id: str, actual: float, timestamp: Optional[datetime] = None):
        """Add actual value for a previous prediction."""
        self.metrics_tracker.add_actual(model_id, actual, timestamp)
    
    def record_latency(self, latency_ms: float):
        """Record prediction latency."""
        self.metrics_tracker.record_latency(latency_ms)
    
    def record_error(self):
        """Record an error occurrence."""
        self.metrics_tracker.record_error()
    
    def get_model_performance(self, model_id: str, hours: int = 24) -> List[Dict]:
        """Get recent performance metrics for a model."""
        metrics = self.metrics_tracker.get_recent_metrics(model_id, hours)
        return [m.to_dict() for m in metrics]
    
    def get_ensemble_performance(self, hours: int = 24) -> List[Dict]:
        """Get recent ensemble performance metrics."""
        metrics = self.metrics_tracker.get_recent_ensemble_metrics(hours)
        return [m.to_dict() for m in metrics]
    
    def get_system_health(self, hours: int = 24) -> List[Dict]:
        """Get recent system health metrics."""
        metrics = self.metrics_tracker.get_recent_system_health(hours)
        return [m.to_dict() for m in metrics]
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        return self.dashboard.generate_html_dashboard()
    
    def save_dashboard(self, filepath: str):
        """Save dashboard to HTML file."""
        self.dashboard.save_dashboard(filepath)
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file."""
        self.dashboard.export_metrics(filepath)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        self.dashboard._update_dashboard_data()
        return self.dashboard.dashboard_data.get('summary', {})