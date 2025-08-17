"""
Unit tests for performance monitoring system.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from enhanced_timeseries.monitoring.performance_monitor import (
    ModelPerformanceMetrics, EnsemblePerformanceMetrics, SystemHealthMetrics,
    MetricsCalculator, RealTimeMetricsTracker, PerformanceDashboard, PerformanceMonitor
)


class TestModelPerformanceMetrics(unittest.TestCase):
    """Test ModelPerformanceMetrics dataclass."""
    
    def test_metrics_creation_and_serialization(self):
        """Test metrics creation and serialization."""
        metrics = ModelPerformanceMetrics(
            model_id='test_model',
            timestamp=datetime.now(),
            mae=0.05,
            rmse=0.07,
            mape=5.2,
            directional_accuracy=65.5,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            hit_rate=70.0,
            profit_factor=1.8,
            prediction_count=100,
            confidence_score=0.85
        )
        
        self.assertEqual(metrics.model_id, 'test_model')
        self.assertEqual(metrics.mae, 0.05)
        self.assertEqual(metrics.directional_accuracy, 65.5)
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['model_id'], 'test_model')
        self.assertEqual(metrics_dict['mae'], 0.05)
        self.assertIn('timestamp', metrics_dict)


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.calculator = MetricsCalculator()
        
        # Create test predictions and actuals
        self.predictions = np.array([1.0, 1.5, 0.8, 2.0, 1.2])
        self.actuals = np.array([1.1, 1.4, 0.9, 1.8, 1.3])
        self.returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    
    def test_calculate_mae(self):
        """Test MAE calculation."""
        mae = self.calculator.calculate_mae(self.predictions, self.actuals)
        
        expected_mae = np.mean(np.abs(self.predictions - self.actuals))
        self.assertAlmostEqual(mae, expected_mae, places=6)
        self.assertGreaterEqual(mae, 0)
    
    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        rmse = self.calculator.calculate_rmse(self.predictions, self.actuals)
        
        expected_rmse = np.sqrt(np.mean((self.predictions - self.actuals) ** 2))
        self.assertAlmostEqual(rmse, expected_rmse, places=6)
        self.assertGreaterEqual(rmse, 0)
    
    def test_calculate_mape(self):
        """Test MAPE calculation."""
        mape = self.calculator.calculate_mape(self.predictions, self.actuals)
        
        self.assertGreaterEqual(mape, 0)
        self.assertIsInstance(mape, float)
    
    def test_calculate_mape_with_zeros(self):
        """Test MAPE calculation with zero actuals."""
        actuals_with_zeros = np.array([0, 1, 2, 0, 3])
        predictions = np.array([0.1, 1.1, 1.9, 0.2, 2.8])
        
        mape = self.calculator.calculate_mape(predictions, actuals_with_zeros)
        
        # Should handle zeros gracefully
        self.assertGreaterEqual(mape, 0)
        self.assertTrue(np.isfinite(mape))
    
    def test_calculate_directional_accuracy(self):
        """Test directional accuracy calculation."""
        accuracy = self.calculator.calculate_directional_accuracy(self.predictions, self.actuals)
        
        self.assertTrue(0 <= accuracy <= 100)
        self.assertIsInstance(accuracy, float)
    
    def test_calculate_directional_accuracy_single_value(self):
        """Test directional accuracy with single value."""
        single_pred = np.array([1.0])
        single_actual = np.array([1.1])
        
        accuracy = self.calculator.calculate_directional_accuracy(single_pred, single_actual)
        self.assertEqual(accuracy, 0.0)  # Can't calculate direction with single value
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.calculator.calculate_sharpe_ratio(self.returns)
        
        self.assertIsInstance(sharpe, float)
        self.assertTrue(np.isfinite(sharpe))
    
    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        constant_returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = self.calculator.calculate_sharpe_ratio(constant_returns)
        self.assertEqual(sharpe, 0.0)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        drawdown = self.calculator.calculate_max_drawdown(self.returns)
        
        self.assertGreaterEqual(drawdown, 0)
        self.assertLessEqual(drawdown, 1)
        self.assertIsInstance(drawdown, float)
    
    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        hit_rate = self.calculator.calculate_hit_rate(self.predictions, self.actuals, threshold=0.2)
        
        self.assertTrue(0 <= hit_rate <= 100)
        self.assertIsInstance(hit_rate, float)
    
    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        profit_factor = self.calculator.calculate_profit_factor(self.returns)
        
        self.assertGreaterEqual(profit_factor, 0)
        self.assertIsInstance(profit_factor, float)
    
    def test_calculate_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        positive_returns = np.array([0.01, 0.02, 0.03])
        
        profit_factor = self.calculator.calculate_profit_factor(positive_returns)
        self.assertEqual(profit_factor, float('inf'))
    
    def test_calculate_profit_factor_no_gains(self):
        """Test profit factor with no gains."""
        negative_returns = np.array([-0.01, -0.02, -0.03])
        
        profit_factor = self.calculator.calculate_profit_factor(negative_returns)
        self.assertEqual(profit_factor, 0.0)


class TestRealTimeMetricsTracker(unittest.TestCase):
    """Test RealTimeMetricsTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.tracker = RealTimeMetricsTracker(max_history=100, update_interval=1)
    
    def tearDown(self):
        """Clean up test environment."""
        self.tracker.stop_monitoring()
    
    def test_tracker_creation(self):
        """Test tracker creation."""
        self.assertEqual(self.tracker.max_history, 100)
        self.assertEqual(self.tracker.update_interval, 1)
        self.assertEqual(self.tracker.prediction_count, 0)
        self.assertEqual(self.tracker.error_count, 0)
    
    def test_add_prediction(self):
        """Test adding predictions."""
        self.tracker.add_prediction('model1', 1.5, 1.4, confidence=0.8)
        
        self.assertEqual(len(self.tracker.current_predictions['model1']), 1)
        self.assertEqual(len(self.tracker.current_actuals['model1']), 1)
        self.assertEqual(self.tracker.prediction_count, 1)
    
    def test_add_prediction_without_actual(self):
        """Test adding prediction without actual value."""
        self.tracker.add_prediction('model1', 1.5, confidence=0.8)
        
        self.assertEqual(len(self.tracker.current_predictions['model1']), 1)
        self.assertEqual(len(self.tracker.current_actuals['model1']), 0)
    
    def test_add_actual(self):
        """Test adding actual values."""
        self.tracker.add_prediction('model1', 1.5)
        self.tracker.add_actual('model1', 1.4)
        
        self.assertEqual(len(self.tracker.current_actuals['model1']), 1)
    
    def test_record_latency(self):
        """Test recording latency."""
        self.tracker.record_latency(50.0)
        self.tracker.record_latency(75.0)
        
        self.assertEqual(len(self.tracker.latency_measurements), 2)
    
    def test_record_error(self):
        """Test recording errors."""
        initial_count = self.tracker.error_count
        self.tracker.record_error()
        
        self.assertEqual(self.tracker.error_count, initial_count + 1)
    
    def test_calculate_model_metrics(self):
        """Test model metrics calculation."""
        # Add some predictions and actuals
        predictions = [1.0, 1.5, 0.8, 2.0, 1.2]
        actuals = [1.1, 1.4, 0.9, 1.8, 1.3]
        
        for pred, actual in zip(predictions, actuals):
            self.tracker.add_prediction('model1', pred, actual)
        
        metrics = self.tracker.calculate_model_metrics('model1')
        
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, ModelPerformanceMetrics)
        self.assertEqual(metrics.model_id, 'model1')
        self.assertGreater(metrics.mae, 0)
        self.assertGreater(metrics.rmse, 0)
        self.assertEqual(metrics.prediction_count, 5)
    
    def test_calculate_model_metrics_insufficient_data(self):
        """Test model metrics calculation with insufficient data."""
        metrics = self.tracker.calculate_model_metrics('nonexistent_model')
        
        self.assertIsNone(metrics)
    
    def test_calculate_ensemble_metrics(self):
        """Test ensemble metrics calculation."""
        # Add predictions for multiple models
        for model_id in ['model1', 'model2', 'model3']:
            for i in range(5):
                pred = 1.0 + np.random.normal(0, 0.1)
                actual = pred + np.random.normal(0, 0.05)
                self.tracker.add_prediction(model_id, pred, actual)
        
        ensemble_metrics = self.tracker.calculate_ensemble_metrics()
        
        self.assertIsNotNone(ensemble_metrics)
        self.assertIsInstance(ensemble_metrics, EnsemblePerformanceMetrics)
        self.assertGreater(ensemble_metrics.ensemble_mae, 0)
        self.assertTrue(0 <= ensemble_metrics.model_agreement <= 1)
        self.assertTrue(0 <= ensemble_metrics.diversity_score <= 1)
    
    def test_calculate_ensemble_metrics_insufficient_models(self):
        """Test ensemble metrics with insufficient models."""
        # Add predictions for only one model
        self.tracker.add_prediction('model1', 1.0, 1.1)
        
        ensemble_metrics = self.tracker.calculate_ensemble_metrics()
        
        self.assertIsNone(ensemble_metrics)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_calculate_system_health(self, mock_memory, mock_cpu):
        """Test system health calculation."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value.percent = 60.0
        
        # Add some latency measurements
        self.tracker.record_latency(50.0)
        self.tracker.record_latency(75.0)
        
        health = self.tracker.calculate_system_health()
        
        self.assertIsInstance(health, SystemHealthMetrics)
        self.assertEqual(health.cpu_usage, 45.0)
        self.assertEqual(health.memory_usage, 60.0)
        self.assertGreaterEqual(health.prediction_latency, 0)
        self.assertGreaterEqual(health.uptime, 0)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        self.tracker.start_monitoring()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        self.assertIsNotNone(self.tracker._monitoring_thread)
        self.assertTrue(self.tracker._monitoring_thread.is_alive())
        
        self.tracker.stop_monitoring()
        
        # Give it a moment to stop
        time.sleep(0.1)
        
        self.assertTrue(self.tracker._stop_monitoring.is_set())
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        # Add some predictions
        for i in range(3):
            self.tracker.add_prediction('model1', 1.0 + i * 0.1, 1.0 + i * 0.1)
        
        # Calculate metrics (this would normally be done by monitoring thread)
        metrics = self.tracker.calculate_model_metrics('model1')
        if metrics:
            self.tracker.model_metrics_history['model1'].append(metrics)
        
        recent_metrics = self.tracker.get_recent_metrics('model1', hours=1)
        
        self.assertIsInstance(recent_metrics, list)
        if recent_metrics:
            self.assertIsInstance(recent_metrics[0], ModelPerformanceMetrics)


class TestPerformanceDashboard(unittest.TestCase):
    """Test PerformanceDashboard class."""
    
    def setUp(self):
        """Set up test environment."""
        self.tracker = RealTimeMetricsTracker(max_history=100, update_interval=1)
        self.dashboard = PerformanceDashboard(self.tracker, dashboard_port=8051)
        
        # Add some test data
        for model_id in ['model1', 'model2']:
            for i in range(5):
                pred = 1.0 + np.random.normal(0, 0.1)
                actual = pred + np.random.normal(0, 0.05)
                self.tracker.add_prediction(model_id, pred, actual)
    
    def tearDown(self):
        """Clean up test environment."""
        self.tracker.stop_monitoring()
    
    def test_dashboard_creation(self):
        """Test dashboard creation."""
        self.assertEqual(self.dashboard.dashboard_port, 8051)
        self.assertEqual(self.dashboard.auto_refresh, 30)
        self.assertIsInstance(self.dashboard.dashboard_data, dict)
    
    def test_update_dashboard_data(self):
        """Test dashboard data update."""
        self.dashboard._update_dashboard_data()
        
        self.assertIn('timestamp', self.dashboard.dashboard_data)
        self.assertIn('model_metrics', self.dashboard.dashboard_data)
        self.assertIn('summary', self.dashboard.dashboard_data)
    
    def test_generate_html_dashboard(self):
        """Test HTML dashboard generation."""
        html = self.dashboard.generate_html_dashboard()
        
        self.assertIsInstance(html, str)
        self.assertIn('<html>', html)
        self.assertIn('Performance Dashboard', html)
        self.assertIn('</html>', html)
    
    def test_save_dashboard(self):
        """Test saving dashboard to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.dashboard.save_dashboard(temp_filepath)
            
            # Check file exists and contains HTML
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                content = f.read()
            
            self.assertIn('<html>', content)
            self.assertIn('Performance Dashboard', content)
        
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_export_metrics(self):
        """Test exporting metrics to JSON."""
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.dashboard.export_metrics(temp_filepath)
            
            # Check file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, dict)
            self.assertIn('timestamp', data)
        
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor main class."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = PerformanceMonitor(
            max_history=100,
            update_interval=1,
            dashboard_port=8052
        )
    
    def tearDown(self):
        """Clean up test environment."""
        self.monitor.stop()
    
    def test_monitor_creation(self):
        """Test monitor creation."""
        self.assertIsNotNone(self.monitor.metrics_tracker)
        self.assertIsNotNone(self.monitor.dashboard)
        self.assertFalse(self.monitor.is_running)
    
    def test_start_stop_monitor(self):
        """Test starting and stopping monitor."""
        self.monitor.start()
        self.assertTrue(self.monitor.is_running)
        
        self.monitor.stop()
        self.assertFalse(self.monitor.is_running)
    
    def test_add_prediction(self):
        """Test adding predictions through monitor."""
        self.monitor.add_prediction('model1', 1.5, 1.4, confidence=0.8)
        
        # Check that prediction was added to tracker
        self.assertEqual(len(self.monitor.metrics_tracker.current_predictions['model1']), 1)
    
    def test_add_actual(self):
        """Test adding actual values through monitor."""
        self.monitor.add_prediction('model1', 1.5)
        self.monitor.add_actual('model1', 1.4)
        
        # Check that actual was added to tracker
        self.assertEqual(len(self.monitor.metrics_tracker.current_actuals['model1']), 1)
    
    def test_record_latency(self):
        """Test recording latency through monitor."""
        self.monitor.record_latency(50.0)
        
        # Check that latency was recorded
        self.assertEqual(len(self.monitor.metrics_tracker.latency_measurements), 1)
    
    def test_record_error(self):
        """Test recording errors through monitor."""
        initial_count = self.monitor.metrics_tracker.error_count
        self.monitor.record_error()
        
        self.assertEqual(self.monitor.metrics_tracker.error_count, initial_count + 1)
    
    def test_get_model_performance(self):
        """Test getting model performance."""
        # Add some predictions
        for i in range(3):
            self.monitor.add_prediction('model1', 1.0 + i * 0.1, 1.0 + i * 0.1)
        
        performance = self.monitor.get_model_performance('model1')
        
        self.assertIsInstance(performance, list)
    
    def test_generate_dashboard_html(self):
        """Test generating dashboard HTML."""
        html = self.monitor.generate_dashboard_html()
        
        self.assertIsInstance(html, str)
        self.assertIn('<html>', html)
    
    def test_save_dashboard(self):
        """Test saving dashboard."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.monitor.save_dashboard(temp_filepath)
            self.assertTrue(os.path.exists(temp_filepath))
        
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_export_metrics(self):
        """Test exporting metrics."""
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.monitor.export_metrics(temp_filepath)
            
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, dict)
        
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        summary = self.monitor.get_performance_summary()
        
        self.assertIsInstance(summary, dict)


if __name__ == '__main__':
    unittest.main()