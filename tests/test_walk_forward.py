"""
Unit tests for walk-forward backtesting.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from enhanced_timeseries.backtesting.walk_forward import (
    WalkForwardBacktester, BacktestConfig, BacktestPeriod, BacktestResult,
    create_walk_forward_backtester
)
from enhanced_timeseries.core.interfaces import PerformanceMetrics


class MockPredictor:
    """Mock predictor for testing."""
    
    def __init__(self, prediction_value: float = 0.05):
        self.prediction_value = prediction_value
        self.is_trained = False
        
    def train(self, data, **kwargs):
        """Mock training."""
        self.is_trained = True
        return {'mae': 0.01}
    
    def predict(self, data):
        """Mock prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Return mock prediction result
        mock_result = Mock()
        mock_result.prediction = self.prediction_value
        return mock_result
    
    def get_predictor_info(self):
        """Mock predictor info."""
        return {'type': 'MockPredictor', 'is_trained': self.is_trained}


class TestBacktestConfig(unittest.TestCase):
    """Test backtest configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()
        
        self.assertEqual(config.initial_train_size, 252)
        self.assertEqual(config.retraining_frequency, 21)
        self.assertEqual(config.test_size, 21)
        self.assertEqual(config.min_train_size, 126)
        self.assertIsNone(config.max_train_size)
        self.assertTrue(config.expanding_window)
        self.assertEqual(config.purge_days, 0)
        self.assertEqual(config.embargo_days, 0)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_train_size=500,
            retraining_frequency=10,
            test_size=5,
            expanding_window=False,
            purge_days=2,
            embargo_days=1
        )
        
        self.assertEqual(config.initial_train_size, 500)
        self.assertEqual(config.retraining_frequency, 10)
        self.assertEqual(config.test_size, 5)
        self.assertFalse(config.expanding_window)
        self.assertEqual(config.purge_days, 2)
        self.assertEqual(config.embargo_days, 1)
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = BacktestConfig(initial_train_size=100, test_size=10)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['initial_train_size'], 100)
        self.assertEqual(config_dict['test_size'], 10)


class TestBacktestPeriod(unittest.TestCase):
    """Test backtest period."""
    
    def setUp(self):
        """Set up test period."""
        self.period = BacktestPeriod(
            period_id=1,
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-12-31'),
            test_start=pd.Timestamp('2021-01-01'),
            test_end=pd.Timestamp('2021-01-31'),
            train_size=252,
            test_size=21
        )
    
    def test_period_creation(self):
        """Test period creation."""
        self.assertEqual(self.period.period_id, 1)
        self.assertEqual(self.period.train_start, pd.Timestamp('2020-01-01'))
        self.assertEqual(self.period.test_end, pd.Timestamp('2021-01-31'))
        self.assertEqual(self.period.train_size, 252)
        self.assertEqual(self.period.test_size, 21)
    
    def test_period_to_dict(self):
        """Test period serialization."""
        period_dict = self.period.to_dict()
        
        expected_keys = [
            'period_id', 'train_start', 'train_end', 'test_start', 
            'test_end', 'train_size', 'test_size'
        ]
        
        for key in expected_keys:
            self.assertIn(key, period_dict)
        
        self.assertEqual(period_dict['period_id'], 1)
        self.assertEqual(period_dict['train_size'], 252)


class TestWalkForwardBacktester(unittest.TestCase):
    """Test walk-forward backtester."""
    
    def setUp(self):
        """Set up test data and backtester."""
        # Create sample time series data
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Generate synthetic data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.cumprod(1 + returns)
        
        self.data = pd.DataFrame({
            'price': prices,
            'target': returns,  # Use returns as target
            'feature1': np.random.randn(n_days),
            'feature2': np.random.randn(n_days)
        }, index=dates)
        
        # Create backtester with small config for testing
        self.config = BacktestConfig(
            initial_train_size=50,
            retraining_frequency=10,
            test_size=5,
            min_train_size=30
        )
        
        self.backtester = WalkForwardBacktester(self.config)
        
        # Mock predictor factory
        self.predictor_factory = lambda: MockPredictor(prediction_value=0.001)
    
    def test_backtester_creation(self):
        """Test backtester creation."""
        self.assertIsInstance(self.backtester, WalkForwardBacktester)
        self.assertEqual(self.backtester.config.initial_train_size, 50)
        self.assertEqual(len(self.backtester.results), 0)
    
    def test_generate_periods_expanding(self):
        """Test period generation with expanding window."""
        periods = self.backtester._generate_periods(self.data)
        
        self.assertGreater(len(periods), 0)
        
        # Check first period
        first_period = periods[0]
        self.assertEqual(first_period.period_id, 0)
        self.assertEqual(first_period.train_size, 50)  # Initial train size
        self.assertEqual(first_period.test_size, 5)
        
        # Check that training window expands
        if len(periods) > 1:
            second_period = periods[1]
            self.assertGreater(second_period.train_size, first_period.train_size)
    
    def test_generate_periods_rolling(self):
        """Test period generation with rolling window."""
        config = BacktestConfig(
            initial_train_size=50,
            retraining_frequency=10,
            test_size=5,
            expanding_window=False,
            max_train_size=60
        )
        
        backtester = WalkForwardBacktester(config)
        periods = backtester._generate_periods(self.data)
        
        self.assertGreater(len(periods), 0)
        
        # Check that training window doesn't exceed max size
        for period in periods:
            self.assertLessEqual(period.train_size, 60)
    
    def test_generate_periods_with_purge_embargo(self):
        """Test period generation with purge and embargo."""
        config = BacktestConfig(
            initial_train_size=50,
            retraining_frequency=10,
            test_size=5,
            purge_days=2,
            embargo_days=1
        )
        
        backtester = WalkForwardBacktester(config)
        periods = backtester._generate_periods(self.data)
        
        self.assertGreater(len(periods), 0)
        
        # Check that periods are properly spaced
        for period in periods:
            self.assertIsInstance(period, BacktestPeriod)
    
    def test_run_single_period(self):
        """Test running a single backtest period."""
        periods = self.backtester._generate_periods(self.data)
        
        if len(periods) > 0:
            period = periods[0]
            result = self.backtester._run_single_period(
                self.data, period, self.predictor_factory, 'target'
            )
            
            self.assertIsInstance(result, BacktestResult)
            self.assertEqual(result.period, period)
            self.assertGreater(len(result.predictions), 0)
            self.assertGreater(len(result.actuals), 0)
            self.assertEqual(len(result.predictions), len(result.actuals))
            self.assertIsInstance(result.metrics, PerformanceMetrics)
            self.assertGreater(result.training_time, 0)
            self.assertGreaterEqual(result.prediction_time, 0)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        predictions = [0.01, -0.005, 0.02, -0.01, 0.015]
        actuals = [0.012, -0.008, 0.018, -0.015, 0.01]
        
        metrics = self.backtester.calculate_performance_metrics(predictions, actuals)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.mae, 0)
        self.assertGreater(metrics.rmse, 0)
        self.assertTrue(0 <= metrics.directional_accuracy <= 100)
        self.assertTrue(np.isfinite(metrics.sharpe_ratio))
    
    def test_calculate_performance_metrics_empty(self):
        """Test performance metrics with empty data."""
        metrics = self.backtester.calculate_performance_metrics([], [])
        
        self.assertEqual(metrics.mae, float('inf'))
        self.assertEqual(metrics.rmse, float('inf'))
        self.assertEqual(metrics.directional_accuracy, 0.0)
    
    def test_run_backtest(self):
        """Test full backtest run."""
        # Use smaller dataset for faster testing
        small_data = self.data.head(100)
        
        results = self.backtester.run_backtest(
            small_data, 
            self.predictor_factory, 
            target_column='target'
        )
        
        self.assertIsInstance(results, dict)
        
        expected_keys = ['config', 'periods', 'summary_metrics', 'n_periods']
        for key in expected_keys:
            self.assertIn(key, results)
        
        self.assertGreater(results['n_periods'], 0)
        self.assertIsInstance(results['summary_metrics'], dict)
    
    def test_run_backtest_invalid_data(self):
        """Test backtest with invalid data."""
        # Data without datetime index
        invalid_data = pd.DataFrame({'target': [1, 2, 3]})
        
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(invalid_data, self.predictor_factory)
        
        # Data without target column
        data_no_target = self.data.drop('target', axis=1)
        
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(data_no_target, self.predictor_factory, 'missing_target')
    
    def test_get_results_dataframe(self):
        """Test results DataFrame generation."""
        # Run a small backtest first
        small_data = self.data.head(80)
        self.backtester.run_backtest(small_data, self.predictor_factory, 'target')
        
        df = self.backtester.get_results_dataframe()
        
        if not df.empty:
            expected_columns = [
                'period_id', 'timestamp', 'prediction', 'actual', 'error',
                'squared_error', 'directional_correct', 'train_start', 'train_end', 'train_size'
            ]
            
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            self.assertGreater(len(df), 0)
    
    def test_summary_metrics_calculation(self):
        """Test summary metrics calculation."""
        # Create mock results
        mock_metrics = PerformanceMetrics(
            mae=0.01, rmse=0.015, directional_accuracy=65.0,
            sharpe_ratio=1.2, max_drawdown=0.05, hit_rate=0.65,
            profit_factor=1.5, regime_performance={}
        )
        
        mock_period = BacktestPeriod(
            period_id=0, train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-12-31'), test_start=pd.Timestamp('2021-01-01'),
            test_end=pd.Timestamp('2021-01-31'), train_size=252, test_size=21
        )
        
        mock_result = BacktestResult(
            period=mock_period, predictions=[0.01, 0.02], actuals=[0.012, 0.018],
            timestamps=[pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-02')],
            metrics=mock_metrics, training_time=1.0, prediction_time=0.1,
            model_info={'type': 'mock'}
        )
        
        self.backtester.results = [mock_result]
        summary = self.backtester._calculate_summary_metrics()
        
        self.assertIsInstance(summary, dict)
        
        expected_keys = [
            'overall_mae', 'overall_directional_accuracy', 'n_periods',
            'win_rate', 'total_training_time', 'total_predictions'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
    
    def test_export_results(self):
        """Test results export."""
        import tempfile
        import os
        
        # Run a small backtest
        small_data = self.data.head(80)
        self.backtester.run_backtest(small_data, self.predictor_factory, 'target')
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            try:
                self.backtester.export_results(f.name, format='csv')
                self.assertTrue(os.path.exists(f.name))
                self.assertGreater(os.path.getsize(f.name), 0)
            finally:
                os.unlink(f.name)
    
    def test_create_walk_forward_backtester(self):
        """Test factory function."""
        config_dict = {
            'initial_train_size': 100,
            'retraining_frequency': 20,
            'test_size': 10
        }
        
        backtester = create_walk_forward_backtester(config_dict)
        
        self.assertIsInstance(backtester, WalkForwardBacktester)
        self.assertEqual(backtester.config.initial_train_size, 100)
        self.assertEqual(backtester.config.retraining_frequency, 20)
        self.assertEqual(backtester.config.test_size, 10)


class TestBacktestResult(unittest.TestCase):
    """Test backtest result."""
    
    def setUp(self):
        """Set up test result."""
        self.period = BacktestPeriod(
            period_id=1,
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-12-31'),
            test_start=pd.Timestamp('2021-01-01'),
            test_end=pd.Timestamp('2021-01-31'),
            train_size=252,
            test_size=21
        )
        
        self.metrics = PerformanceMetrics(
            mae=0.01, rmse=0.015, directional_accuracy=65.0,
            sharpe_ratio=1.2, max_drawdown=0.05, hit_rate=0.65,
            profit_factor=1.5, regime_performance={}
        )
        
        self.result = BacktestResult(
            period=self.period,
            predictions=[0.01, 0.02, -0.005],
            actuals=[0.012, 0.018, -0.008],
            timestamps=[
                pd.Timestamp('2021-01-01'),
                pd.Timestamp('2021-01-02'),
                pd.Timestamp('2021-01-03')
            ],
            metrics=self.metrics,
            training_time=2.5,
            prediction_time=0.3,
            model_info={'type': 'TestModel'}
        )
    
    def test_result_creation(self):
        """Test result creation."""
        self.assertEqual(self.result.period, self.period)
        self.assertEqual(len(self.result.predictions), 3)
        self.assertEqual(len(self.result.actuals), 3)
        self.assertEqual(len(self.result.timestamps), 3)
        self.assertEqual(self.result.metrics, self.metrics)
        self.assertEqual(self.result.training_time, 2.5)
        self.assertEqual(self.result.prediction_time, 0.3)
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result_dict = self.result.to_dict()
        
        expected_keys = [
            'period', 'predictions', 'actuals', 'timestamps',
            'metrics', 'training_time', 'prediction_time', 'model_info'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result_dict)
        
        self.assertEqual(len(result_dict['predictions']), 3)
        self.assertEqual(len(result_dict['timestamps']), 3)


if __name__ == '__main__':
    unittest.main()