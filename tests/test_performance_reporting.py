"""
Unit tests for performance reporting and statistical analysis.
"""

import unittest
import numpy as np
import pandas as pd
from enhanced_timeseries.backtesting.performance_reporting import (
    PerformanceAnalyzer, ReportGenerator, StatisticalTest, PerformanceReport
)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analyzer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample backtest results
        n_predictions = 100
        dates = pd.date_range('2020-01-01', periods=n_predictions, freq='D')
        
        # Generate realistic predictions and actuals
        true_returns = np.random.normal(0.001, 0.02, n_predictions)
        predictions = true_returns + np.random.normal(0, 0.005, n_predictions)
        
        self.results_df = pd.DataFrame({
            'timestamp': dates,
            'prediction': predictions,
            'actual': true_returns,
            'error': np.abs(predictions - true_returns),
            'squared_error': (predictions - true_returns) ** 2,
            'directional_correct': np.sign(predictions) == np.sign(true_returns),
            'period_id': np.repeat(range(10), 10)  # 10 periods with 10 predictions each
        })
        
        # Create benchmark returns
        self.benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, n_predictions),
            index=dates
        )
        
        self.analyzer = PerformanceAnalyzer()
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        self.assertEqual(self.analyzer.significance_level, 0.05)
    
    def test_calculate_summary_metrics(self):
        """Test summary metrics calculation."""
        metrics = self.analyzer._calculate_summary_metrics(self.results_df)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'mae', 'rmse', 'mape', 'directional_accuracy', 'correlation',
            'mean_return', 'annualized_return', 'volatility', 'annualized_volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown',
            'win_rate', 'loss_rate', 'profit_factor', 'skewness', 'kurtosis',
            'var_95', 'var_99', 'expected_shortfall_95', 'expected_shortfall_99',
            'n_predictions', 'n_periods'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertTrue(np.isfinite(metrics[metric]))
        
        # Check reasonable ranges
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertTrue(0 <= metrics['directional_accuracy'] <= 100)
        self.assertTrue(-1 <= metrics['correlation'] <= 1)
        self.assertTrue(0 <= metrics['win_rate'] <= 1)
        self.assertEqual(metrics['n_predictions'], 100)
        self.assertEqual(metrics['n_periods'], 10)
    
    def test_calculate_summary_metrics_empty(self):
        """Test summary metrics with empty data."""
        empty_df = pd.DataFrame()
        metrics = self.analyzer._calculate_summary_metrics(empty_df)
        
        self.assertEqual(metrics, {})
    
    def test_perform_statistical_tests(self):
        """Test statistical tests."""
        tests = self.analyzer._perform_statistical_tests(self.results_df)
        
        self.assertIsInstance(tests, list)
        self.assertGreater(len(tests), 0)
        
        # Check test structure
        for test in tests:
            self.assertIsInstance(test, StatisticalTest)
            self.assertIsInstance(test.test_name, str)
            self.assertIsInstance(test.statistic, (int, float))
            self.assertIsInstance(test.p_value, (int, float))
            self.assertIsInstance(test.is_significant, bool)
            self.assertIsInstance(test.interpretation, str)
            
            # Check p-value range
            self.assertTrue(0 <= test.p_value <= 1)
    
    def test_perform_statistical_tests_empty(self):
        """Test statistical tests with empty data."""
        empty_df = pd.DataFrame()
        tests = self.analyzer._perform_statistical_tests(empty_df)
        
        self.assertEqual(tests, [])
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        risk_metrics = self.analyzer._calculate_risk_metrics(self.results_df)
        
        expected_metrics = [
            'downside_deviation', 'var_95', 'var_99', 'cvar_95', 'cvar_99',
            'max_consecutive_losses', 'ulcer_index', 'pain_index',
            'sterling_ratio', 'burke_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)
            self.assertIsInstance(risk_metrics[metric], (int, float))
            self.assertTrue(np.isfinite(risk_metrics[metric]))
        
        # Check reasonable ranges
        self.assertGreaterEqual(risk_metrics['downside_deviation'], 0)
        self.assertGreaterEqual(risk_metrics['max_consecutive_losses'], 0)
        self.assertGreaterEqual(risk_metrics['ulcer_index'], 0)
        self.assertGreaterEqual(risk_metrics['pain_index'], 0)
    
    def test_calculate_consistency_metrics(self):
        """Test consistency metrics calculation."""
        consistency_metrics = self.analyzer._calculate_consistency_metrics(self.results_df)
        
        expected_metrics = [
            'mae_consistency', 'accuracy_consistency', 'mae_trend_slope',
            'mae_trend_r_squared', 'accuracy_trend_slope', 'accuracy_trend_r_squared',
            'rolling_mae_stability', 'rolling_accuracy_stability',
            'periods_above_median_mae', 'periods_above_median_accuracy', 'n_periods'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, consistency_metrics)
            self.assertIsInstance(consistency_metrics[metric], (int, float))
            self.assertTrue(np.isfinite(consistency_metrics[metric]))
        
        # Check reasonable ranges
        self.assertTrue(0 <= consistency_metrics['mae_consistency'] <= 1)
        self.assertTrue(0 <= consistency_metrics['accuracy_consistency'] <= 1)
        self.assertTrue(0 <= consistency_metrics['periods_above_median_mae'] <= 100)
        self.assertTrue(0 <= consistency_metrics['periods_above_median_accuracy'] <= 100)
        self.assertEqual(consistency_metrics['n_periods'], 10)
    
    def test_compare_to_benchmark(self):
        """Test benchmark comparison."""
        comparison = self.analyzer._compare_to_benchmark(self.results_df, self.benchmark_returns)
        
        expected_metrics = [
            'strategy_return', 'benchmark_return', 'excess_return',
            'strategy_volatility', 'benchmark_volatility', 'strategy_sharpe',
            'benchmark_sharpe', 'tracking_error', 'information_ratio',
            'beta', 'alpha', 'correlation', 'up_capture_ratio', 'down_capture_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, comparison)
            self.assertIsInstance(comparison[metric], (int, float))
            self.assertTrue(np.isfinite(comparison[metric]))
        
        # Check reasonable ranges
        self.assertTrue(-1 <= comparison['correlation'] <= 1)
        self.assertGreaterEqual(comparison['strategy_volatility'], 0)
        self.assertGreaterEqual(comparison['benchmark_volatility'], 0)
        self.assertGreaterEqual(comparison['tracking_error'], 0)
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Create metrics that should trigger recommendations
        summary_metrics = {
            'mae': 0.03,  # High error
            'directional_accuracy': 45,  # Low accuracy
            'sharpe_ratio': 0.5,  # Low Sharpe
            'max_drawdown': 0.25  # High drawdown
        }
        
        risk_metrics = {}
        consistency_metrics = {'mae_consistency': 0.5}  # Low consistency
        
        # Create statistical tests
        statistical_tests = [
            StatisticalTest(
                test_name="One-Sample t-test (Zero Mean Errors)",
                statistic=2.5,
                p_value=0.01,
                critical_value=None,
                is_significant=True,
                interpretation="Errors have significant bias"
            )
        ]
        
        recommendations = self.analyzer._generate_recommendations(
            summary_metrics, risk_metrics, consistency_metrics, statistical_tests
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check that recommendations are strings
        for rec in recommendations:
            self.assertIsInstance(rec, str)
            self.assertGreater(len(rec), 0)
    
    def test_analyze_backtest_results(self):
        """Test complete backtest analysis."""
        report = self.analyzer.analyze_backtest_results(
            self.results_df,
            benchmark_returns=self.benchmark_returns
        )
        
        self.assertIsInstance(report, PerformanceReport)
        
        # Check report structure
        self.assertIsInstance(report.summary_metrics, dict)
        self.assertIsInstance(report.statistical_tests, list)
        self.assertIsInstance(report.risk_metrics, dict)
        self.assertIsInstance(report.consistency_metrics, dict)
        self.assertIsInstance(report.benchmark_comparison, dict)
        self.assertIsInstance(report.recommendations, list)
        
        # Check that report can be serialized
        report_dict = report.to_dict()
        self.assertIsInstance(report_dict, dict)
    
    def test_runs_test(self):
        """Test runs test implementation."""
        # Test with alternating pattern (should be random)
        alternating = np.array([True, False, True, False, True, False])
        runs, n1, n2 = self.analyzer._runs_test(alternating)
        
        self.assertEqual(runs, 6)  # Each element is a run
        self.assertEqual(n1, 3)  # Three True values
        self.assertEqual(n2, 3)  # Three False values
        
        # Test with clustered pattern (should be non-random)
        clustered = np.array([True, True, True, False, False, False])
        runs, n1, n2 = self.analyzer._runs_test(clustered)
        
        self.assertEqual(runs, 2)  # Two runs (TTT and FFF)
        self.assertEqual(n1, 3)
        self.assertEqual(n2, 3)


class TestReportGenerator(unittest.TestCase):
    """Test report generator."""
    
    def setUp(self):
        """Set up test data."""
        self.generator = ReportGenerator()
        
        # Create sample performance report
        self.performance_report = PerformanceReport(
            summary_metrics={
                'mae': 0.015,
                'directional_accuracy': 65.5,
                'sharpe_ratio': 1.2
            },
            regime_analysis={},
            statistical_tests=[
                StatisticalTest(
                    test_name="Test Example",
                    statistic=1.5,
                    p_value=0.13,
                    critical_value=1.96,
                    is_significant=False,
                    interpretation="Test passed"
                )
            ],
            risk_metrics={'var_95': -0.02},
            consistency_metrics={'mae_consistency': 0.8},
            benchmark_comparison={'excess_return': 0.05},
            recommendations=['Good performance overall']
        )
    
    def test_generator_creation(self):
        """Test generator creation."""
        self.assertIsInstance(self.generator.analyzer, PerformanceAnalyzer)
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        html_report = self.generator.generate_html_report(self.performance_report)
        
        self.assertIsInstance(html_report, str)
        self.assertIn('<html>', html_report)
        self.assertIn('</html>', html_report)
        self.assertIn('Summary Metrics', html_report)
        self.assertIn('Statistical Tests', html_report)
        self.assertIn('Recommendations', html_report)
        
        # Check that metrics are included
        self.assertIn('0.015', html_report)  # MAE value
        self.assertIn('65.5', html_report)   # Directional accuracy
    
    def test_generate_text_report(self):
        """Test text report generation."""
        text_report = self.generator.generate_text_report(self.performance_report)
        
        self.assertIsInstance(text_report, str)
        self.assertIn('SUMMARY METRICS', text_report)
        self.assertIn('STATISTICAL TESTS', text_report)
        self.assertIn('RECOMMENDATIONS', text_report)
        
        # Check that metrics are included
        self.assertIn('0.015', text_report)  # MAE value
        self.assertIn('65.5', text_report)   # Directional accuracy
        self.assertIn('Good performance overall', text_report)  # Recommendation


class TestStatisticalTest(unittest.TestCase):
    """Test statistical test dataclass."""
    
    def test_statistical_test_creation(self):
        """Test statistical test creation."""
        test = StatisticalTest(
            test_name="Example Test",
            statistic=2.5,
            p_value=0.01,
            critical_value=1.96,
            is_significant=True,
            interpretation="Significant result"
        )
        
        self.assertEqual(test.test_name, "Example Test")
        self.assertEqual(test.statistic, 2.5)
        self.assertEqual(test.p_value, 0.01)
        self.assertEqual(test.critical_value, 1.96)
        self.assertTrue(test.is_significant)
        self.assertEqual(test.interpretation, "Significant result")
    
    def test_statistical_test_to_dict(self):
        """Test statistical test serialization."""
        test = StatisticalTest(
            test_name="Example Test",
            statistic=2.5,
            p_value=0.01,
            critical_value=1.96,
            is_significant=True,
            interpretation="Significant result"
        )
        
        test_dict = test.to_dict()
        
        expected_keys = [
            'test_name', 'statistic', 'p_value', 'critical_value',
            'is_significant', 'interpretation'
        ]
        
        for key in expected_keys:
            self.assertIn(key, test_dict)
        
        self.assertEqual(test_dict['test_name'], "Example Test")
        self.assertEqual(test_dict['statistic'], 2.5)


if __name__ == '__main__':
    unittest.main()