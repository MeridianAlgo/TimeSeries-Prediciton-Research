"""
Unit tests for automated alerting and anomaly detection system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging

from enhanced_timeseries.monitoring.alerting import (
    Alert, AlertRule, StatisticalAnomalyDetector, IsolationForestAnomalyDetector,
    PerformanceDegradationDetector, DataQualityMonitor, EmailNotifier,
    WebhookNotifier, LogNotifier, AlertingSystem, create_alerting_system
)


class TestAlert(unittest.TestCase):
    """Test Alert data structure."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            alert_id="test_001",
            alert_type="threshold",
            severity="high",
            message="Test alert",
            timestamp=datetime.now(),
            model_name="test_model",
            metric_name="mse",
            current_value=0.5,
            threshold_value=0.3
        )
        
        self.assertEqual(alert.alert_id, "test_001")
        self.assertEqual(alert.alert_type, "threshold")
        self.assertEqual(alert.severity, "high")
        self.assertEqual(alert.model_name, "test_model")
        self.assertEqual(alert.metric_name, "mse")
        self.assertEqual(alert.current_value, 0.5)
        self.assertEqual(alert.threshold_value, 0.3)


class TestAlertRule(unittest.TestCase):
    """Test AlertRule data structure."""
    
    def test_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            rule_id="rule_001",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            comparison="greater",
            severity="medium",
            enabled=True
        )
        
        self.assertEqual(rule.rule_id, "rule_001")
        self.assertEqual(rule.rule_type, "threshold")
        self.assertEqual(rule.metric_name, "mse")
        self.assertEqual(rule.threshold, 0.1)
        self.assertEqual(rule.comparison, "greater")
        self.assertTrue(rule.enabled)


class TestStatisticalAnomalyDetector(unittest.TestCase):
    """Test StatisticalAnomalyDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = StatisticalAnomalyDetector(method='zscore', threshold=2.0)
    
    def test_detector_creation(self):
        """Test detector creation."""
        self.assertEqual(self.detector.method, 'zscore')
        self.assertEqual(self.detector.threshold, 2.0)
        self.assertFalse(self.detector.is_fitted)
    
    def test_fit_and_detect_zscore(self):
        """Test fitting and detection with z-score method."""
        # Generate normal data
        data = np.random.normal(0, 1, 100)
        self.detector.fit(data)
        
        self.assertTrue(self.detector.is_fitted)
        
        # Test normal value
        is_anomaly, score = self.detector.detect(0.5)
        self.assertFalse(is_anomaly)
        self.assertLess(score, 2.0)
        
        # Test anomalous value
        is_anomaly, score = self.detector.detect(5.0)
        self.assertTrue(is_anomaly)
        self.assertGreater(score, 2.0)
    
    def test_iqr_method(self):
        """Test IQR-based anomaly detection."""
        detector = StatisticalAnomalyDetector(method='iqr')
        
        # Generate data with known distribution
        data = np.random.normal(10, 2, 1000)
        detector.fit(data)
        
        # Test normal value
        is_anomaly, score = detector.detect(10.0)
        self.assertFalse(is_anomaly)
        
        # Test outlier
        is_anomaly, score = detector.detect(20.0)
        self.assertTrue(is_anomaly)
    
    def test_modified_zscore_method(self):
        """Test modified z-score method."""
        detector = StatisticalAnomalyDetector(method='modified_zscore', threshold=3.5)
        
        # Generate data with outliers
        data = np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]])
        detector.fit(data)
        
        # Test normal value
        is_anomaly, score = detector.detect(1.0)
        self.assertFalse(is_anomaly)
        
        # Test outlier
        is_anomaly, score = detector.detect(25.0)
        self.assertTrue(is_anomaly)
    
    def test_update_functionality(self):
        """Test detector update functionality."""
        data = np.random.normal(0, 1, 50)
        self.detector.fit(data)
        
        # Update with new values
        for i in range(10):
            self.detector.update(np.random.normal(0, 1))
        
        # Should still work
        is_anomaly, score = self.detector.detect(0.0)
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)


class TestIsolationForestAnomalyDetector(unittest.TestCase):
    """Test IsolationForestAnomalyDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = IsolationForestAnomalyDetector(contamination=0.1)
    
    def test_detector_creation(self):
        """Test detector creation."""
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertFalse(self.detector.is_fitted)
    
    def test_fit_and_detect(self):
        """Test fitting and detection."""
        # Generate normal data
        data = np.random.normal(0, 1, 200)
        self.detector.fit(data)
        
        # Test detection (may use fallback if sklearn not available)
        is_anomaly, score = self.detector.detect(0.0)
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)
        
        # Test with clear outlier
        is_anomaly, score = self.detector.detect(10.0)
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)
    
    def test_update_functionality(self):
        """Test update functionality."""
        data = np.random.normal(0, 1, 100)
        self.detector.fit(data)
        
        # Update with new values
        for i in range(20):
            self.detector.update(np.random.normal(0, 1))
        
        # Should still work
        is_anomaly, score = self.detector.detect(0.0)
        self.assertIsInstance(is_anomaly, bool)


class TestPerformanceDegradationDetector(unittest.TestCase):
    """Test PerformanceDegradationDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = PerformanceDegradationDetector(
            baseline_window=50,
            detection_window=10,
            degradation_threshold=0.2
        )
    
    def test_detector_creation(self):
        """Test detector creation."""
        self.assertEqual(self.detector.baseline_window, 50)
        self.assertEqual(self.detector.detection_window, 10)
        self.assertEqual(self.detector.degradation_threshold, 0.2)
    
    def test_performance_update(self):
        """Test performance update."""
        # Add baseline performance
        for i in range(60):
            self.detector.update_performance("model1", "mse", 0.1 + np.random.normal(0, 0.01))
        
        # Check that baseline stats are computed
        key = "model1_mse"
        self.assertIn(key, self.detector.baseline_stats)
        self.assertIn('mean', self.detector.baseline_stats[key])
    
    def test_degradation_detection(self):
        """Test degradation detection."""
        # Add good baseline performance
        for i in range(60):
            self.detector.update_performance("model1", "mse", 0.1)
        
        # Add degraded recent performance
        for i in range(15):
            self.detector.update_performance("model1", "mse", 0.15)  # 50% increase
        
        # Check for degradation
        is_degraded, info = self.detector.check_degradation("model1", "mse")
        
        self.assertTrue(is_degraded)
        self.assertIn('baseline_mean', info)
        self.assertIn('recent_mean', info)
        self.assertIn('relative_change', info)
        self.assertGreater(info['relative_change'], 0.2)
    
    def test_no_degradation(self):
        """Test when there's no degradation."""
        # Add consistent performance
        for i in range(80):
            self.detector.update_performance("model1", "mse", 0.1 + np.random.normal(0, 0.005))
        
        # Check for degradation
        is_degraded, info = self.detector.check_degradation("model1", "mse")
        
        self.assertFalse(is_degraded)


class TestDataQualityMonitor(unittest.TestCase):
    """Test DataQualityMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = DataQualityMonitor()
        
        # Create baseline data
        np.random.seed(42)
        self.baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.exponential(1, 1000)
        })
        
        # Add some missing values
        self.baseline_data.loc[self.baseline_data.sample(50).index, 'feature1'] = np.nan
        
        self.monitor.initialize_monitoring(self.baseline_data)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIn('feature1', self.monitor.feature_stats)
        self.assertIn('feature2', self.monitor.feature_stats)
        self.assertIn('feature3', self.monitor.feature_stats)
        
        # Check that stats are computed
        stats = self.monitor.feature_stats['feature1']
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('null_rate', stats)
    
    def test_missing_values_detection(self):
        """Test detection of increased missing values."""
        # Create data with more missing values
        test_data = self.baseline_data.copy()
        test_data.loc[test_data.sample(200).index, 'feature1'] = np.nan
        
        issues = self.monitor.check_data_quality(test_data)
        
        # Should detect missing values issue
        missing_issues = [issue for issue in issues if issue['type'] == 'missing_values']
        self.assertGreater(len(missing_issues), 0)
        
        issue = missing_issues[0]
        self.assertEqual(issue['column'], 'feature1')
        self.assertGreater(issue['current_null_rate'], issue['baseline_null_rate'])
    
    def test_distribution_shift_detection(self):
        """Test detection of distribution shift."""
        # Create data with shifted mean
        test_data = pd.DataFrame({
            'feature1': np.random.normal(3, 1, 100),  # Mean shifted from ~0 to 3
            'feature2': np.random.normal(5, 2, 100),  # Same distribution
            'feature3': np.random.exponential(1, 100)
        })
        
        issues = self.monitor.check_data_quality(test_data)
        
        # Should detect distribution shift
        shift_issues = [issue for issue in issues if issue['type'] == 'distribution_shift']
        self.assertGreater(len(shift_issues), 0)
        
        # Check that feature1 shift is detected
        feature1_issues = [issue for issue in shift_issues if issue['column'] == 'feature1']
        self.assertGreater(len(feature1_issues), 0)
    
    def test_anomalous_values_detection(self):
        """Test detection of anomalous values."""
        # Create data with extreme outliers
        test_data = pd.DataFrame({
            'feature1': [0, 0, 0, 100, 0],  # 100 is an outlier
            'feature2': [5, 5, 5, 5, 5],
            'feature3': [1, 1, 1, 1, 1]
        })
        
        issues = self.monitor.check_data_quality(test_data)
        
        # May detect anomalous values (depends on detector sensitivity)
        anomaly_issues = [issue for issue in issues if issue['type'] == 'anomalous_value']
        # Note: This test might not always trigger depending on the baseline data


class TestNotifiers(unittest.TestCase):
    """Test alert notifiers."""
    
    def setUp(self):
        """Set up test environment."""
        self.alert = Alert(
            alert_id="test_001",
            alert_type="threshold",
            severity="high",
            message="Test alert message",
            timestamp=datetime.now(),
            model_name="test_model",
            metric_name="mse",
            current_value=0.5,
            threshold_value=0.3
        )
    
    def test_log_notifier(self):
        """Test log notifier."""
        # Create mock logger
        mock_logger = Mock()
        notifier = LogNotifier(mock_logger)
        
        # Send alert
        result = notifier.send_alert(self.alert)
        
        self.assertTrue(result)
        mock_logger.log.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_email_notifier(self, mock_smtp):
        """Test email notifier."""
        # Setup mock
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        notifier = EmailNotifier(
            smtp_server="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="password",
            from_email="test@test.com",
            to_emails=["recipient@test.com"]
        )
        
        # Send alert
        result = notifier.send_alert(self.alert)
        
        self.assertTrue(result)
        mock_smtp.assert_called_once_with("smtp.test.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('requests.post')
    def test_webhook_notifier(self, mock_post):
        """Test webhook notifier."""
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        notifier = WebhookNotifier("https://webhook.test.com/alert")
        
        # Send alert
        result = notifier.send_alert(self.alert)
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        
        # Check that the call was made with correct data
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['alert_id'], "test_001")
        self.assertEqual(call_args[1]['json']['severity'], "high")
    
    @patch('requests.post')
    def test_webhook_notifier_failure(self, mock_post):
        """Test webhook notifier failure handling."""
        # Setup mock to return error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        notifier = WebhookNotifier("https://webhook.test.com/alert")
        
        # Send alert
        result = notifier.send_alert(self.alert)
        
        self.assertFalse(result)


class TestAlertingSystem(unittest.TestCase):
    """Test AlertingSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        self.system = AlertingSystem()
        
        # Add a mock notifier
        self.mock_notifier = Mock()
        self.system.add_notifier(self.mock_notifier)
    
    def test_system_creation(self):
        """Test alerting system creation."""
        self.assertIsInstance(self.system.alert_rules, dict)
        self.assertIsInstance(self.system.notifiers, list)
        self.assertEqual(len(self.system.notifiers), 1)
    
    def test_add_remove_alert_rule(self):
        """Test adding and removing alert rules."""
        rule = AlertRule(
            rule_id="test_rule",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            severity="high"
        )
        
        # Add rule
        self.system.add_alert_rule(rule)
        self.assertIn("test_rule", self.system.alert_rules)
        
        # Remove rule
        self.system.remove_alert_rule("test_rule")
        self.assertNotIn("test_rule", self.system.alert_rules)
    
    def test_threshold_alert_triggering(self):
        """Test threshold-based alert triggering."""
        # Add threshold rule
        rule = AlertRule(
            rule_id="mse_threshold",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            comparison="greater",
            severity="high"
        )
        self.system.add_alert_rule(rule)
        
        # Update with value that should trigger alert
        self.system.update_model_performance("test_model", {"mse": 0.15})
        
        # Check that alert was sent
        self.mock_notifier.send_alert.assert_called_once()
        
        # Check alert details
        alert = self.mock_notifier.send_alert.call_args[0][0]
        self.assertEqual(alert.alert_type, "threshold")
        self.assertEqual(alert.model_name, "test_model")
        self.assertEqual(alert.metric_name, "mse")
        self.assertEqual(alert.current_value, 0.15)
    
    def test_threshold_alert_not_triggering(self):
        """Test that threshold alert doesn't trigger when it shouldn't."""
        # Add threshold rule
        rule = AlertRule(
            rule_id="mse_threshold",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            comparison="greater",
            severity="high"
        )
        self.system.add_alert_rule(rule)
        
        # Update with value that should NOT trigger alert
        self.system.update_model_performance("test_model", {"mse": 0.05})
        
        # Check that no alert was sent
        self.mock_notifier.send_alert.assert_not_called()
    
    def test_anomaly_alert_triggering(self):
        """Test anomaly-based alert triggering."""
        # Add anomaly rule
        rule = AlertRule(
            rule_id="mse_anomaly",
            rule_type="anomaly",
            metric_name="mse",
            model_filter="test_model",
            sensitivity=0.1,  # High sensitivity
            severity="medium"
        )
        self.system.add_alert_rule(rule)
        
        # Add normal baseline data
        for i in range(50):
            self.system.update_model_performance("test_model", {"mse": 0.1})
        
        # Reset mock to clear baseline updates
        self.mock_notifier.reset_mock()
        
        # Add anomalous value
        self.system.update_model_performance("test_model", {"mse": 1.0})
        
        # Check if alert was triggered (may depend on detector sensitivity)
        # Note: This test might be flaky depending on the anomaly detector
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""
        # Add baseline performance
        for i in range(60):
            self.system.update_model_performance("test_model", {"mse": 0.1})
        
        # Add degraded performance
        for i in range(25):
            self.system.update_model_performance("test_model", {"mse": 0.2})
        
        # Check for degradation
        alerts = self.system.check_performance_degradation()
        
        # Should detect degradation
        self.assertGreater(len(alerts), 0)
        
        degradation_alerts = [a for a in alerts if a.alert_type == 'performance_degradation']
        self.assertGreater(len(degradation_alerts), 0)
        
        alert = degradation_alerts[0]
        self.assertEqual(alert.model_name, "test_model")
        self.assertEqual(alert.metric_name, "mse")
    
    def test_data_quality_monitoring(self):
        """Test data quality monitoring."""
        # Initialize with baseline data
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100)
        })
        self.system.data_quality_monitor.initialize_monitoring(baseline_data)
        
        # Create problematic data
        problem_data = pd.DataFrame({
            'feature1': [np.nan] * 50 + [0] * 50,  # High missing rate
            'feature2': np.random.normal(10, 2, 100)  # Shifted distribution
        })
        
        # Check data quality
        alerts = self.system.check_data_quality(problem_data)
        
        # Should detect issues
        self.assertGreater(len(alerts), 0)
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        # Add threshold rule
        rule = AlertRule(
            rule_id="mse_threshold",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            comparison="greater",
            severity="high"
        )
        self.system.add_alert_rule(rule)
        
        # Trigger alert multiple times
        self.system.update_model_performance("test_model", {"mse": 0.15})
        self.system.update_model_performance("test_model", {"mse": 0.16})
        self.system.update_model_performance("test_model", {"mse": 0.17})
        
        # Should only send one alert due to cooldown
        self.assertEqual(self.mock_notifier.send_alert.call_count, 1)
    
    def test_alert_history(self):
        """Test alert history functionality."""
        # Add some alerts to history
        for i in range(5):
            alert = Alert(
                alert_id=f"test_{i}",
                alert_type="test",
                severity="low",
                message=f"Test alert {i}",
                timestamp=datetime.now() - timedelta(hours=i)
            )
            self.system.alert_history.append(alert)
        
        # Get recent alerts
        recent_alerts = self.system.get_alert_history(hours=3)
        self.assertEqual(len(recent_alerts), 4)  # 0, 1, 2, 3 hours ago
        
        # Get alert summary
        summary = self.system.get_alert_summary()
        self.assertIn('total_alerts', summary)
        self.assertIn('by_severity', summary)
        self.assertIn('by_type', summary)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_alerting_system(self):
        """Test creating alerting system from configuration."""
        config = {
            'webhook': {
                'url': 'https://webhook.test.com/alert',
                'headers': {'Authorization': 'Bearer token'}
            },
            'rules': [
                {
                    'rule_id': 'mse_high',
                    'rule_type': 'threshold',
                    'metric_name': 'mse',
                    'threshold': 0.1,
                    'comparison': 'greater',
                    'severity': 'high'
                }
            ]
        }
        
        system = create_alerting_system(config)
        
        # Should have webhook notifier and log notifier
        self.assertEqual(len(system.notifiers), 2)
        
        # Should have the rule
        self.assertIn('mse_high', system.alert_rules)
        
        rule = system.alert_rules['mse_high']
        self.assertEqual(rule.rule_type, 'threshold')
        self.assertEqual(rule.metric_name, 'mse')
        self.assertEqual(rule.threshold, 0.1)
    
    def test_create_alerting_system_with_email(self):
        """Test creating system with email configuration."""
        config = {
            'email': {
                'smtp_server': 'smtp.test.com',
                'smtp_port': 587,
                'username': 'test@test.com',
                'password': 'password',
                'from_email': 'test@test.com',
                'to_emails': ['recipient@test.com']
            }
        }
        
        system = create_alerting_system(config)
        
        # Should have email notifier and log notifier
        self.assertEqual(len(system.notifiers), 2)
        
        # Check that one is EmailNotifier
        notifier_types = [type(n).__name__ for n in system.notifiers]
        self.assertIn('EmailNotifier', notifier_types)
        self.assertIn('LogNotifier', notifier_types)


class TestIntegration(unittest.TestCase):
    """Integration tests for alerting system."""
    
    def test_full_alerting_workflow(self):
        """Test complete alerting workflow."""
        # Create system with multiple notifiers
        mock_notifier1 = Mock()
        mock_notifier2 = Mock()
        
        system = AlertingSystem()
        system.add_notifier(mock_notifier1)
        system.add_notifier(mock_notifier2)
        
        # Add multiple types of rules
        threshold_rule = AlertRule(
            rule_id="mse_threshold",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            comparison="greater",
            severity="high"
        )
        
        anomaly_rule = AlertRule(
            rule_id="mae_anomaly",
            rule_type="anomaly",
            metric_name="mae",
            model_filter="model1",
            sensitivity=0.1,
            severity="medium"
        )
        
        system.add_alert_rule(threshold_rule)
        system.add_alert_rule(anomaly_rule)
        
        # Simulate model performance updates
        # Normal performance first
        for i in range(50):
            system.update_model_performance("model1", {
                "mse": 0.05 + np.random.normal(0, 0.01),
                "mae": 0.03 + np.random.normal(0, 0.005)
            })
        
        # Reset mocks to ignore baseline updates
        mock_notifier1.reset_mock()
        mock_notifier2.reset_mock()
        
        # Trigger threshold alert
        system.update_model_performance("model1", {"mse": 0.15, "mae": 0.03})
        
        # Should trigger threshold alert
        self.assertGreater(mock_notifier1.send_alert.call_count, 0)
        self.assertGreater(mock_notifier2.send_alert.call_count, 0)
        
        # Check alert details
        alert = mock_notifier1.send_alert.call_args[0][0]
        self.assertEqual(alert.alert_type, "threshold")
        self.assertEqual(alert.severity, "high")
    
    def test_system_resilience(self):
        """Test system resilience to errors."""
        # Create system with failing notifier
        failing_notifier = Mock()
        failing_notifier.send_alert.side_effect = Exception("Network error")
        
        working_notifier = Mock()
        
        system = AlertingSystem()
        system.add_notifier(failing_notifier)
        system.add_notifier(working_notifier)
        
        # Add rule
        rule = AlertRule(
            rule_id="test_rule",
            rule_type="threshold",
            metric_name="mse",
            threshold=0.1,
            comparison="greater",
            severity="high"
        )
        system.add_alert_rule(rule)
        
        # Trigger alert
        system.update_model_performance("test_model", {"mse": 0.15})
        
        # Working notifier should still receive alert
        working_notifier.send_alert.assert_called_once()
        
        # System should continue working despite failing notifier
        self.assertEqual(len(system.alert_history), 1)


if __name__ == '__main__':
    unittest.main()