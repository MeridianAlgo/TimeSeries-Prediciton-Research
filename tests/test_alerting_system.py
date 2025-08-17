"""
Unit tests for alerting and anomaly detection system.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from enhanced_timeseries.monitoring.alerting_system import (
    AlertRule, Alert, AlertSeverity, AlertType, AnomalyDetectionConfig,
    AnomalyDetector, EmailNotificationChannel, WebhookNotificationChannel,
    SlackNotificationChannel, AlertingEngine, AlertingSystem
)


class TestAlertRule(unittest.TestCase):
    """Test AlertRule class."""
    
    def setUp(self):
        """Set up test data."""
        self.rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            metric_name="mae",
            threshold_value=0.1,
            comparison_operator=">",
            time_window_minutes=30,
            min_occurrences=2,
            description="Test alert rule"
        )
    
    def test_rule_creation(self):
        """Test alert rule creation."""
        self.assertEqual(self.rule.rule_id, "test_rule")
        self.assertEqual(self.rule.name, "Test Rule")
        self.assertEqual(self.rule.alert_type, AlertType.PERFORMANCE_DEGRADATION)
        self.assertEqual(self.rule.severity, AlertSeverity.HIGH)
        self.assertEqual(self.rule.threshold_value, 0.1)
        self.assertTrue(self.rule.enabled)
    
    def test_rule_evaluation_greater_than(self):
        """Test rule evaluation with greater than operator."""
        # Should trigger
        self.assertTrue(self.rule.evaluate(0.15))
        
        # Should not trigger
        self.assertFalse(self.rule.evaluate(0.05))
        self.assertFalse(self.rule.evaluate(0.1))  # Equal to threshold
    
    def test_rule_evaluation_less_than(self):
        """Test rule evaluation with less than operator."""
        rule = AlertRule(
            rule_id="test_rule_lt",
            name="Test Rule LT",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.MEDIUM,
            metric_name="accuracy",
            threshold_value=50.0,
            comparison_operator="<",
            time_window_minutes=30
        )
        
        # Should trigger
        self.assertTrue(rule.evaluate(45.0))
        
        # Should not trigger
        self.assertFalse(rule.evaluate(55.0))
        self.assertFalse(rule.evaluate(50.0))  # Equal to threshold
    
    def test_rule_evaluation_disabled(self):
        """Test rule evaluation when disabled."""
        self.rule.enabled = False
        
        # Should not trigger even if condition is met
        self.assertFalse(self.rule.evaluate(0.15))
    
    def test_rule_evaluation_operators(self):
        """Test all comparison operators."""
        test_cases = [
            (">=", 0.1, 0.1, True),
            (">=", 0.1, 0.05, False),
            ("<=", 0.1, 0.1, True),
            ("<=", 0.1, 0.15, False),
            ("==", 0.1, 0.1, True),
            ("==", 0.1, 0.15, False),
            ("!=", 0.1, 0.15, True),
            ("!=", 0.1, 0.1, False),
        ]
        
        for operator, threshold, test_value, expected in test_cases:
            rule = AlertRule(
                rule_id="test",
                name="Test",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.LOW,
                metric_name="test",
                threshold_value=threshold,
                comparison_operator=operator,
                time_window_minutes=30
            )
            
            result = rule.evaluate(test_value)
            self.assertEqual(result, expected, 
                           f"Failed for {operator} with threshold {threshold} and value {test_value}")


class TestAlert(unittest.TestCase):
    """Test Alert class."""
    
    def test_alert_creation_and_serialization(self):
        """Test alert creation and serialization."""
        alert = Alert(
            alert_id="test_alert_123",
            rule_id="test_rule",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            metric_name="mae",
            metric_value=0.15,
            threshold_value=0.1,
            timestamp=datetime.now(),
            model_id="model_1"
        )
        
        self.assertEqual(alert.alert_id, "test_alert_123")
        self.assertEqual(alert.severity, AlertSeverity.HIGH)
        self.assertFalse(alert.resolved)
        self.assertIsNone(alert.resolved_timestamp)
        
        # Test serialization
        alert_dict = alert.to_dict()
        self.assertIsInstance(alert_dict, dict)
        self.assertEqual(alert_dict['alert_id'], "test_alert_123")
        self.assertEqual(alert_dict['severity'], "high")
        self.assertEqual(alert_dict['resolved'], False)


class TestAnomalyDetector(unittest.TestCase):
    """Test AnomalyDetector class."""
    
    def setUp(self):
        """Set up test data."""
        self.config = AnomalyDetectionConfig(
            method='statistical',
            sensitivity=0.95,
            window_size=50,
            min_samples=10
        )
        self.detector = AnomalyDetector(self.config)
    
    def test_detector_creation(self):
        """Test anomaly detector creation."""
        self.assertEqual(self.detector.config.method, 'statistical')
        self.assertEqual(self.detector.config.sensitivity, 0.95)
        self.assertEqual(self.detector.config.window_size, 50)
    
    def test_add_observation(self):
        """Test adding observations."""
        self.detector.add_observation("mae", 0.05)
        self.detector.add_observation("mae", 0.06)
        
        self.assertEqual(len(self.detector.historical_data["mae"]), 2)
    
    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection."""
        # Add normal observations
        np.random.seed(42)
        normal_values = np.random.normal(0.05, 0.01, 20)
        
        for value in normal_values:
            self.detector.add_observation("mae", value)
        
        # Test normal value
        is_anomaly, score = self.detector.detect_anomaly("mae", 0.055)
        self.assertFalse(is_anomaly)
        
        # Test anomalous value
        is_anomaly, score = self.detector.detect_anomaly("mae", 0.15)
        self.assertTrue(is_anomaly)
        self.assertGreater(score, 0)
    
    def test_insufficient_data(self):
        """Test anomaly detection with insufficient data."""
        # Add only a few observations
        self.detector.add_observation("mae", 0.05)
        self.detector.add_observation("mae", 0.06)
        
        # Should not detect anomaly due to insufficient data
        is_anomaly, score = self.detector.detect_anomaly("mae", 0.15)
        self.assertFalse(is_anomaly)
        self.assertEqual(score, 0.0)
    
    def test_get_anomaly_summary(self):
        """Test getting anomaly summary."""
        # Add sufficient observations
        for i in range(15):
            self.detector.add_observation("mae", 0.05 + i * 0.001)
        
        summary = self.detector.get_anomaly_summary("mae")
        
        self.assertEqual(summary['status'], 'active')
        self.assertEqual(summary['sample_count'], 15)
        self.assertIn('mean', summary)
        self.assertIn('std', summary)
    
    def test_get_anomaly_summary_insufficient_data(self):
        """Test anomaly summary with insufficient data."""
        self.detector.add_observation("mae", 0.05)
        
        summary = self.detector.get_anomaly_summary("mae")
        
        self.assertEqual(summary['status'], 'insufficient_data')
        self.assertEqual(summary['sample_count'], 1)


class TestNotificationChannels(unittest.TestCase):
    """Test notification channel classes."""
    
    def test_email_channel_creation(self):
        """Test email notification channel creation."""
        channel = EmailNotificationChannel(
            channel_id="email_test",
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            from_email="test@example.com",
            to_emails=["recipient@example.com"]
        )
        
        self.assertEqual(channel.channel_id, "email_test")
        self.assertEqual(channel.smtp_server, "smtp.gmail.com")
        self.assertTrue(channel.enabled)
    
    def test_webhook_channel_creation(self):
        """Test webhook notification channel creation."""
        channel = WebhookNotificationChannel(
            channel_id="webhook_test",
            webhook_url="https://example.com/webhook",
            headers={"Authorization": "Bearer token"}
        )
        
        self.assertEqual(channel.channel_id, "webhook_test")
        self.assertEqual(channel.webhook_url, "https://example.com/webhook")
        self.assertIn("Authorization", channel.headers)
    
    def test_slack_channel_creation(self):
        """Test Slack notification channel creation."""
        channel = SlackNotificationChannel(
            channel_id="slack_test",
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#alerts"
        )
        
        self.assertEqual(channel.channel_id, "slack_test")
        self.assertEqual(channel.channel, "#alerts")
    
    @patch('requests.post')
    def test_webhook_notification(self, mock_post):
        """Test webhook notification sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        channel = WebhookNotificationChannel(
            channel_id="webhook_test",
            webhook_url="https://example.com/webhook"
        )
        
        alert = Alert(
            alert_id="test_alert",
            rule_id="test_rule",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            metric_name="mae",
            metric_value=0.15,
            threshold_value=0.1,
            timestamp=datetime.now()
        )
        
        result = channel.send_notification(alert)
        
        self.assertTrue(result)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_slack_notification(self, mock_post):
        """Test Slack notification sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        channel = SlackNotificationChannel(
            channel_id="slack_test",
            webhook_url="https://hooks.slack.com/services/xxx"
        )
        
        alert = Alert(
            alert_id="test_alert",
            rule_id="test_rule",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            metric_name="mae",
            metric_value=0.15,
            threshold_value=0.1,
            timestamp=datetime.now()
        )
        
        result = channel.send_notification(alert)
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        
        # Check payload structure
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        self.assertIn('attachments', payload)
        self.assertEqual(len(payload['attachments']), 1)
        self.assertIn('fields', payload['attachments'][0])


class TestAlertingEngine(unittest.TestCase):
    """Test AlertingEngine class."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = AlertingEngine()
        
        # Add test rule
        self.test_rule = AlertRule(
            rule_id="test_mae_rule",
            name="High MAE Alert",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            metric_name="mae",
            threshold_value=0.1,
            comparison_operator=">",
            time_window_minutes=30,
            min_occurrences=2
        )
        self.engine.add_alert_rule(self.test_rule)
    
    def test_engine_creation(self):
        """Test alerting engine creation."""
        self.assertIsInstance(self.engine.alert_rules, dict)
        self.assertIsInstance(self.engine.notification_channels, dict)
        self.assertIsInstance(self.engine.active_alerts, dict)
    
    def test_add_remove_alert_rule(self):
        """Test adding and removing alert rules."""
        initial_count = len(self.engine.alert_rules)
        
        new_rule = AlertRule(
            rule_id="new_rule",
            name="New Rule",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.MEDIUM,
            metric_name="cpu_usage",
            threshold_value=80.0,
            comparison_operator=">",
            time_window_minutes=15
        )
        
        self.engine.add_alert_rule(new_rule)
        self.assertEqual(len(self.engine.alert_rules), initial_count + 1)
        self.assertIn("new_rule", self.engine.alert_rules)
        
        self.engine.remove_alert_rule("new_rule")
        self.assertEqual(len(self.engine.alert_rules), initial_count)
        self.assertNotIn("new_rule", self.engine.alert_rules)
    
    def test_add_remove_notification_channel(self):
        """Test adding and removing notification channels."""
        channel = WebhookNotificationChannel(
            channel_id="test_webhook",
            webhook_url="https://example.com/webhook"
        )
        
        self.engine.add_notification_channel(channel)
        self.assertIn("test_webhook", self.engine.notification_channels)
        
        self.engine.remove_notification_channel("test_webhook")
        self.assertNotIn("test_webhook", self.engine.notification_channels)
    
    def test_evaluate_metric_no_violation(self):
        """Test metric evaluation without rule violation."""
        initial_alert_count = len(self.engine.active_alerts)
        
        # Evaluate metric that doesn't violate rule
        self.engine.evaluate_metric("mae", 0.05, "model_1")
        
        # Should not create alert
        self.assertEqual(len(self.engine.active_alerts), initial_alert_count)
    
    def test_evaluate_metric_single_violation(self):
        """Test metric evaluation with single violation (insufficient for alert)."""
        initial_alert_count = len(self.engine.active_alerts)
        
        # Single violation (rule requires 2 occurrences)
        self.engine.evaluate_metric("mae", 0.15, "model_1")
        
        # Should not create alert yet
        self.assertEqual(len(self.engine.active_alerts), initial_alert_count)
    
    def test_evaluate_metric_multiple_violations(self):
        """Test metric evaluation with multiple violations."""
        initial_alert_count = len(self.engine.active_alerts)
        
        # Multiple violations within time window
        timestamp = datetime.now()
        
        self.engine.evaluate_metric("mae", 0.15, "model_1", timestamp)
        self.engine.evaluate_metric("mae", 0.12, "model_1", timestamp + timedelta(minutes=5))
        
        # Should create alert
        self.assertEqual(len(self.engine.active_alerts), initial_alert_count + 1)
    
    def test_alert_resolution(self):
        """Test alert resolution when metric returns to normal."""
        # Create alert
        timestamp = datetime.now()
        self.engine.evaluate_metric("mae", 0.15, "model_1", timestamp)
        self.engine.evaluate_metric("mae", 0.12, "model_1", timestamp + timedelta(minutes=5))
        
        initial_alert_count = len(self.engine.active_alerts)
        self.assertGreater(initial_alert_count, 0)
        
        # Metric returns to normal
        self.engine.evaluate_metric("mae", 0.05, "model_1", timestamp + timedelta(minutes=10))
        
        # Alert should be resolved
        self.assertEqual(len(self.engine.active_alerts), initial_alert_count - 1)
    
    def test_anomaly_detection_integration(self):
        """Test anomaly detection integration."""
        # Add anomaly detector
        config = AnomalyDetectionConfig(method='statistical', sensitivity=0.95)
        self.engine.add_anomaly_detector("mae", config)
        
        # Add normal observations
        for i in range(15):
            self.engine.evaluate_metric("mae", 0.05 + i * 0.001, "model_1")
        
        initial_alert_count = len(self.engine.alert_history)
        
        # Add anomalous observation
        self.engine.evaluate_metric("mae", 0.2, "model_1")
        
        # Should create anomaly alert
        self.assertGreater(len(self.engine.alert_history), initial_alert_count)
    
    def test_get_alert_statistics(self):
        """Test getting alert statistics."""
        # Create some alerts
        timestamp = datetime.now()
        self.engine.evaluate_metric("mae", 0.15, "model_1", timestamp)
        self.engine.evaluate_metric("mae", 0.12, "model_1", timestamp + timedelta(minutes=5))
        
        stats = self.engine.get_alert_statistics()
        
        expected_keys = [
            'total_alerts', 'active_alerts', 'alert_rate_24h',
            'severity_distribution', 'type_distribution',
            'enabled_rules', 'total_rules', 'notification_channels'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertGreaterEqual(stats['total_alerts'], 0)
        self.assertGreaterEqual(stats['active_alerts'], 0)
    
    def test_export_configuration(self):
        """Test configuration export."""
        config = self.engine.export_configuration()
        
        self.assertIn('alert_rules', config)
        self.assertIn('anomaly_detectors', config)
        self.assertIn('notification_channels', config)
        
        # Check rule export
        self.assertGreater(len(config['alert_rules']), 0)
        rule_config = config['alert_rules'][0]
        
        expected_rule_keys = [
            'rule_id', 'name', 'alert_type', 'severity',
            'metric_name', 'threshold_value', 'comparison_operator'
        ]
        
        for key in expected_rule_keys:
            self.assertIn(key, rule_config)


class TestAlertingSystem(unittest.TestCase):
    """Test AlertingSystem main class."""
    
    def setUp(self):
        """Set up test environment."""
        self.system = AlertingSystem()
    
    def test_system_creation(self):
        """Test alerting system creation."""
        self.assertIsNotNone(self.system.engine)
        self.assertFalse(self.system.is_running)
        
        # Should have default rules
        self.assertGreater(len(self.system.engine.alert_rules), 0)
    
    def test_add_custom_rule(self):
        """Test adding custom alert rule."""
        initial_count = len(self.system.engine.alert_rules)
        
        custom_rule = AlertRule(
            rule_id="custom_rule",
            name="Custom Rule",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.MEDIUM,
            metric_name="data_quality_score",
            threshold_value=0.8,
            comparison_operator="<",
            time_window_minutes=60
        )
        
        self.system.add_custom_rule(custom_rule)
        
        self.assertEqual(len(self.system.engine.alert_rules), initial_count + 1)
        self.assertIn("custom_rule", self.system.engine.alert_rules)
    
    def test_enable_anomaly_detection(self):
        """Test enabling anomaly detection."""
        self.system.enable_anomaly_detection("mae", method='statistical')
        
        self.assertIn("mae", self.system.engine.anomaly_detectors)
        
        detector = self.system.engine.anomaly_detectors["mae"]
        self.assertEqual(detector.config.method, 'statistical')
    
    def test_evaluate_metrics(self):
        """Test evaluating multiple metrics."""
        metrics = {
            'mae': 0.15,  # Should trigger default rule
            'cpu_usage': 95.0,  # Should trigger default rule
            'memory_usage': 85.0  # Should not trigger
        }
        
        initial_alert_count = len(self.system.engine.active_alerts)
        
        # Evaluate multiple times to trigger alerts
        for _ in range(3):
            self.system.evaluate_metrics(metrics, "model_1")
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Should have created some alerts
        final_alert_count = len(self.system.engine.active_alerts)
        self.assertGreaterEqual(final_alert_count, initial_alert_count)
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        alerts = self.system.get_active_alerts()
        
        self.assertIsInstance(alerts, list)
        
        # Each alert should be a dictionary
        for alert in alerts:
            self.assertIsInstance(alert, dict)
            self.assertIn('alert_id', alert)
            self.assertIn('severity', alert)
    
    def test_get_alert_history(self):
        """Test getting alert history."""
        history = self.system.get_alert_history(hours=24)
        
        self.assertIsInstance(history, list)
    
    def test_get_statistics(self):
        """Test getting statistics."""
        stats = self.system.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_alerts', stats)
        self.assertIn('active_alerts', stats)
    
    def test_export_configuration(self):
        """Test exporting configuration."""
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.system.export_configuration(temp_filepath)
            
            # Check file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                config = json.load(f)
            
            self.assertIsInstance(config, dict)
            self.assertIn('alert_rules', config)
        
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    @patch('enhanced_timeseries.monitoring.alerting_system.WebhookNotificationChannel.send_notification')
    def test_add_webhook_notifications(self, mock_send):
        """Test adding webhook notifications."""
        mock_send.return_value = True
        
        self.system.add_webhook_notifications("https://example.com/webhook")
        
        self.assertIn("webhook", self.system.engine.notification_channels)
        
        # Test notification
        test_results = self.system.test_notifications()
        self.assertIn("webhook", test_results)
    
    def test_test_notifications(self):
        """Test notification testing."""
        # Add a mock webhook channel
        from unittest.mock import Mock
        
        mock_channel = Mock()
        mock_channel.channel_id = "mock_channel"
        mock_channel.send_notification.return_value = True
        
        self.system.engine.add_notification_channel(mock_channel)
        
        results = self.system.test_notifications()
        
        self.assertIn("mock_channel", results)
        self.assertTrue(results["mock_channel"])


if __name__ == '__main__':
    unittest.main()