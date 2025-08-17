"""
Automated alerting and anomaly detection system for time series models.
Implements accuracy degradation detection, data quality monitoring, and alert generation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import json

# Optional imports for email functionality
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# Optional import for HTTP requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
from collections import deque, defaultdict
import logging

warnings.filterwarnings('ignore')


@dataclass
class Alert:
    """Alert data structure."""
    
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    model_name: Optional[str] = None
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    additional_data: Optional[Dict] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    rule_id: str
    rule_type: str  # 'threshold', 'trend', 'anomaly'
    metric_name: str
    threshold: Optional[float] = None
    comparison: str = 'greater'  # 'greater', 'less', 'equal'
    severity: str = 'medium'
    enabled: bool = True
    model_filter: Optional[str] = None  # Filter by model name
    window_size: int = 10  # For trend-based rules
    sensitivity: float = 0.05  # For anomaly detection


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection algorithms."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the anomaly detector to historical data."""
        pass
    
    @abstractmethod
    def detect(self, value: float) -> Tuple[bool, float]:
        """
        Detect if a value is anomalous.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        pass
    
    @abstractmethod
    def update(self, value: float) -> None:
        """Update the detector with a new value."""
        pass


class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detector using z-score and IQR methods."""
    
    def __init__(self, method: str = 'zscore', window_size: int = 100, 
                 threshold: float = 3.0):
        """
        Initialize statistical anomaly detector.
        
        Args:
            method: Detection method ('zscore', 'iqr', 'modified_zscore')
            window_size: Size of sliding window for statistics
            threshold: Threshold for anomaly detection
        """
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> None:
        """Fit detector to historical data."""
        self.data_window.extend(data[-self.window_size:])
        self.is_fitted = True
        
    def detect(self, value: float) -> Tuple[bool, float]:
        """Detect anomaly using statistical methods."""
        if not self.is_fitted or len(self.data_window) < 10:
            return False, 0.0
            
        data_array = np.array(self.data_window)
        
        if self.method == 'zscore':
            mean = np.mean(data_array)
            std = np.std(data_array)
            if std == 0:
                return False, 0.0
            z_score = abs(value - mean) / std
            return z_score > self.threshold, z_score
            
        elif self.method == 'iqr':
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            is_anomaly = value < lower_bound or value > upper_bound
            # Calculate anomaly score as distance from bounds
            if value < lower_bound:
                score = (lower_bound - value) / iqr if iqr > 0 else 0
            elif value > upper_bound:
                score = (value - upper_bound) / iqr if iqr > 0 else 0
            else:
                score = 0
            return is_anomaly, score
            
        elif self.method == 'modified_zscore':
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            if mad == 0:
                return False, 0.0
            modified_z_score = 0.6745 * (value - median) / mad
            return abs(modified_z_score) > self.threshold, abs(modified_z_score)
            
        return False, 0.0
    
    def update(self, value: float) -> None:
        """Update detector with new value."""
        self.data_window.append(value)


class IsolationForestAnomalyDetector(AnomalyDetector):
    """Isolation Forest-based anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, window_size: int = 1000):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            window_size: Size of training window
        """
        self.contamination = contamination
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.model = None
        self.is_fitted = False
        
        try:
            from sklearn.ensemble import IsolationForest
            self.IsolationForest = IsolationForest
        except ImportError:
            logging.warning("sklearn not available, falling back to statistical detector")
            self.fallback_detector = StatisticalAnomalyDetector()
            self.use_fallback = True
        else:
            self.use_fallback = False
    
    def fit(self, data: np.ndarray) -> None:
        """Fit Isolation Forest to data."""
        if self.use_fallback:
            self.fallback_detector.fit(data)
            return
            
        self.data_window.extend(data[-self.window_size:])
        
        if len(self.data_window) >= 50:  # Minimum samples for IF
            X = np.array(self.data_window).reshape(-1, 1)
            self.model = self.IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(X)
            self.is_fitted = True
    
    def detect(self, value: float) -> Tuple[bool, float]:
        """Detect anomaly using Isolation Forest."""
        if self.use_fallback:
            return self.fallback_detector.detect(value)
            
        if not self.is_fitted:
            return False, 0.0
            
        X = np.array([[value]])
        prediction = self.model.predict(X)[0]
        anomaly_score = -self.model.score_samples(X)[0]  # Negative for anomaly score
        
        is_anomaly = prediction == -1
        return is_anomaly, anomaly_score
    
    def update(self, value: float) -> None:
        """Update detector with new value."""
        if self.use_fallback:
            self.fallback_detector.update(value)
            return
            
        self.data_window.append(value)
        
        # Refit periodically
        if len(self.data_window) >= self.window_size and len(self.data_window) % 100 == 0:
            self.fit(np.array(self.data_window))


class PerformanceDegradationDetector:
    """Detector for model performance degradation."""
    
    def __init__(self, baseline_window: int = 100, detection_window: int = 20,
                 degradation_threshold: float = 0.1):
        """
        Initialize performance degradation detector.
        
        Args:
            baseline_window: Size of baseline performance window
            detection_window: Size of recent performance window
            degradation_threshold: Threshold for degradation detection (relative)
        """
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.degradation_threshold = degradation_threshold
        
        self.performance_history = defaultdict(lambda: deque(maxlen=baseline_window))
        self.baseline_stats = {}
        
    def update_performance(self, model_name: str, metric_name: str, value: float):
        """Update performance history for a model."""
        key = f"{model_name}_{metric_name}"
        self.performance_history[key].append(value)
        
        # Update baseline statistics
        if len(self.performance_history[key]) >= self.baseline_window // 2:
            history = list(self.performance_history[key])
            self.baseline_stats[key] = {
                'mean': np.mean(history),
                'std': np.std(history),
                'median': np.median(history)
            }
    
    def check_degradation(self, model_name: str, metric_name: str) -> Tuple[bool, Dict]:
        """
        Check if model performance has degraded.
        
        Returns:
            Tuple of (is_degraded, degradation_info)
        """
        key = f"{model_name}_{metric_name}"
        
        if key not in self.baseline_stats:
            return False, {}
            
        history = list(self.performance_history[key])
        if len(history) < self.detection_window:
            return False, {}
        
        # Get recent performance
        recent_performance = history[-self.detection_window:]
        recent_mean = np.mean(recent_performance)
        
        # Compare with baseline
        baseline_mean = self.baseline_stats[key]['mean']
        
        # For error metrics (lower is better), degradation means increase
        # For accuracy metrics (higher is better), degradation means decrease
        # Assume error metrics for now (MSE, MAE, etc.)
        relative_change = (recent_mean - baseline_mean) / (baseline_mean + 1e-8)
        
        is_degraded = relative_change > self.degradation_threshold
        
        degradation_info = {
            'baseline_mean': baseline_mean,
            'recent_mean': recent_mean,
            'relative_change': relative_change,
            'threshold': self.degradation_threshold,
            'recent_window_size': len(recent_performance)
        }
        
        return is_degraded, degradation_info


class DataQualityMonitor:
    """Monitor data quality and detect issues."""
    
    def __init__(self):
        """Initialize data quality monitor."""
        self.feature_stats = {}
        self.anomaly_detectors = {}
        
    def initialize_monitoring(self, data: pd.DataFrame):
        """Initialize monitoring with baseline data."""
        for column in data.columns:
            if data[column].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Initialize statistics
                self.feature_stats[column] = {
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'null_rate': data[column].isnull().mean()
                }
                
                # Initialize anomaly detector
                self.anomaly_detectors[column] = StatisticalAnomalyDetector(
                    method='modified_zscore',
                    threshold=3.0
                )
                self.anomaly_detectors[column].fit(data[column].dropna().values)
    
    def check_data_quality(self, data: pd.DataFrame) -> List[Dict]:
        """
        Check data quality and return list of issues.
        
        Returns:
            List of data quality issues
        """
        issues = []
        
        for column in data.columns:
            if column not in self.feature_stats:
                continue
                
            # Check for missing values
            null_rate = data[column].isnull().mean()
            baseline_null_rate = self.feature_stats[column]['null_rate']
            
            if null_rate > baseline_null_rate + 0.1:  # 10% increase in null rate
                issues.append({
                    'type': 'missing_values',
                    'column': column,
                    'current_null_rate': null_rate,
                    'baseline_null_rate': baseline_null_rate,
                    'severity': 'medium' if null_rate < 0.5 else 'high'
                })
            
            # Check for distribution shift
            if not data[column].isnull().all():
                current_mean = data[column].mean()
                current_std = data[column].std()
                
                baseline_mean = self.feature_stats[column]['mean']
                baseline_std = self.feature_stats[column]['std']
                
                # Check mean shift
                if abs(current_mean - baseline_mean) > 2 * baseline_std:
                    issues.append({
                        'type': 'distribution_shift',
                        'column': column,
                        'shift_type': 'mean',
                        'current_value': current_mean,
                        'baseline_value': baseline_mean,
                        'severity': 'medium'
                    })
                
                # Check variance change
                if baseline_std > 0 and abs(current_std - baseline_std) / baseline_std > 0.5:
                    issues.append({
                        'type': 'distribution_shift',
                        'column': column,
                        'shift_type': 'variance',
                        'current_value': current_std,
                        'baseline_value': baseline_std,
                        'severity': 'medium'
                    })
            
            # Check for anomalous values
            if column in self.anomaly_detectors:
                for value in data[column].dropna():
                    is_anomaly, score = self.anomaly_detectors[column].detect(value)
                    if is_anomaly and score > 5.0:  # High anomaly score
                        issues.append({
                            'type': 'anomalous_value',
                            'column': column,
                            'value': value,
                            'anomaly_score': score,
                            'severity': 'low'
                        })
        
        return issues


class AlertNotifier(ABC):
    """Abstract base class for alert notification systems."""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert notification."""
        pass


class EmailNotifier(AlertNotifier):
    """Email-based alert notifier."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        """
        Initialize email notifier.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not EMAIL_AVAILABLE:
            logging.error("Email functionality not available - missing email modules")
            return False
            
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.alert_type}: {alert.message}"
            
            # Create email body
            body = f"""
            Alert Details:
            - Alert ID: {alert.alert_id}
            - Type: {alert.alert_type}
            - Severity: {alert.severity}
            - Timestamp: {alert.timestamp}
            - Model: {alert.model_name or 'N/A'}
            - Metric: {alert.metric_name or 'N/A'}
            - Current Value: {alert.current_value or 'N/A'}
            - Threshold: {alert.threshold_value or 'N/A'}
            
            Message: {alert.message}
            
            Additional Data: {alert.additional_data or 'None'}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False


class WebhookNotifier(AlertNotifier):
    """Webhook-based alert notifier."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        """
        Initialize webhook notifier.
        
        Args:
            webhook_url: Webhook URL
            headers: Optional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not REQUESTS_AVAILABLE:
            logging.error("Webhook functionality not available - missing requests module")
            return False
            
        try:
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'model_name': alert.model_name,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'additional_data': alert.additional_data
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"Failed to send webhook alert: {e}")
            return False


class LogNotifier(AlertNotifier):
    """Log-based alert notifier."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize log notifier."""
        self.logger = logger or logging.getLogger(__name__)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to log."""
        try:
            log_level = {
                'low': logging.INFO,
                'medium': logging.WARNING,
                'high': logging.ERROR,
                'critical': logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            message = (f"ALERT [{alert.alert_id}] {alert.alert_type}: {alert.message} "
                      f"(Model: {alert.model_name}, Metric: {alert.metric_name}, "
                      f"Value: {alert.current_value}, Threshold: {alert.threshold_value})")
            
            self.logger.log(log_level, message)
            return True
            
        except Exception as e:
            logging.error(f"Failed to log alert: {e}")
            return False


class AlertingSystem:
    """Main alerting system that coordinates detection and notification."""
    
    def __init__(self):
        """Initialize alerting system."""
        self.alert_rules = {}
        self.notifiers = []
        self.performance_detector = PerformanceDegradationDetector()
        self.data_quality_monitor = DataQualityMonitor()
        self.anomaly_detectors = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_cooldown = {}  # Prevent spam
        
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        
        # Initialize anomaly detector if needed
        if rule.rule_type == 'anomaly':
            detector_key = f"{rule.model_filter}_{rule.metric_name}"
            if detector_key not in self.anomaly_detectors:
                self.anomaly_detectors[detector_key] = StatisticalAnomalyDetector(
                    threshold=1.0 / rule.sensitivity
                )
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
    
    def add_notifier(self, notifier: AlertNotifier):
        """Add an alert notifier."""
        self.notifiers.append(notifier)
    
    def update_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Update model performance metrics."""
        for metric_name, value in metrics.items():
            # Update performance degradation detector
            self.performance_detector.update_performance(model_name, metric_name, value)
            
            # Update anomaly detectors
            detector_key = f"{model_name}_{metric_name}"
            if detector_key in self.anomaly_detectors:
                self.anomaly_detectors[detector_key].update(value)
            
            # Check alert rules
            self._check_metric_rules(model_name, metric_name, value)
    
    def check_data_quality(self, data: pd.DataFrame) -> List[Alert]:
        """Check data quality and generate alerts."""
        issues = self.data_quality_monitor.check_data_quality(data)
        alerts = []
        
        for issue in issues:
            alert = Alert(
                alert_id=f"dq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{issue['type']}",
                alert_type='data_quality',
                severity=issue['severity'],
                message=f"Data quality issue detected: {issue['type']} in column {issue['column']}",
                timestamp=datetime.now(),
                additional_data=issue
            )
            alerts.append(alert)
            self._send_alert(alert)
        
        return alerts
    
    def check_performance_degradation(self) -> List[Alert]:
        """Check for performance degradation and generate alerts."""
        alerts = []
        
        # Check all models and metrics
        for key in self.performance_detector.baseline_stats.keys():
            model_name, metric_name = key.rsplit('_', 1)
            
            is_degraded, info = self.performance_detector.check_degradation(
                model_name, metric_name
            )
            
            if is_degraded:
                alert = Alert(
                    alert_id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{metric_name}",
                    alert_type='performance_degradation',
                    severity='high' if info['relative_change'] > 0.2 else 'medium',
                    message=f"Performance degradation detected for {model_name} on {metric_name}",
                    timestamp=datetime.now(),
                    model_name=model_name,
                    metric_name=metric_name,
                    current_value=info['recent_mean'],
                    threshold_value=info['baseline_mean'],
                    additional_data=info
                )
                alerts.append(alert)
                self._send_alert(alert)
        
        return alerts
    
    def _check_metric_rules(self, model_name: str, metric_name: str, value: float):
        """Check metric against alert rules."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
                
            # Check model filter
            if rule.model_filter and rule.model_filter != model_name:
                continue
                
            # Check metric name
            if rule.metric_name != metric_name:
                continue
            
            # Check rule type
            if rule.rule_type == 'threshold':
                self._check_threshold_rule(rule, model_name, metric_name, value)
            elif rule.rule_type == 'anomaly':
                self._check_anomaly_rule(rule, model_name, metric_name, value)
    
    def _check_threshold_rule(self, rule: AlertRule, model_name: str, 
                            metric_name: str, value: float):
        """Check threshold-based rule."""
        triggered = False
        
        if rule.comparison == 'greater' and value > rule.threshold:
            triggered = True
        elif rule.comparison == 'less' and value < rule.threshold:
            triggered = True
        elif rule.comparison == 'equal' and abs(value - rule.threshold) < 1e-6:
            triggered = True
        
        if triggered:
            # Check cooldown
            cooldown_key = f"{rule.rule_id}_{model_name}_{metric_name}"
            if self._is_in_cooldown(cooldown_key):
                return
            
            alert = Alert(
                alert_id=f"thresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.rule_id}",
                alert_type='threshold',
                severity=rule.severity,
                message=f"Threshold rule '{rule.rule_id}' triggered",
                timestamp=datetime.now(),
                model_name=model_name,
                metric_name=metric_name,
                current_value=value,
                threshold_value=rule.threshold
            )
            
            self._send_alert(alert)
            self._set_cooldown(cooldown_key)
    
    def _check_anomaly_rule(self, rule: AlertRule, model_name: str, 
                          metric_name: str, value: float):
        """Check anomaly-based rule."""
        detector_key = f"{model_name}_{metric_name}"
        
        if detector_key in self.anomaly_detectors:
            is_anomaly, score = self.anomaly_detectors[detector_key].detect(value)
            
            if is_anomaly:
                # Check cooldown
                cooldown_key = f"{rule.rule_id}_{model_name}_{metric_name}"
                if self._is_in_cooldown(cooldown_key):
                    return
                
                alert = Alert(
                    alert_id=f"anom_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.rule_id}",
                    alert_type='anomaly',
                    severity=rule.severity,
                    message=f"Anomaly detected in {metric_name} for {model_name}",
                    timestamp=datetime.now(),
                    model_name=model_name,
                    metric_name=metric_name,
                    current_value=value,
                    additional_data={'anomaly_score': score}
                )
                
                self._send_alert(alert)
                self._set_cooldown(cooldown_key)
    
    def _send_alert(self, alert: Alert):
        """Send alert through all notifiers."""
        self.alert_history.append(alert)
        
        for notifier in self.notifiers:
            try:
                notifier.send_alert(alert)
            except Exception as e:
                logging.error(f"Failed to send alert via {type(notifier).__name__}: {e}")
    
    def _is_in_cooldown(self, key: str, cooldown_minutes: int = 30) -> bool:
        """Check if alert is in cooldown period."""
        if key in self.alert_cooldown:
            time_diff = datetime.now() - self.alert_cooldown[key]
            return time_diff < timedelta(minutes=cooldown_minutes)
        return False
    
    def _set_cooldown(self, key: str):
        """Set cooldown for alert."""
        self.alert_cooldown[key] = datetime.now()
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts."""
        recent_alerts = self.get_alert_history(24)
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': defaultdict(int),
            'by_type': defaultdict(int),
            'by_model': defaultdict(int)
        }
        
        for alert in recent_alerts:
            summary['by_severity'][alert.severity] += 1
            summary['by_type'][alert.alert_type] += 1
            if alert.model_name:
                summary['by_model'][alert.model_name] += 1
        
        return dict(summary)


# Utility functions
def create_alerting_system(config: Dict) -> AlertingSystem:
    """
    Create alerting system from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured alerting system
    """
    system = AlertingSystem()
    
    # Add notifiers
    if 'email' in config:
        email_config = config['email']
        notifier = EmailNotifier(
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            username=email_config['username'],
            password=email_config['password'],
            from_email=email_config['from_email'],
            to_emails=email_config['to_emails']
        )
        system.add_notifier(notifier)
    
    if 'webhook' in config:
        webhook_config = config['webhook']
        notifier = WebhookNotifier(
            webhook_url=webhook_config['url'],
            headers=webhook_config.get('headers')
        )
        system.add_notifier(notifier)
    
    # Always add log notifier
    system.add_notifier(LogNotifier())
    
    # Add alert rules
    if 'rules' in config:
        for rule_config in config['rules']:
            rule = AlertRule(**rule_config)
            system.add_alert_rule(rule)
    
    return system