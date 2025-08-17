"""
Automated alerting and anomaly detection system for time series prediction models.
Monitors model performance, data quality, and system health with configurable alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import threading
import time
import json
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    smtplib = None
    MimeText = None
    MimeMultipart = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None
from collections import deque, defaultdict
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    MODEL_FAILURE = "model_failure"
    ANOMALY_DETECTION = "anomaly_detection"
    THRESHOLD_BREACH = "threshold_breach"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    time_window_minutes: int
    min_occurrences: int = 1
    enabled: bool = True
    description: str = ""
    
    def evaluate(self, metric_value: float) -> bool:
        """Evaluate if the rule should trigger an alert."""
        if not self.enabled:
            return False
        
        if self.comparison_operator == '>':
            return metric_value > self.threshold_value
        elif self.comparison_operator == '<':
            return metric_value < self.threshold_value
        elif self.comparison_operator == '>=':
            return metric_value >= self.threshold_value
        elif self.comparison_operator == '<=':
            return metric_value <= self.threshold_value
        elif self.comparison_operator == '==':
            return abs(metric_value - self.threshold_value) < 1e-6
        elif self.comparison_operator == '!=':
            return abs(metric_value - self.threshold_value) >= 1e-6
        else:
            return False


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold_value: float
    timestamp: datetime
    model_id: Optional[str] = None
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'resolved': self.resolved,
            'resolved_timestamp': self.resolved_timestamp.isoformat() if self.resolved_timestamp else None
        }


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    method: str  # 'statistical', 'isolation_forest', 'local_outlier_factor'
    sensitivity: float = 0.95  # Confidence level for statistical methods
    window_size: int = 100  # Number of recent observations to consider
    min_samples: int = 30  # Minimum samples needed for detection
    contamination: float = 0.1  # Expected proportion of outliers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'sensitivity': self.sensitivity,
            'window_size': self.window_size,
            'min_samples': self.min_samples,
            'contamination': self.contamination
        }


class AnomalyDetector:
    """Anomaly detection for model performance and data quality."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        """
        Initialize anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.historical_data = defaultdict(lambda: deque(maxlen=config.window_size))
        
        # Initialize detection models
        self._init_detection_models()
    
    def _init_detection_models(self):
        """Initialize anomaly detection models."""
        if self.config.method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                self.isolation_forest = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42
                )
            except ImportError:
                logger.warning("scikit-learn not available, falling back to statistical method")
                self.config.method = 'statistical'
        
        elif self.config.method == 'local_outlier_factor':
            try:
                from sklearn.neighbors import LocalOutlierFactor
                self.lof = LocalOutlierFactor(
                    contamination=self.config.contamination,
                    novelty=True
                )
            except ImportError:
                logger.warning("scikit-learn not available, falling back to statistical method")
                self.config.method = 'statistical'
    
    def add_observation(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Add a new observation for anomaly detection.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Observation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.historical_data[metric_name].append((timestamp, value))
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> Tuple[bool, float]:
        """
        Detect if current value is anomalous.
        
        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if metric_name not in self.historical_data:
            return False, 0.0
        
        historical_values = [value for _, value in self.historical_data[metric_name]]
        
        if len(historical_values) < self.config.min_samples:
            return False, 0.0
        
        if self.config.method == 'statistical':
            return self._detect_statistical_anomaly(historical_values, current_value)
        elif self.config.method == 'isolation_forest':
            return self._detect_isolation_forest_anomaly(historical_values, current_value)
        elif self.config.method == 'local_outlier_factor':
            return self._detect_lof_anomaly(historical_values, current_value)
        else:
            return False, 0.0
    
    def _detect_statistical_anomaly(self, historical_values: List[float], current_value: float) -> Tuple[bool, float]:
        """Detect anomaly using statistical methods (z-score)."""
        if len(historical_values) < 2:
            return False, 0.0
        
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        if std_val == 0:
            return False, 0.0
        
        z_score = abs((current_value - mean_val) / std_val)
        
        # Use confidence level to determine threshold
        from scipy import stats
        threshold = stats.norm.ppf((1 + self.config.sensitivity) / 2)
        
        is_anomaly = z_score > threshold
        anomaly_score = min(z_score / threshold, 1.0) if threshold > 0 else 0.0
        
        return is_anomaly, anomaly_score
    
    def _detect_isolation_forest_anomaly(self, historical_values: List[float], current_value: float) -> Tuple[bool, float]:
        """Detect anomaly using Isolation Forest."""
        try:
            # Prepare data
            X = np.array(historical_values).reshape(-1, 1)
            
            # Fit model
            self.isolation_forest.fit(X)
            
            # Predict anomaly
            prediction = self.isolation_forest.predict([[current_value]])
            anomaly_score = self.isolation_forest.decision_function([[current_value]])[0]
            
            # Convert to 0-1 scale (lower scores indicate anomalies)
            normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            
            is_anomaly = prediction[0] == -1
            
            return is_anomaly, 1.0 - normalized_score if is_anomaly else 0.0
            
        except Exception as e:
            logger.error(f"Isolation Forest anomaly detection failed: {e}")
            return self._detect_statistical_anomaly(historical_values, current_value)
    
    def _detect_lof_anomaly(self, historical_values: List[float], current_value: float) -> Tuple[bool, float]:
        """Detect anomaly using Local Outlier Factor."""
        try:
            # Prepare data
            X = np.array(historical_values).reshape(-1, 1)
            
            # Fit model
            self.lof.fit(X)
            
            # Predict anomaly
            prediction = self.lof.predict([[current_value]])
            anomaly_score = self.lof.decision_function([[current_value]])[0]
            
            # Convert to 0-1 scale
            normalized_score = max(0, min(1, abs(anomaly_score)))
            
            is_anomaly = prediction[0] == -1
            
            return is_anomaly, normalized_score if is_anomaly else 0.0
            
        except Exception as e:
            logger.error(f"LOF anomaly detection failed: {e}")
            return self._detect_statistical_anomaly(historical_values, current_value)
    
    def get_anomaly_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get anomaly detection summary for a metric."""
        if metric_name not in self.historical_data:
            return {}
        
        historical_values = [value for _, value in self.historical_data[metric_name]]
        
        if len(historical_values) < self.config.min_samples:
            return {'status': 'insufficient_data', 'sample_count': len(historical_values)}
        
        # Calculate basic statistics
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        min_val = np.min(historical_values)
        max_val = np.max(historical_values)
        
        return {
            'status': 'active',
            'sample_count': len(historical_values),
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'method': self.config.method,
            'sensitivity': self.config.sensitivity
        }


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, channel_id: str, enabled: bool = True):
        self.channel_id = channel_id
        self.enabled = enabled
    
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, channel_id: str, smtp_server: str, smtp_port: int,
                 username: str, password: str, from_email: str, to_emails: List[str],
                 enabled: bool = True):
        super().__init__(channel_id, enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.enabled or not EMAIL_AVAILABLE:
            if not EMAIL_AVAILABLE:
                logger.warning("Email functionality not available - missing email modules")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
            Alert Details:
            
            Alert ID: {alert.alert_id}
            Severity: {alert.severity.value.upper()}
            Type: {alert.alert_type.value}
            Timestamp: {alert.timestamp}
            
            Message: {alert.message}
            
            Metric: {alert.metric_name}
            Current Value: {alert.metric_value}
            Threshold: {alert.threshold_value}
            
            Model ID: {alert.model_id or 'N/A'}
            
            This is an automated alert from the Time Series Monitoring System.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(self, channel_id: str, webhook_url: str, headers: Optional[Dict[str, str]] = None,
                 enabled: bool = True):
        super().__init__(channel_id, enabled)
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.enabled or not REQUESTS_AVAILABLE:
            if not REQUESTS_AVAILABLE:
                logger.warning("Webhook functionality not available - requests module not installed")
            return False
        
        try:
            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'timestamp': alert.timestamp.isoformat(),
                'model_id': alert.model_id
            }
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, channel_id: str, webhook_url: str, channel: str = "#alerts",
                 enabled: bool = True):
        super().__init__(channel_id, enabled)
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.enabled or not REQUESTS_AVAILABLE:
            if not REQUESTS_AVAILABLE:
                logger.warning("Slack functionality not available - requests module not installed")
            return False
        
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8B0000"  # Dark Red
            }
            
            color = color_map.get(alert.severity, "#36a64f")
            
            # Prepare Slack payload
            payload = {
                "channel": self.channel,
                "username": "Time Series Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Type",
                                "value": alert.alert_type.value,
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": alert.metric_name,
                                "short": True
                            },
                            {
                                "title": "Value",
                                "value": f"{alert.metric_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Model",
                                "value": alert.model_id or "N/A",
                                "short": True
                            }
                        ],
                        "timestamp": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"Slack webhook failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertingEngine:
    """Main alerting engine that evaluates rules and sends notifications."""
    
    def __init__(self):
        """Initialize alerting engine."""
        self.alert_rules = {}
        self.notification_channels = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        
        # Rule evaluation tracking
        self.rule_violations = defaultdict(lambda: deque(maxlen=100))
        
        # Anomaly detection
        self.anomaly_detectors = {}
        
        # Threading
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = None
        
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.channel_id}")
    
    def remove_notification_channel(self, channel_id: str):
        """Remove a notification channel."""
        if channel_id in self.notification_channels:
            del self.notification_channels[channel_id]
            logger.info(f"Removed notification channel: {channel_id}")
    
    def add_anomaly_detector(self, metric_name: str, config: AnomalyDetectionConfig):
        """Add anomaly detector for a metric."""
        self.anomaly_detectors[metric_name] = AnomalyDetector(config)
        logger.info(f"Added anomaly detector for metric: {metric_name}")
    
    def evaluate_metric(self, metric_name: str, metric_value: float, 
                       model_id: Optional[str] = None, timestamp: Optional[datetime] = None):
        """
        Evaluate a metric against all applicable rules.
        
        Args:
            metric_name: Name of the metric
            metric_value: Current metric value
            model_id: Optional model identifier
            timestamp: Metric timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Evaluate alert rules
        for rule in self.alert_rules.values():
            if rule.metric_name == metric_name and rule.enabled:
                self._evaluate_rule(rule, metric_value, model_id, timestamp)
        
        # Evaluate anomaly detection
        if metric_name in self.anomaly_detectors:
            self._evaluate_anomaly_detection(metric_name, metric_value, model_id, timestamp)
    
    def _evaluate_rule(self, rule: AlertRule, metric_value: float, 
                      model_id: Optional[str], timestamp: datetime):
        """Evaluate a single rule."""
        violation_occurred = rule.evaluate(metric_value)
        
        if violation_occurred:
            # Record violation
            self.rule_violations[rule.rule_id].append(timestamp)
            
            # Check if we have enough violations in the time window
            cutoff_time = timestamp - timedelta(minutes=rule.time_window_minutes)
            recent_violations = [
                t for t in self.rule_violations[rule.rule_id] 
                if t >= cutoff_time
            ]
            
            if len(recent_violations) >= rule.min_occurrences:
                # Check if alert is already active
                alert_key = f"{rule.rule_id}_{model_id or 'global'}"
                
                if alert_key not in self.active_alerts:
                    # Create new alert
                    alert = self._create_alert(rule, metric_value, model_id, timestamp)
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    
                    # Send notifications
                    self._send_alert_notifications(alert)
        else:
            # Check if we should resolve an active alert
            alert_key = f"{rule.rule_id}_{model_id or 'global'}"
            
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolved_timestamp = timestamp
                
                del self.active_alerts[alert_key]
                
                logger.info(f"Alert resolved: {alert.alert_id}")
    
    def _evaluate_anomaly_detection(self, metric_name: str, metric_value: float,
                                  model_id: Optional[str], timestamp: datetime):
        """Evaluate anomaly detection for a metric."""
        detector = self.anomaly_detectors[metric_name]
        
        # Add observation to detector
        detector.add_observation(metric_name, metric_value, timestamp)
        
        # Check for anomaly
        is_anomaly, anomaly_score = detector.detect_anomaly(metric_name, metric_value)
        
        if is_anomaly:
            # Create anomaly alert
            alert_id = f"anomaly_{metric_name}_{model_id or 'global'}_{int(timestamp.timestamp())}"
            
            alert = Alert(
                alert_id=alert_id,
                rule_id=f"anomaly_{metric_name}",
                alert_type=AlertType.ANOMALY_DETECTION,
                severity=AlertSeverity.MEDIUM if anomaly_score < 0.8 else AlertSeverity.HIGH,
                title=f"Anomaly Detected in {metric_name}",
                message=f"Anomalous value detected for {metric_name}: {metric_value:.4f} (score: {anomaly_score:.3f})",
                metric_name=metric_name,
                metric_value=metric_value,
                threshold_value=anomaly_score,
                timestamp=timestamp,
                model_id=model_id
            )
            
            self.alert_history.append(alert)
            
            # Send notifications for high-severity anomalies
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                self._send_alert_notifications(alert)
    
    def _create_alert(self, rule: AlertRule, metric_value: float, 
                     model_id: Optional[str], timestamp: datetime) -> Alert:
        """Create an alert from a rule violation."""
        alert_id = f"{rule.rule_id}_{model_id or 'global'}_{int(timestamp.timestamp())}"
        
        # Generate alert message
        message = (f"{rule.description or rule.name}: "
                  f"{rule.metric_name} = {metric_value:.4f} "
                  f"{rule.comparison_operator} {rule.threshold_value:.4f}")
        
        if model_id:
            message += f" (Model: {model_id})"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.name,
            message=message,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold_value=rule.threshold_value,
            timestamp=timestamp,
            model_id=model_id
        )
        
        logger.warning(f"Alert triggered: {alert.title} - {alert.message}")
        
        return alert
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through all enabled channels."""
        for channel in self.notification_channels.values():
            if channel.enabled:
                try:
                    success = channel.send_notification(alert)
                    if success:
                        logger.info(f"Alert notification sent via {channel.channel_id}")
                    else:
                        logger.error(f"Failed to send alert via {channel.channel_id}")
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.channel_id}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in self.alert_history:
            type_counts[alert.alert_type.value] += 1
        
        # Recent alert rate (last 24 hours)
        recent_alerts = self.get_alert_history(24)
        alert_rate_24h = len(recent_alerts)
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'alert_rate_24h': alert_rate_24h,
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'total_rules': len(self.alert_rules),
            'notification_channels': len(self.notification_channels)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export alerting configuration."""
        return {
            'alert_rules': [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'alert_type': rule.alert_type.value,
                    'severity': rule.severity.value,
                    'metric_name': rule.metric_name,
                    'threshold_value': rule.threshold_value,
                    'comparison_operator': rule.comparison_operator,
                    'time_window_minutes': rule.time_window_minutes,
                    'min_occurrences': rule.min_occurrences,
                    'enabled': rule.enabled,
                    'description': rule.description
                }
                for rule in self.alert_rules.values()
            ],
            'anomaly_detectors': {
                metric_name: detector.config.to_dict()
                for metric_name, detector in self.anomaly_detectors.items()
            },
            'notification_channels': [
                {
                    'channel_id': channel.channel_id,
                    'type': type(channel).__name__,
                    'enabled': channel.enabled
                }
                for channel in self.notification_channels.values()
            ]
        }


class AlertingSystem:
    """
    Main alerting system that integrates with performance monitoring.
    """
    
    def __init__(self):
        """Initialize alerting system."""
        self.engine = AlertingEngine()
        self.is_running = False
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules for common scenarios."""
        
        # Performance degradation rules
        self.engine.add_alert_rule(AlertRule(
            rule_id="high_mae",
            name="High Mean Absolute Error",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            metric_name="mae",
            threshold_value=0.1,
            comparison_operator=">",
            time_window_minutes=30,
            min_occurrences=3,
            description="Model MAE exceeds acceptable threshold"
        ))
        
        self.engine.add_alert_rule(AlertRule(
            rule_id="low_accuracy",
            name="Low Directional Accuracy",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.MEDIUM,
            metric_name="directional_accuracy",
            threshold_value=50.0,
            comparison_operator="<",
            time_window_minutes=60,
            min_occurrences=2,
            description="Model directional accuracy below 50%"
        ))
        
        # System health rules
        self.engine.add_alert_rule(AlertRule(
            rule_id="high_cpu",
            name="High CPU Usage",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.MEDIUM,
            metric_name="cpu_usage",
            threshold_value=90.0,
            comparison_operator=">",
            time_window_minutes=15,
            min_occurrences=3,
            description="CPU usage consistently above 90%"
        ))
        
        self.engine.add_alert_rule(AlertRule(
            rule_id="high_memory",
            name="High Memory Usage",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            metric_name="memory_usage",
            threshold_value=95.0,
            comparison_operator=">",
            time_window_minutes=10,
            min_occurrences=2,
            description="Memory usage critically high"
        ))
        
        self.engine.add_alert_rule(AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            alert_type=AlertType.MODEL_FAILURE,
            severity=AlertSeverity.CRITICAL,
            metric_name="error_rate",
            threshold_value=10.0,
            comparison_operator=">",
            time_window_minutes=20,
            min_occurrences=1,
            description="Error rate exceeds 10%"
        ))
    
    def add_email_notifications(self, smtp_server: str, smtp_port: int,
                              username: str, password: str, from_email: str, to_emails: List[str]):
        """Add email notification channel."""
        channel = EmailNotificationChannel(
            channel_id="email",
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            from_email=from_email,
            to_emails=to_emails
        )
        self.engine.add_notification_channel(channel)
    
    def add_webhook_notifications(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """Add webhook notification channel."""
        channel = WebhookNotificationChannel(
            channel_id="webhook",
            webhook_url=webhook_url,
            headers=headers
        )
        self.engine.add_notification_channel(channel)
    
    def add_slack_notifications(self, webhook_url: str, channel: str = "#alerts"):
        """Add Slack notification channel."""
        channel = SlackNotificationChannel(
            channel_id="slack",
            webhook_url=webhook_url,
            channel=channel
        )
        self.engine.add_notification_channel(channel)
    
    def add_custom_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.engine.add_alert_rule(rule)
    
    def enable_anomaly_detection(self, metric_name: str, method: str = 'statistical',
                                sensitivity: float = 0.95, window_size: int = 100):
        """Enable anomaly detection for a metric."""
        config = AnomalyDetectionConfig(
            method=method,
            sensitivity=sensitivity,
            window_size=window_size
        )
        self.engine.add_anomaly_detector(metric_name, config)
    
    def evaluate_metrics(self, metrics: Dict[str, float], model_id: Optional[str] = None):
        """Evaluate multiple metrics at once."""
        timestamp = datetime.now()
        
        for metric_name, metric_value in metrics.items():
            self.engine.evaluate_metric(metric_name, metric_value, model_id, timestamp)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        alerts = self.engine.get_active_alerts()
        return [alert.to_dict() for alert in alerts]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history."""
        alerts = self.engine.get_alert_history(hours)
        return [alert.to_dict() for alert in alerts]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return self.engine.get_alert_statistics()
    
    def export_configuration(self, filepath: str):
        """Export alerting configuration to file."""
        config = self.engine.export_configuration()
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Alerting configuration exported to {filepath}")
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification channels."""
        test_alert = Alert(
            alert_id="test_alert",
            rule_id="test_rule",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.LOW,
            title="Test Alert",
            message="This is a test alert to verify notification channels are working.",
            metric_name="test_metric",
            metric_value=1.0,
            threshold_value=0.5,
            timestamp=datetime.now()
        )
        
        results = {}
        
        for channel_id, channel in self.engine.notification_channels.items():
            try:
                success = channel.send_notification(test_alert)
                results[channel_id] = success
            except Exception as e:
                logger.error(f"Test notification failed for {channel_id}: {e}")
                results[channel_id] = False
        
        return results