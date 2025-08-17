"""
Adaptive model switching system based on performance monitoring.
Automatically switches between models based on real-time performance metrics and conditions.
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
from collections import deque, defaultdict
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SwitchingTrigger(Enum):
    """Types of triggers for model switching."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RELATIVE_PERFORMANCE = "relative_performance"
    TIME_BASED = "time_based"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"
    MARKET_REGIME_CHANGE = "market_regime_change"
    MANUAL = "manual"


class SwitchingStrategy(Enum):
    """Strategies for model switching."""
    BEST_PERFORMER = "best_performer"
    WEIGHTED_ENSEMBLE = "weighted_ensemble"
    ROUND_ROBIN = "round_robin"
    CONFIDENCE_BASED = "confidence_based"
    REGIME_SPECIFIC = "regime_specific"


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time."""
    model_id: str
    timestamp: datetime
    mae: float
    rmse: float
    directional_accuracy: float
    sharpe_ratio: float
    confidence_score: float
    prediction_count: int
    recent_predictions: List[float] = field(default_factory=list)
    recent_actuals: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'mae': self.mae,
            'rmse': self.rmse,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'confidence_score': self.confidence_score,
            'prediction_count': self.prediction_count,
            'recent_predictions': self.recent_predictions,
            'recent_actuals': self.recent_actuals
        }


@dataclass
class SwitchingRule:
    """Rule for automatic model switching."""
    rule_id: str
    name: str
    trigger: SwitchingTrigger
    strategy: SwitchingStrategy
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<='
    evaluation_window_minutes: int
    min_observations: int
    cooldown_minutes: int = 30  # Minimum time between switches
    enabled: bool = True
    priority: int = 1  # Higher priority rules evaluated first
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, current_performance: Dict[str, ModelPerformanceSnapshot],
                active_model_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if the rule should trigger a model switch.
        
        Args:
            current_performance: Current performance snapshots for all models
            active_model_id: Currently active model ID
            
        Returns:
            Tuple of (should_switch, recommended_model_id)
        """
        if not self.enabled:
            return False, None
        
        if active_model_id not in current_performance:
            return False, None
        
        active_model_perf = current_performance[active_model_id]
        
        # Get metric value for active model
        metric_value = getattr(active_model_perf, self.metric_name, None)
        if metric_value is None:
            return False, None
        
        # Check if threshold is breached
        threshold_breached = self._check_threshold(metric_value)
        
        if not threshold_breached:
            return False, None
        
        # Determine recommended model based on strategy
        recommended_model = self._get_recommended_model(
            current_performance, active_model_id
        )
        
        return True, recommended_model
    
    def _check_threshold(self, metric_value: float) -> bool:
        """Check if metric value breaches threshold."""
        if self.comparison_operator == '>':
            return metric_value > self.threshold_value
        elif self.comparison_operator == '<':
            return metric_value < self.threshold_value
        elif self.comparison_operator == '>=':
            return metric_value >= self.threshold_value
        elif self.comparison_operator == '<=':
            return metric_value <= self.threshold_value
        else:
            return False
    
    def _get_recommended_model(self, performance_data: Dict[str, ModelPerformanceSnapshot],
                             active_model_id: str) -> Optional[str]:
        """Get recommended model based on strategy."""
        if self.strategy == SwitchingStrategy.BEST_PERFORMER:
            return self._get_best_performer(performance_data, active_model_id)
        elif self.strategy == SwitchingStrategy.CONFIDENCE_BASED:
            return self._get_highest_confidence_model(performance_data, active_model_id)
        else:
            # Default to best performer
            return self._get_best_performer(performance_data, active_model_id)
    
    def _get_best_performer(self, performance_data: Dict[str, ModelPerformanceSnapshot],
                          active_model_id: str) -> Optional[str]:
        """Get the best performing model based on the metric."""
        best_model = None
        best_value = None
        
        # Determine if lower is better for this metric
        lower_is_better = self.metric_name in ['mae', 'rmse']
        
        for model_id, perf in performance_data.items():
            if model_id == active_model_id:
                continue
            
            metric_value = getattr(perf, self.metric_name, None)
            if metric_value is None:
                continue
            
            if best_value is None:
                best_model = model_id
                best_value = metric_value
            else:
                if lower_is_better and metric_value < best_value:
                    best_model = model_id
                    best_value = metric_value
                elif not lower_is_better and metric_value > best_value:
                    best_model = model_id
                    best_value = metric_value
        
        return best_model
    
    def _get_highest_confidence_model(self, performance_data: Dict[str, ModelPerformanceSnapshot],
                                    active_model_id: str) -> Optional[str]:
        """Get the model with highest confidence score."""
        best_model = None
        best_confidence = 0.0
        
        for model_id, perf in performance_data.items():
            if model_id == active_model_id:
                continue
            
            if perf.confidence_score > best_confidence:
                best_model = model_id
                best_confidence = perf.confidence_score
        
        return best_model


@dataclass
class ModelSwitchEvent:
    """Record of a model switching event."""
    event_id: str
    timestamp: datetime
    trigger: SwitchingTrigger
    rule_id: str
    from_model_id: str
    to_model_id: str
    trigger_metric: str
    trigger_value: float
    threshold_value: float
    performance_improvement: Optional[float] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'trigger': self.trigger.value,
            'rule_id': self.rule_id,
            'from_model_id': self.from_model_id,
            'to_model_id': self.to_model_id,
            'trigger_metric': self.trigger_metric,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'performance_improvement': self.performance_improvement,
            'success': self.success
        }


class PerformanceEvaluator:
    """Evaluates model performance for switching decisions."""
    
    def __init__(self, evaluation_window: int = 100):
        """
        Initialize performance evaluator.
        
        Args:
            evaluation_window: Number of recent predictions to consider
        """
        self.evaluation_window = evaluation_window
        self.model_predictions = defaultdict(lambda: deque(maxlen=evaluation_window))
        self.model_actuals = defaultdict(lambda: deque(maxlen=evaluation_window))
        self.model_confidences = defaultdict(lambda: deque(maxlen=evaluation_window))
        self.model_timestamps = defaultdict(lambda: deque(maxlen=evaluation_window))
    
    def add_prediction(self, model_id: str, prediction: float, actual: Optional[float] = None,
                      confidence: float = 1.0, timestamp: Optional[datetime] = None):
        """
        Add a prediction for performance evaluation.
        
        Args:
            model_id: Model identifier
            prediction: Predicted value
            actual: Actual value (if available)
            confidence: Prediction confidence
            timestamp: Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.model_predictions[model_id].append(prediction)
        self.model_confidences[model_id].append(confidence)
        self.model_timestamps[model_id].append(timestamp)
        
        if actual is not None:
            self.model_actuals[model_id].append(actual)
    
    def get_performance_snapshot(self, model_id: str) -> Optional[ModelPerformanceSnapshot]:
        """
        Get current performance snapshot for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Performance snapshot or None if insufficient data
        """
        if (model_id not in self.model_predictions or 
            len(self.model_predictions[model_id]) == 0):
            return None
        
        predictions = list(self.model_predictions[model_id])
        actuals = list(self.model_actuals[model_id])
        confidences = list(self.model_confidences[model_id])
        timestamps = list(self.model_timestamps[model_id])
        
        # Align predictions and actuals
        min_len = min(len(predictions), len(actuals))
        if min_len < 5:  # Need minimum observations
            return ModelPerformanceSnapshot(
                model_id=model_id,
                timestamp=datetime.now(),
                mae=float('inf'),
                rmse=float('inf'),
                directional_accuracy=0.0,
                sharpe_ratio=0.0,
                confidence_score=np.mean(confidences) if confidences else 0.0,
                prediction_count=len(predictions),
                recent_predictions=predictions[-10:],
                recent_actuals=actuals[-10:]
            )
        
        aligned_predictions = np.array(predictions[-min_len:])
        aligned_actuals = np.array(actuals[-min_len:])
        
        # Calculate performance metrics
        mae = float(np.mean(np.abs(aligned_predictions - aligned_actuals)))
        rmse = float(np.sqrt(np.mean((aligned_predictions - aligned_actuals) ** 2)))
        
        # Directional accuracy
        if len(aligned_predictions) > 1:
            pred_directions = np.sign(np.diff(aligned_predictions))
            actual_directions = np.sign(np.diff(aligned_actuals))
            directional_accuracy = float(np.mean(pred_directions == actual_directions) * 100)
        else:
            directional_accuracy = 0.0
        
        # Sharpe ratio (assuming predictions are returns)
        if np.std(aligned_predictions) > 0:
            sharpe_ratio = float(np.mean(aligned_predictions) / np.std(aligned_predictions))
        else:
            sharpe_ratio = 0.0
        
        # Average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        return ModelPerformanceSnapshot(
            model_id=model_id,
            timestamp=timestamps[-1] if timestamps else datetime.now(),
            mae=mae,
            rmse=rmse,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            confidence_score=avg_confidence,
            prediction_count=len(predictions),
            recent_predictions=predictions[-10:],
            recent_actuals=actuals[-10:]
        )
    
    def get_all_performance_snapshots(self) -> Dict[str, ModelPerformanceSnapshot]:
        """Get performance snapshots for all models."""
        snapshots = {}
        
        for model_id in self.model_predictions.keys():
            snapshot = self.get_performance_snapshot(model_id)
            if snapshot:
                snapshots[model_id] = snapshot
        
        return snapshots
    
    def compare_models(self, model1_id: str, model2_id: str, 
                      metric: str = 'mae') -> Optional[float]:
        """
        Compare two models on a specific metric.
        
        Args:
            model1_id: First model ID
            model2_id: Second model ID
            metric: Metric to compare ('mae', 'rmse', 'directional_accuracy', etc.)
            
        Returns:
            Difference in metric (model1 - model2), or None if comparison not possible
        """
        snapshot1 = self.get_performance_snapshot(model1_id)
        snapshot2 = self.get_performance_snapshot(model2_id)
        
        if not snapshot1 or not snapshot2:
            return None
        
        value1 = getattr(snapshot1, metric, None)
        value2 = getattr(snapshot2, metric, None)
        
        if value1 is None or value2 is None:
            return None
        
        return float(value1 - value2)


class ModelSwitchingEngine:
    """Core engine for adaptive model switching."""
    
    def __init__(self, cooldown_minutes: int = 30):
        """
        Initialize model switching engine.
        
        Args:
            cooldown_minutes: Minimum time between model switches
        """
        self.cooldown_minutes = cooldown_minutes
        self.switching_rules = {}
        self.performance_evaluator = PerformanceEvaluator()
        
        # State tracking
        self.active_model_id = None
        self.last_switch_time = None
        self.switch_history = deque(maxlen=1000)
        
        # Model registry
        self.available_models = {}
        self.model_metadata = {}
        
        # Threading
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = None
        
    def register_model(self, model_id: str, model_instance: Any, 
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Register a model for switching.
        
        Args:
            model_id: Unique model identifier
            model_instance: Model instance or callable
            metadata: Optional metadata about the model
        """
        self.available_models[model_id] = model_instance
        self.model_metadata[model_id] = metadata or {}
        
        logger.info(f"Registered model: {model_id}")
        
        # Set as active if it's the first model
        if self.active_model_id is None:
            self.active_model_id = model_id
            logger.info(f"Set {model_id} as active model")
    
    def unregister_model(self, model_id: str):
        """Unregister a model."""
        if model_id in self.available_models:
            del self.available_models[model_id]
        
        if model_id in self.model_metadata:
            del self.model_metadata[model_id]
        
        # Switch to another model if this was active
        if self.active_model_id == model_id:
            remaining_models = list(self.available_models.keys())
            if remaining_models:
                self.active_model_id = remaining_models[0]
                logger.info(f"Switched to {self.active_model_id} after unregistering {model_id}")
            else:
                self.active_model_id = None
        
        logger.info(f"Unregistered model: {model_id}")
    
    def add_switching_rule(self, rule: SwitchingRule):
        """Add a switching rule."""
        self.switching_rules[rule.rule_id] = rule
        logger.info(f"Added switching rule: {rule.name}")
    
    def remove_switching_rule(self, rule_id: str):
        """Remove a switching rule."""
        if rule_id in self.switching_rules:
            del self.switching_rules[rule_id]
            logger.info(f"Removed switching rule: {rule_id}")
    
    def add_prediction(self, model_id: str, prediction: float, actual: Optional[float] = None,
                      confidence: float = 1.0, timestamp: Optional[datetime] = None):
        """
        Add a prediction and evaluate switching rules.
        
        Args:
            model_id: Model that made the prediction
            prediction: Predicted value
            actual: Actual value (if available)
            confidence: Prediction confidence
            timestamp: Prediction timestamp
        """
        # Add to performance evaluator
        self.performance_evaluator.add_prediction(
            model_id, prediction, actual, confidence, timestamp
        )
        
        # Evaluate switching rules if this is the active model
        if model_id == self.active_model_id:
            self._evaluate_switching_rules()
    
    def _evaluate_switching_rules(self):
        """Evaluate all switching rules and potentially switch models."""
        if not self.active_model_id:
            return
        
        # Check cooldown period
        if (self.last_switch_time and 
            datetime.now() - self.last_switch_time < timedelta(minutes=self.cooldown_minutes)):
            return
        
        # Get current performance snapshots
        performance_snapshots = self.performance_evaluator.get_all_performance_snapshots()
        
        if self.active_model_id not in performance_snapshots:
            return
        
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(
            self.switching_rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        # Evaluate rules
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            should_switch, recommended_model = rule.evaluate(
                performance_snapshots, self.active_model_id
            )
            
            if should_switch and recommended_model:
                # Perform the switch
                success = self._switch_model(
                    recommended_model, rule, performance_snapshots[self.active_model_id]
                )
                
                if success:
                    break  # Stop evaluating rules after successful switch
    
    def _switch_model(self, new_model_id: str, triggered_rule: SwitchingRule,
                     current_performance: ModelPerformanceSnapshot) -> bool:
        """
        Switch to a new model.
        
        Args:
            new_model_id: ID of model to switch to
            triggered_rule: Rule that triggered the switch
            current_performance: Current model's performance
            
        Returns:
            True if switch was successful
        """
        if new_model_id not in self.available_models:
            logger.error(f"Cannot switch to unknown model: {new_model_id}")
            return False
        
        old_model_id = self.active_model_id
        
        # Create switch event
        event = ModelSwitchEvent(
            event_id=f"switch_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            trigger=triggered_rule.trigger,
            rule_id=triggered_rule.rule_id,
            from_model_id=old_model_id,
            to_model_id=new_model_id,
            trigger_metric=triggered_rule.metric_name,
            trigger_value=getattr(current_performance, triggered_rule.metric_name),
            threshold_value=triggered_rule.threshold_value
        )
        
        try:
            # Perform the switch
            self.active_model_id = new_model_id
            self.last_switch_time = datetime.now()
            
            # Record the event
            self.switch_history.append(event)
            
            logger.info(
                f"Model switched from {old_model_id} to {new_model_id} "
                f"(trigger: {triggered_rule.name})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            event.success = False
            self.switch_history.append(event)
            return False
    
    def manual_switch(self, new_model_id: str, reason: str = "Manual switch") -> bool:
        """
        Manually switch to a different model.
        
        Args:
            new_model_id: ID of model to switch to
            reason: Reason for manual switch
            
        Returns:
            True if switch was successful
        """
        if new_model_id not in self.available_models:
            logger.error(f"Cannot switch to unknown model: {new_model_id}")
            return False
        
        if new_model_id == self.active_model_id:
            logger.info(f"Model {new_model_id} is already active")
            return True
        
        old_model_id = self.active_model_id
        
        # Create manual switch event
        event = ModelSwitchEvent(
            event_id=f"manual_switch_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            trigger=SwitchingTrigger.MANUAL,
            rule_id="manual",
            from_model_id=old_model_id,
            to_model_id=new_model_id,
            trigger_metric="manual",
            trigger_value=0.0,
            threshold_value=0.0
        )
        
        try:
            self.active_model_id = new_model_id
            self.last_switch_time = datetime.now()
            self.switch_history.append(event)
            
            logger.info(f"Manual switch from {old_model_id} to {new_model_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Manual model switch failed: {e}")
            event.success = False
            self.switch_history.append(event)
            return False
    
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model instance."""
        if self.active_model_id:
            return self.available_models.get(self.active_model_id)
        return None
    
    def get_active_model_id(self) -> Optional[str]:
        """Get the currently active model ID."""
        return self.active_model_id
    
    def get_model_performance(self, model_id: Optional[str] = None) -> Optional[ModelPerformanceSnapshot]:
        """Get performance snapshot for a model."""
        if model_id is None:
            model_id = self.active_model_id
        
        if model_id:
            return self.performance_evaluator.get_performance_snapshot(model_id)
        return None
    
    def get_all_model_performance(self) -> Dict[str, ModelPerformanceSnapshot]:
        """Get performance snapshots for all models."""
        return self.performance_evaluator.get_all_performance_snapshots()
    
    def get_switch_history(self, hours: int = 24) -> List[ModelSwitchEvent]:
        """Get model switch history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.switch_history
            if event.timestamp >= cutoff_time
        ]
    
    def get_switching_statistics(self) -> Dict[str, Any]:
        """Get switching statistics."""
        total_switches = len(self.switch_history)
        successful_switches = sum(1 for event in self.switch_history if event.success)
        
        # Count switches by trigger type
        trigger_counts = defaultdict(int)
        for event in self.switch_history:
            trigger_counts[event.trigger.value] += 1
        
        # Recent switch rate (last 24 hours)
        recent_switches = self.get_switch_history(24)
        switch_rate_24h = len(recent_switches)
        
        # Model usage statistics
        model_usage = defaultdict(int)
        for event in self.switch_history:
            model_usage[event.to_model_id] += 1
        
        return {
            'total_switches': total_switches,
            'successful_switches': successful_switches,
            'success_rate': (successful_switches / total_switches * 100) if total_switches > 0 else 0,
            'switch_rate_24h': switch_rate_24h,
            'trigger_distribution': dict(trigger_counts),
            'model_usage': dict(model_usage),
            'active_model': self.active_model_id,
            'available_models': list(self.available_models.keys()),
            'enabled_rules': len([r for r in self.switching_rules.values() if r.enabled]),
            'total_rules': len(self.switching_rules)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export switching configuration."""
        return {
            'switching_rules': [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'trigger': rule.trigger.value,
                    'strategy': rule.strategy.value,
                    'metric_name': rule.metric_name,
                    'threshold_value': rule.threshold_value,
                    'comparison_operator': rule.comparison_operator,
                    'evaluation_window_minutes': rule.evaluation_window_minutes,
                    'min_observations': rule.min_observations,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'enabled': rule.enabled,
                    'priority': rule.priority,
                    'conditions': rule.conditions
                }
                for rule in self.switching_rules.values()
            ],
            'model_metadata': self.model_metadata,
            'cooldown_minutes': self.cooldown_minutes,
            'active_model_id': self.active_model_id
        }


class AdaptiveModelSwitcher:
    """
    Main adaptive model switching system.
    """
    
    def __init__(self, cooldown_minutes: int = 30):
        """
        Initialize adaptive model switcher.
        
        Args:
            cooldown_minutes: Minimum time between model switches
        """
        self.engine = ModelSwitchingEngine(cooldown_minutes)
        self.is_running = False
        
        # Setup default switching rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default switching rules."""
        
        # Performance degradation rule
        self.engine.add_switching_rule(SwitchingRule(
            rule_id="mae_degradation",
            name="MAE Degradation",
            trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
            strategy=SwitchingStrategy.BEST_PERFORMER,
            metric_name="mae",
            threshold_value=0.1,
            comparison_operator=">",
            evaluation_window_minutes=60,
            min_observations=10,
            cooldown_minutes=30,
            priority=1
        ))
        
        # Relative performance rule
        self.engine.add_switching_rule(SwitchingRule(
            rule_id="accuracy_degradation",
            name="Accuracy Degradation",
            trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
            strategy=SwitchingStrategy.BEST_PERFORMER,
            metric_name="directional_accuracy",
            threshold_value=45.0,
            comparison_operator="<",
            evaluation_window_minutes=90,
            min_observations=15,
            cooldown_minutes=45,
            priority=2
        ))
        
        # Confidence-based rule
        self.engine.add_switching_rule(SwitchingRule(
            rule_id="low_confidence",
            name="Low Confidence",
            trigger=SwitchingTrigger.CONFIDENCE_THRESHOLD,
            strategy=SwitchingStrategy.CONFIDENCE_BASED,
            metric_name="confidence_score",
            threshold_value=0.3,
            comparison_operator="<",
            evaluation_window_minutes=30,
            min_observations=5,
            cooldown_minutes=20,
            priority=3
        ))
    
    def register_model(self, model_id: str, model_instance: Any, 
                      metadata: Optional[Dict[str, Any]] = None):
        """Register a model for switching."""
        self.engine.register_model(model_id, model_instance, metadata)
    
    def unregister_model(self, model_id: str):
        """Unregister a model."""
        self.engine.unregister_model(model_id)
    
    def add_custom_rule(self, rule: SwitchingRule):
        """Add a custom switching rule."""
        self.engine.add_switching_rule(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a switching rule."""
        self.engine.remove_switching_rule(rule_id)
    
    def add_prediction(self, model_id: str, prediction: float, actual: Optional[float] = None,
                      confidence: float = 1.0, timestamp: Optional[datetime] = None):
        """Add a prediction and potentially trigger model switching."""
        self.engine.add_prediction(model_id, prediction, actual, confidence, timestamp)
    
    def manual_switch(self, new_model_id: str, reason: str = "Manual switch") -> bool:
        """Manually switch to a different model."""
        return self.engine.manual_switch(new_model_id, reason)
    
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model instance."""
        return self.engine.get_active_model()
    
    def get_active_model_id(self) -> Optional[str]:
        """Get the currently active model ID."""
        return self.engine.get_active_model_id()
    
    def get_model_performance(self, model_id: Optional[str] = None) -> Optional[ModelPerformanceSnapshot]:
        """Get performance snapshot for a model."""
        return self.engine.get_model_performance(model_id)
    
    def get_all_model_performance(self) -> Dict[str, ModelPerformanceSnapshot]:
        """Get performance snapshots for all models."""
        return self.engine.get_all_model_performance()
    
    def get_switch_history(self, hours: int = 24) -> List[ModelSwitchEvent]:
        """Get model switch history."""
        return self.engine.get_switch_history(hours)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get switching statistics."""
        return self.engine.get_switching_statistics()
    
    def export_configuration(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export switching configuration."""
        config = self.engine.export_configuration()
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        return config
    
    def start_monitoring(self):
        """Start automatic monitoring and switching."""
        self.is_running = True
        logger.info("Adaptive model switcher started")
    
    def stop_monitoring(self):
        """Stop automatic monitoring and switching."""
        self.is_running = False
        logger.info("Adaptive model switcher stopped")
    
    def predict(self, *args, **kwargs) -> Any:
        """
        Make a prediction using the active model.
        
        This method delegates to the active model's predict method.
        """
        active_model = self.get_active_model()
        if active_model is None:
            raise RuntimeError("No active model available for prediction")
        
        # Check if model has predict method
        if hasattr(active_model, 'predict'):
            return active_model.predict(*args, **kwargs)
        elif callable(active_model):
            return active_model(*args, **kwargs)
        else:
            raise RuntimeError(f"Active model {self.get_active_model_id()} is not callable")
    
    def evaluate_and_switch(self, predictions: Dict[str, float], 
                           actuals: Dict[str, float], 
                           confidences: Optional[Dict[str, float]] = None,
                           timestamp: Optional[datetime] = None):
        """
        Evaluate multiple model predictions and potentially switch models.
        
        Args:
            predictions: Dictionary of model_id -> prediction
            actuals: Dictionary of model_id -> actual_value
            confidences: Optional dictionary of model_id -> confidence
            timestamp: Prediction timestamp
        """
        if confidences is None:
            confidences = {model_id: 1.0 for model_id in predictions.keys()}
        
        # Add predictions for all models
        for model_id in predictions.keys():
            prediction = predictions.get(model_id)
            actual = actuals.get(model_id)
            confidence = confidences.get(model_id, 1.0)
            
            if prediction is not None:
                self.add_prediction(model_id, prediction, actual, confidence, timestamp)


def create_default_switching_system(models: Dict[str, Any], 
                                   cooldown_minutes: int = 30) -> AdaptiveModelSwitcher:
    """
    Create a default adaptive model switching system with common rules.
    
    Args:
        models: Dictionary of model_id -> model_instance
        cooldown_minutes: Minimum time between switches
        
    Returns:
        Configured AdaptiveModelSwitcher instance
    """
    switcher = AdaptiveModelSwitcher(cooldown_minutes)
    
    # Register all models
    for model_id, model_instance in models.items():
        switcher.register_model(model_id, model_instance)
    
    # Add additional common rules
    
    # RMSE degradation rule
    switcher.add_custom_rule(SwitchingRule(
        rule_id="rmse_degradation",
        name="RMSE Degradation",
        trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
        strategy=SwitchingStrategy.BEST_PERFORMER,
        metric_name="rmse",
        threshold_value=0.15,
        comparison_operator=">",
        evaluation_window_minutes=45,
        min_observations=8,
        cooldown_minutes=25,
        priority=2
    ))
    
    # Sharpe ratio degradation rule
    switcher.add_custom_rule(SwitchingRule(
        rule_id="sharpe_degradation",
        name="Sharpe Ratio Degradation",
        trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
        strategy=SwitchingStrategy.BEST_PERFORMER,
        metric_name="sharpe_ratio",
        threshold_value=-0.5,
        comparison_operator="<",
        evaluation_window_minutes=120,
        min_observations=20,
        cooldown_minutes=60,
        priority=4
    ))
    
    return switcher


# Example usage and utility functions
def create_performance_based_switcher(models: Dict[str, Any]) -> AdaptiveModelSwitcher:
    """Create a switcher focused on performance metrics."""
    switcher = AdaptiveModelSwitcher(cooldown_minutes=20)
    
    for model_id, model_instance in models.items():
        switcher.register_model(model_id, model_instance)
    
    # Aggressive performance-based switching
    switcher.add_custom_rule(SwitchingRule(
        rule_id="aggressive_mae",
        name="Aggressive MAE Switching",
        trigger=SwitchingTrigger.PERFORMANCE_DEGRADATION,
        strategy=SwitchingStrategy.BEST_PERFORMER,
        metric_name="mae",
        threshold_value=0.05,
        comparison_operator=">",
        evaluation_window_minutes=30,
        min_observations=5,
        cooldown_minutes=15,
        priority=1
    ))
    
    return switcher


def create_confidence_based_switcher(models: Dict[str, Any]) -> AdaptiveModelSwitcher:
    """Create a switcher focused on prediction confidence."""
    switcher = AdaptiveModelSwitcher(cooldown_minutes=15)
    
    for model_id, model_instance in models.items():
        switcher.register_model(model_id, model_instance)
    
    # Confidence-based switching
    switcher.add_custom_rule(SwitchingRule(
        rule_id="confidence_switching",
        name="Confidence-Based Switching",
        trigger=SwitchingTrigger.CONFIDENCE_THRESHOLD,
        strategy=SwitchingStrategy.CONFIDENCE_BASED,
        metric_name="confidence_score",
        threshold_value=0.6,
        comparison_operator="<",
        evaluation_window_minutes=20,
        min_observations=3,
        cooldown_minutes=10,
        priority=1
    ))
    
    return switcher