"""
Prediction confidence scoring and risk assessment for time series models.
Implements confidence metrics, risk-adjusted position sizing, and alert systems.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class RiskLevel(Enum):
    """Risk level categories."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ConfidenceScore:
    """Confidence score with detailed breakdown."""
    overall_confidence: float
    prediction_confidence: float
    model_confidence: float
    data_confidence: float
    historical_confidence: float
    confidence_level: ConfidenceLevel
    confidence_factors: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_confidence': self.overall_confidence,
            'prediction_confidence': self.prediction_confidence,
            'model_confidence': self.model_confidence,
            'data_confidence': self.data_confidence,
            'historical_confidence': self.historical_confidence,
            'confidence_level': self.confidence_level.value,
            'confidence_factors': self.confidence_factors
        }


@dataclass
class RiskAssessment:
    """Risk assessment with position sizing recommendations."""
    risk_score: float
    risk_level: RiskLevel
    max_position_size: float
    recommended_position_size: float
    stop_loss_level: float
    take_profit_level: float
    risk_factors: Dict[str, float]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'max_position_size': self.max_position_size,
            'recommended_position_size': self.recommended_position_size,
            'stop_loss_level': self.stop_loss_level,
            'take_profit_level': self.take_profit_level,
            'risk_factors': self.risk_factors,
            'warnings': self.warnings
        }


class ConfidenceScorer:
    """
    Comprehensive confidence scoring system for predictions.
    """
    
    def __init__(self, confidence_thresholds: Dict[str, float] = None):
        self.confidence_thresholds = confidence_thresholds or {
            'very_high': 0.9,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25,
            'very_low': 0.0
        }
        
        # Historical performance tracking
        self.prediction_history = []
        self.accuracy_history = []
        self.confidence_history = []
        
    def calculate_confidence(self, prediction: float, uncertainty: float,
                           model_agreement: Optional[float] = None,
                           data_quality: Optional[float] = None,
                           historical_accuracy: Optional[float] = None,
                           additional_factors: Optional[Dict[str, float]] = None) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score.
        
        Args:
            prediction: Model prediction
            uncertainty: Prediction uncertainty
            model_agreement: Agreement between multiple models (0-1)
            data_quality: Quality of input data (0-1)
            historical_accuracy: Historical model accuracy (0-1)
            additional_factors: Additional confidence factors
            
        Returns:
            Detailed confidence score
        """
        # Initialize factors
        factors = additional_factors or {}
        
        # 1. Prediction confidence (based on uncertainty)
        prediction_confidence = self._uncertainty_to_confidence(uncertainty)
        factors['prediction_uncertainty'] = prediction_confidence
        
        # 2. Model confidence (based on ensemble agreement)
        if model_agreement is not None:
            model_confidence = model_agreement
        else:
            model_confidence = 0.7  # Default moderate confidence
        factors['model_agreement'] = model_confidence
        
        # 3. Data confidence (based on data quality)
        if data_quality is not None:
            data_confidence = data_quality
        else:
            data_confidence = 0.8  # Default good data quality
        factors['data_quality'] = data_confidence
        
        # 4. Historical confidence (based on past performance)
        if historical_accuracy is not None:
            historical_confidence = historical_accuracy
        else:
            historical_confidence = self._calculate_historical_confidence()
        factors['historical_accuracy'] = historical_confidence
        
        # 5. Additional factors
        volatility_confidence = factors.get('volatility_stability', 0.7)
        regime_confidence = factors.get('regime_stability', 0.7)
        feature_confidence = factors.get('feature_importance', 0.7)
        
        # Weighted combination of confidence factors
        weights = {
            'prediction': 0.3,
            'model': 0.25,
            'data': 0.2,
            'historical': 0.15,
            'volatility': 0.05,
            'regime': 0.03,
            'feature': 0.02
        }
        
        overall_confidence = (
            weights['prediction'] * prediction_confidence +
            weights['model'] * model_confidence +
            weights['data'] * data_confidence +
            weights['historical'] * historical_confidence +
            weights['volatility'] * volatility_confidence +
            weights['regime'] * regime_confidence +
            weights['feature'] * feature_confidence
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(overall_confidence)
        
        return ConfidenceScore(
            overall_confidence=overall_confidence,
            prediction_confidence=prediction_confidence,
            model_confidence=model_confidence,
            data_confidence=data_confidence,
            historical_confidence=historical_confidence,
            confidence_level=confidence_level,
            confidence_factors=factors
        )
    
    def _uncertainty_to_confidence(self, uncertainty: float) -> float:
        """Convert uncertainty to confidence score."""
        # Assume uncertainty is normalized (0-1 range)
        # Higher uncertainty = lower confidence
        return max(0.0, min(1.0, 1.0 - uncertainty))
    
    def _calculate_historical_confidence(self) -> float:
        """Calculate confidence based on historical performance."""
        if len(self.accuracy_history) < 10:
            return 0.7  # Default moderate confidence
        
        # Recent accuracy (last 50 predictions)
        recent_accuracy = np.mean(self.accuracy_history[-50:])
        
        # Trend in accuracy
        if len(self.accuracy_history) >= 20:
            recent_trend = np.mean(self.accuracy_history[-10:]) - np.mean(self.accuracy_history[-20:-10])
            trend_factor = max(0.0, min(0.2, recent_trend))
        else:
            trend_factor = 0.0
        
        # Stability of accuracy
        if len(self.accuracy_history) >= 20:
            accuracy_std = np.std(self.accuracy_history[-20:])
            stability_factor = max(0.0, 0.2 - accuracy_std)
        else:
            stability_factor = 0.1
        
        historical_confidence = recent_accuracy + trend_factor + stability_factor
        return max(0.0, min(1.0, historical_confidence))
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to categorical level."""
        if confidence >= self.confidence_thresholds['very_high']:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= self.confidence_thresholds['high']:
            return ConfidenceLevel.HIGH
        elif confidence >= self.confidence_thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.confidence_thresholds['low']:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def update_performance(self, prediction: float, actual: float, confidence: float):
        """Update performance tracking."""
        self.prediction_history.append(prediction)
        
        # Calculate accuracy (inverse of relative error)
        error = abs(prediction - actual) / (abs(actual) + 1e-8)
        accuracy = max(0.0, 1.0 - error)
        
        self.accuracy_history.append(accuracy)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        max_history = 1000
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
            self.accuracy_history = self.accuracy_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]


class RiskAssessor:
    """
    Risk assessment and position sizing system.
    """
    
    def __init__(self, risk_tolerance: float = 0.02, max_position_size: float = 1.0):
        self.risk_tolerance = risk_tolerance  # Maximum acceptable risk per trade
        self.max_position_size = max_position_size  # Maximum position size
        
        # Risk thresholds
        self.risk_thresholds = {
            'very_low': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8,
            'very_high': 1.0
        }
        
    def assess_risk(self, prediction: float, uncertainty: float, confidence_score: ConfidenceScore,
                   current_price: float, volatility: Optional[float] = None,
                   market_conditions: Optional[Dict[str, float]] = None) -> RiskAssessment:
        """
        Comprehensive risk assessment.
        
        Args:
            prediction: Model prediction
            uncertainty: Prediction uncertainty
            confidence_score: Confidence score object
            current_price: Current asset price
            volatility: Asset volatility
            market_conditions: Market condition factors
            
        Returns:
            Risk assessment with position sizing recommendations
        """
        market_conditions = market_conditions or {}
        
        # Calculate risk factors
        risk_factors = {}
        
        # 1. Prediction risk (based on uncertainty and confidence)
        prediction_risk = 1.0 - confidence_score.overall_confidence
        risk_factors['prediction_uncertainty'] = prediction_risk
        
        # 2. Volatility risk
        if volatility is not None:
            volatility_risk = min(1.0, volatility / 0.3)  # Normalize by 30% volatility
        else:
            volatility_risk = 0.5  # Default moderate volatility risk
        risk_factors['volatility'] = volatility_risk
        
        # 3. Market regime risk
        regime_risk = market_conditions.get('regime_uncertainty', 0.3)
        risk_factors['market_regime'] = regime_risk
        
        # 4. Liquidity risk
        liquidity_risk = market_conditions.get('liquidity_risk', 0.2)
        risk_factors['liquidity'] = liquidity_risk
        
        # 5. Correlation risk (for portfolio context)
        correlation_risk = market_conditions.get('correlation_risk', 0.2)
        risk_factors['correlation'] = correlation_risk
        
        # 6. Tail risk (extreme event probability)
        tail_risk = market_conditions.get('tail_risk', 0.1)
        risk_factors['tail_events'] = tail_risk
        
        # Weighted risk score
        weights = {
            'prediction': 0.4,
            'volatility': 0.25,
            'regime': 0.15,
            'liquidity': 0.1,
            'correlation': 0.05,
            'tail': 0.05
        }
        
        overall_risk = (
            weights['prediction'] * prediction_risk +
            weights['volatility'] * volatility_risk +
            weights['regime'] * regime_risk +
            weights['liquidity'] * liquidity_risk +
            weights['correlation'] * correlation_risk +
            weights['tail'] * tail_risk
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(overall_risk)
        
        # Calculate position sizing
        position_sizing = self._calculate_position_sizing(
            prediction, uncertainty, confidence_score, current_price, overall_risk
        )
        
        # Generate warnings
        warnings = self._generate_risk_warnings(risk_factors, confidence_score)
        
        return RiskAssessment(
            risk_score=overall_risk,
            risk_level=risk_level,
            max_position_size=self.max_position_size,
            recommended_position_size=position_sizing['recommended'],
            stop_loss_level=position_sizing['stop_loss'],
            take_profit_level=position_sizing['take_profit'],
            risk_factors=risk_factors,
            warnings=warnings
        )
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to categorical level."""
        if risk_score <= self.risk_thresholds['very_low']:
            return RiskLevel.VERY_LOW
        elif risk_score <= self.risk_thresholds['low']:
            return RiskLevel.LOW
        elif risk_score <= self.risk_thresholds['medium']:
            return RiskLevel.MEDIUM
        elif risk_score <= self.risk_thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _calculate_position_sizing(self, prediction: float, uncertainty: float,
                                 confidence_score: ConfidenceScore, current_price: float,
                                 risk_score: float) -> Dict[str, float]:
        """Calculate position sizing recommendations."""
        # Base position size using Kelly criterion approximation
        expected_return = prediction
        prediction_std = uncertainty * current_price
        
        if prediction_std > 0:
            # Simplified Kelly fraction
            kelly_fraction = abs(expected_return) / (prediction_std ** 2)
            kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
        else:
            kelly_fraction = 0.1  # Default conservative size
        
        # Adjust for confidence
        confidence_adjustment = confidence_score.overall_confidence
        
        # Adjust for risk
        risk_adjustment = 1.0 - risk_score
        
        # Calculate recommended position size
        recommended_size = kelly_fraction * confidence_adjustment * risk_adjustment
        recommended_size = min(recommended_size, self.max_position_size)
        recommended_size = max(recommended_size, 0.01)  # Minimum 1%
        
        # Calculate stop loss and take profit levels
        if prediction > 0:  # Long position
            stop_loss = current_price * (1 - 2 * uncertainty)
            take_profit = current_price * (1 + abs(prediction) * 2)
        else:  # Short position
            stop_loss = current_price * (1 + 2 * uncertainty)
            take_profit = current_price * (1 - abs(prediction) * 2)
        
        return {
            'recommended': recommended_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def _generate_risk_warnings(self, risk_factors: Dict[str, float],
                              confidence_score: ConfidenceScore) -> List[str]:
        """Generate risk warnings based on factors."""
        warnings = []
        
        # High uncertainty warning
        if risk_factors.get('prediction_uncertainty', 0) > 0.7:
            warnings.append("HIGH UNCERTAINTY: Model prediction has high uncertainty")
        
        # Low confidence warning
        if confidence_score.overall_confidence < 0.3:
            warnings.append("LOW CONFIDENCE: Overall prediction confidence is low")
        
        # High volatility warning
        if risk_factors.get('volatility', 0) > 0.8:
            warnings.append("HIGH VOLATILITY: Asset volatility is elevated")
        
        # Market regime warning
        if risk_factors.get('market_regime', 0) > 0.7:
            warnings.append("REGIME UNCERTAINTY: Market regime is uncertain")
        
        # Liquidity warning
        if risk_factors.get('liquidity', 0) > 0.6:
            warnings.append("LIQUIDITY RISK: Potential liquidity constraints")
        
        # Tail risk warning
        if risk_factors.get('tail_events', 0) > 0.3:
            warnings.append("TAIL RISK: Elevated probability of extreme events")
        
        return warnings


class AlertSystem:
    """
    Alert system for confidence and risk monitoring.
    """
    
    def __init__(self, confidence_threshold: float = 0.3, risk_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.alert_history = []
        
    def check_alerts(self, confidence_score: ConfidenceScore, 
                    risk_assessment: RiskAssessment) -> List[Dict[str, Any]]:
        """
        Check for confidence and risk alerts.
        
        Args:
            confidence_score: Confidence score object
            risk_assessment: Risk assessment object
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Low confidence alert
        if confidence_score.overall_confidence < self.confidence_threshold:
            alerts.append({
                'type': 'LOW_CONFIDENCE',
                'severity': 'WARNING',
                'message': f"Prediction confidence ({confidence_score.overall_confidence:.2f}) below threshold ({self.confidence_threshold})",
                'confidence': confidence_score.overall_confidence,
                'threshold': self.confidence_threshold,
                'timestamp': pd.Timestamp.now()
            })
        
        # High risk alert
        if risk_assessment.risk_score > self.risk_threshold:
            alerts.append({
                'type': 'HIGH_RISK',
                'severity': 'CRITICAL',
                'message': f"Risk score ({risk_assessment.risk_score:.2f}) above threshold ({self.risk_threshold})",
                'risk_score': risk_assessment.risk_score,
                'threshold': self.risk_threshold,
                'timestamp': pd.Timestamp.now()
            })
        
        # Model disagreement alert
        if confidence_score.model_confidence < 0.5:
            alerts.append({
                'type': 'MODEL_DISAGREEMENT',
                'severity': 'WARNING',
                'message': f"Low model agreement ({confidence_score.model_confidence:.2f})",
                'model_confidence': confidence_score.model_confidence,
                'timestamp': pd.Timestamp.now()
            })
        
        # Data quality alert
        if confidence_score.data_confidence < 0.6:
            alerts.append({
                'type': 'DATA_QUALITY',
                'severity': 'WARNING',
                'message': f"Low data quality ({confidence_score.data_confidence:.2f})",
                'data_confidence': confidence_score.data_confidence,
                'timestamp': pd.Timestamp.now()
            })
        
        # Add risk-specific alerts
        for warning in risk_assessment.warnings:
            alerts.append({
                'type': 'RISK_WARNING',
                'severity': 'INFO',
                'message': warning,
                'timestamp': pd.Timestamp.now()
            })
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        # Keep only recent alerts
        max_history = 1000
        if len(self.alert_history) > max_history:
            self.alert_history = self.alert_history[-max_history:]
        
        return alerts
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > cutoff_time
        ]
        
        # Count by type and severity
        alert_counts = {}
        severity_counts = {}
        
        for alert in recent_alerts:
            alert_type = alert['type']
            severity = alert['severity']
            
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'alert_counts': alert_counts,
            'severity_counts': severity_counts,
            'time_period_hours': hours,
            'most_recent_alert': recent_alerts[-1] if recent_alerts else None
        }


class ConfidenceRiskManager:
    """
    Integrated confidence and risk management system.
    """
    
    def __init__(self, confidence_threshold: float = 0.3, risk_threshold: float = 0.8,
                 risk_tolerance: float = 0.02):
        self.confidence_scorer = ConfidenceScorer()
        self.risk_assessor = RiskAssessor(risk_tolerance=risk_tolerance)
        self.alert_system = AlertSystem(confidence_threshold, risk_threshold)
        
    def evaluate_prediction(self, prediction: float, uncertainty: float, current_price: float,
                          model_agreement: Optional[float] = None,
                          data_quality: Optional[float] = None,
                          volatility: Optional[float] = None,
                          market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive prediction evaluation.
        
        Args:
            prediction: Model prediction
            uncertainty: Prediction uncertainty
            current_price: Current asset price
            model_agreement: Agreement between models
            data_quality: Quality of input data
            volatility: Asset volatility
            market_conditions: Market condition factors
            
        Returns:
            Complete evaluation including confidence, risk, and alerts
        """
        # Calculate confidence score
        confidence_score = self.confidence_scorer.calculate_confidence(
            prediction=prediction,
            uncertainty=uncertainty,
            model_agreement=model_agreement,
            data_quality=data_quality
        )
        
        # Assess risk
        risk_assessment = self.risk_assessor.assess_risk(
            prediction=prediction,
            uncertainty=uncertainty,
            confidence_score=confidence_score,
            current_price=current_price,
            volatility=volatility,
            market_conditions=market_conditions
        )
        
        # Check alerts
        alerts = self.alert_system.check_alerts(confidence_score, risk_assessment)
        
        return {
            'confidence_score': confidence_score.to_dict(),
            'risk_assessment': risk_assessment.to_dict(),
            'alerts': alerts,
            'recommendation': self._generate_recommendation(confidence_score, risk_assessment)
        }
    
    def _generate_recommendation(self, confidence_score: ConfidenceScore,
                               risk_assessment: RiskAssessment) -> Dict[str, Any]:
        """Generate trading recommendation."""
        # Determine action based on confidence and risk
        if confidence_score.confidence_level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            if risk_assessment.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
                action = "STRONG_BUY" if risk_assessment.recommended_position_size > 0 else "STRONG_SELL"
            elif risk_assessment.risk_level == RiskLevel.MEDIUM:
                action = "BUY" if risk_assessment.recommended_position_size > 0 else "SELL"
            else:
                action = "HOLD"
        elif confidence_score.confidence_level == ConfidenceLevel.MEDIUM:
            if risk_assessment.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
                action = "WEAK_BUY" if risk_assessment.recommended_position_size > 0 else "WEAK_SELL"
            else:
                action = "HOLD"
        else:
            action = "HOLD"
        
        return {
            'action': action,
            'position_size': risk_assessment.recommended_position_size,
            'confidence_level': confidence_score.confidence_level.value,
            'risk_level': risk_assessment.risk_level.value,
            'stop_loss': risk_assessment.stop_loss_level,
            'take_profit': risk_assessment.take_profit_level
        }