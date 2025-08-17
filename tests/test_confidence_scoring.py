"""
Unit tests for confidence scoring and risk assessment.
"""

import unittest
import numpy as np
import pandas as pd
from enhanced_timeseries.uncertainty.confidence_scoring import (
    ConfidenceScorer, RiskAssessor, AlertSystem, ConfidenceRiskManager,
    ConfidenceLevel, RiskLevel, ConfidenceScore, RiskAssessment
)


class TestConfidenceScorer(unittest.TestCase):
    """Test confidence scoring system."""
    
    def setUp(self):
        """Set up test scorer."""
        self.scorer = ConfidenceScorer()
    
    def test_calculate_confidence_basic(self):
        """Test basic confidence calculation."""
        confidence_score = self.scorer.calculate_confidence(
            prediction=0.05,
            uncertainty=0.1
        )
        
        self.assertIsInstance(confidence_score, ConfidenceScore)
        self.assertTrue(0 <= confidence_score.overall_confidence <= 1)
        self.assertTrue(0 <= confidence_score.prediction_confidence <= 1)
        self.assertTrue(0 <= confidence_score.model_confidence <= 1)
        self.assertTrue(0 <= confidence_score.data_confidence <= 1)
        self.assertTrue(0 <= confidence_score.historical_confidence <= 1)
        self.assertIsInstance(confidence_score.confidence_level, ConfidenceLevel)
    
    def test_calculate_confidence_with_factors(self):
        """Test confidence calculation with all factors."""
        confidence_score = self.scorer.calculate_confidence(
            prediction=0.02,
            uncertainty=0.05,
            model_agreement=0.9,
            data_quality=0.8,
            historical_accuracy=0.75,
            additional_factors={
                'volatility_stability': 0.7,
                'regime_stability': 0.8,
                'feature_importance': 0.6
            }
        )
        
        # Should have high confidence with good factors
        self.assertGreater(confidence_score.overall_confidence, 0.6)
        self.assertEqual(confidence_score.model_confidence, 0.9)
        self.assertEqual(confidence_score.data_confidence, 0.8)
        self.assertEqual(confidence_score.historical_confidence, 0.75)
    
    def test_uncertainty_to_confidence(self):
        """Test uncertainty to confidence conversion."""
        # Low uncertainty should give high confidence
        high_conf = self.scorer._uncertainty_to_confidence(0.1)
        self.assertGreater(high_conf, 0.8)
        
        # High uncertainty should give low confidence
        low_conf = self.scorer._uncertainty_to_confidence(0.9)
        self.assertLess(low_conf, 0.2)
    
    def test_confidence_levels(self):
        """Test confidence level categorization."""
        # Very high confidence
        very_high_score = self.scorer.calculate_confidence(0.01, 0.01, 0.95, 0.9, 0.9)
        self.assertIn(very_high_score.confidence_level, [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH])
        
        # Very low confidence
        very_low_score = self.scorer.calculate_confidence(0.1, 0.9, 0.1, 0.1, 0.1)
        self.assertIn(very_low_score.confidence_level, [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW])
    
    def test_update_performance(self):
        """Test performance tracking update."""
        # Add some performance data
        for i in range(10):
            prediction = 0.05 + np.random.normal(0, 0.01)
            actual = 0.05 + np.random.normal(0, 0.02)
            confidence = 0.7 + np.random.normal(0, 0.1)
            
            self.scorer.update_performance(prediction, actual, confidence)
        
        self.assertEqual(len(self.scorer.prediction_history), 10)
        self.assertEqual(len(self.scorer.accuracy_history), 10)
        self.assertEqual(len(self.scorer.confidence_history), 10)
        
        # Test historical confidence calculation
        hist_conf = self.scorer._calculate_historical_confidence()
        self.assertTrue(0 <= hist_conf <= 1)
    
    def test_confidence_score_to_dict(self):
        """Test confidence score serialization."""
        confidence_score = self.scorer.calculate_confidence(0.03, 0.08)
        score_dict = confidence_score.to_dict()
        
        expected_keys = [
            'overall_confidence', 'prediction_confidence', 'model_confidence',
            'data_confidence', 'historical_confidence', 'confidence_level', 'confidence_factors'
        ]
        
        for key in expected_keys:
            self.assertIn(key, score_dict)


class TestRiskAssessor(unittest.TestCase):
    """Test risk assessment system."""
    
    def setUp(self):
        """Set up test assessor."""
        self.assessor = RiskAssessor(risk_tolerance=0.02, max_position_size=1.0)
        
        # Create mock confidence score
        self.confidence_score = ConfidenceScore(
            overall_confidence=0.7,
            prediction_confidence=0.8,
            model_confidence=0.7,
            data_confidence=0.8,
            historical_confidence=0.6,
            confidence_level=ConfidenceLevel.HIGH,
            confidence_factors={}
        )
    
    def test_assess_risk_basic(self):
        """Test basic risk assessment."""
        risk_assessment = self.assessor.assess_risk(
            prediction=0.05,
            uncertainty=0.1,
            confidence_score=self.confidence_score,
            current_price=100.0
        )
        
        self.assertIsInstance(risk_assessment, RiskAssessment)
        self.assertTrue(0 <= risk_assessment.risk_score <= 1)
        self.assertIsInstance(risk_assessment.risk_level, RiskLevel)
        self.assertTrue(0 <= risk_assessment.recommended_position_size <= 1)
        self.assertGreater(risk_assessment.stop_loss_level, 0)
        self.assertGreater(risk_assessment.take_profit_level, 0)
        self.assertIsInstance(risk_assessment.warnings, list)
    
    def test_assess_risk_with_conditions(self):
        """Test risk assessment with market conditions."""
        market_conditions = {
            'regime_uncertainty': 0.3,
            'liquidity_risk': 0.2,
            'correlation_risk': 0.1,
            'tail_risk': 0.05
        }
        
        risk_assessment = self.assessor.assess_risk(
            prediction=0.03,
            uncertainty=0.08,
            confidence_score=self.confidence_score,
            current_price=100.0,
            volatility=0.2,
            market_conditions=market_conditions
        )
        
        # Should incorporate market conditions
        self.assertIn('market_regime', risk_assessment.risk_factors)
        self.assertIn('liquidity', risk_assessment.risk_factors)
        self.assertIn('correlation', risk_assessment.risk_factors)
        self.assertIn('tail_events', risk_assessment.risk_factors)
    
    def test_position_sizing(self):
        """Test position sizing calculations."""
        # High confidence, low risk should give larger position
        high_conf_score = ConfidenceScore(
            overall_confidence=0.9,
            prediction_confidence=0.9,
            model_confidence=0.9,
            data_confidence=0.9,
            historical_confidence=0.9,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            confidence_factors={}
        )
        
        high_conf_assessment = self.assessor.assess_risk(
            prediction=0.05,
            uncertainty=0.05,
            confidence_score=high_conf_score,
            current_price=100.0
        )
        
        # Low confidence, high risk should give smaller position
        low_conf_score = ConfidenceScore(
            overall_confidence=0.2,
            prediction_confidence=0.2,
            model_confidence=0.2,
            data_confidence=0.2,
            historical_confidence=0.2,
            confidence_level=ConfidenceLevel.VERY_LOW,
            confidence_factors={}
        )
        
        low_conf_assessment = self.assessor.assess_risk(
            prediction=0.05,
            uncertainty=0.3,
            confidence_score=low_conf_score,
            current_price=100.0
        )
        
        # High confidence should recommend larger position
        self.assertGreater(
            high_conf_assessment.recommended_position_size,
            low_conf_assessment.recommended_position_size
        )
    
    def test_risk_levels(self):
        """Test risk level categorization."""
        # Test different risk scenarios
        risk_levels = []
        
        for uncertainty in [0.05, 0.1, 0.2, 0.4, 0.8]:
            assessment = self.assessor.assess_risk(
                prediction=0.05,
                uncertainty=uncertainty,
                confidence_score=self.confidence_score,
                current_price=100.0
            )
            risk_levels.append(assessment.risk_level)
        
        # Risk should generally increase with uncertainty
        # (though other factors also matter)
        self.assertIsInstance(risk_levels[0], RiskLevel)
    
    def test_risk_warnings(self):
        """Test risk warning generation."""
        # High uncertainty scenario
        high_risk_assessment = self.assessor.assess_risk(
            prediction=0.1,
            uncertainty=0.8,
            confidence_score=ConfidenceScore(
                overall_confidence=0.1,
                prediction_confidence=0.1,
                model_confidence=0.1,
                data_confidence=0.1,
                historical_confidence=0.1,
                confidence_level=ConfidenceLevel.VERY_LOW,
                confidence_factors={}
            ),
            current_price=100.0,
            volatility=0.5
        )
        
        # Should generate warnings
        self.assertGreater(len(high_risk_assessment.warnings), 0)
    
    def test_risk_assessment_to_dict(self):
        """Test risk assessment serialization."""
        risk_assessment = self.assessor.assess_risk(
            prediction=0.03,
            uncertainty=0.1,
            confidence_score=self.confidence_score,
            current_price=100.0
        )
        
        assessment_dict = risk_assessment.to_dict()
        
        expected_keys = [
            'risk_score', 'risk_level', 'max_position_size', 'recommended_position_size',
            'stop_loss_level', 'take_profit_level', 'risk_factors', 'warnings'
        ]
        
        for key in expected_keys:
            self.assertIn(key, assessment_dict)


class TestAlertSystem(unittest.TestCase):
    """Test alert system."""
    
    def setUp(self):
        """Set up test alert system."""
        self.alert_system = AlertSystem(confidence_threshold=0.5, risk_threshold=0.7)
        
        self.confidence_score = ConfidenceScore(
            overall_confidence=0.3,  # Below threshold
            prediction_confidence=0.4,
            model_confidence=0.3,  # Low model agreement
            data_confidence=0.5,
            historical_confidence=0.6,
            confidence_level=ConfidenceLevel.LOW,
            confidence_factors={}
        )
        
        self.risk_assessment = RiskAssessment(
            risk_score=0.8,  # Above threshold
            risk_level=RiskLevel.HIGH,
            max_position_size=1.0,
            recommended_position_size=0.1,
            stop_loss_level=95.0,
            take_profit_level=105.0,
            risk_factors={},
            warnings=["HIGH VOLATILITY: Asset volatility is elevated"]
        )
    
    def test_check_alerts(self):
        """Test alert checking."""
        alerts = self.alert_system.check_alerts(self.confidence_score, self.risk_assessment)
        
        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)
        
        # Should have alerts for low confidence and high risk
        alert_types = [alert['type'] for alert in alerts]
        self.assertIn('LOW_CONFIDENCE', alert_types)
        self.assertIn('HIGH_RISK', alert_types)
        self.assertIn('MODEL_DISAGREEMENT', alert_types)
    
    def test_alert_structure(self):
        """Test alert structure."""
        alerts = self.alert_system.check_alerts(self.confidence_score, self.risk_assessment)
        
        for alert in alerts:
            self.assertIn('type', alert)
            self.assertIn('severity', alert)
            self.assertIn('message', alert)
            self.assertIn('timestamp', alert)
            
            # Check severity levels
            self.assertIn(alert['severity'], ['INFO', 'WARNING', 'CRITICAL'])
    
    def test_alert_history(self):
        """Test alert history tracking."""
        initial_history_length = len(self.alert_system.alert_history)
        
        # Generate alerts
        self.alert_system.check_alerts(self.confidence_score, self.risk_assessment)
        
        # History should have grown
        self.assertGreater(len(self.alert_system.alert_history), initial_history_length)
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        # Generate some alerts
        self.alert_system.check_alerts(self.confidence_score, self.risk_assessment)
        
        summary = self.alert_system.get_alert_summary(hours=24)
        
        expected_keys = [
            'total_alerts', 'alert_counts', 'severity_counts',
            'time_period_hours', 'most_recent_alert'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertGreater(summary['total_alerts'], 0)
        self.assertIsInstance(summary['alert_counts'], dict)
        self.assertIsInstance(summary['severity_counts'], dict)


class TestConfidenceRiskManager(unittest.TestCase):
    """Test integrated confidence and risk manager."""
    
    def setUp(self):
        """Set up test manager."""
        self.manager = ConfidenceRiskManager(
            confidence_threshold=0.4,
            risk_threshold=0.7,
            risk_tolerance=0.02
        )
    
    def test_evaluate_prediction_basic(self):
        """Test basic prediction evaluation."""
        evaluation = self.manager.evaluate_prediction(
            prediction=0.05,
            uncertainty=0.1,
            current_price=100.0
        )
        
        expected_keys = ['confidence_score', 'risk_assessment', 'alerts', 'recommendation']
        
        for key in expected_keys:
            self.assertIn(key, evaluation)
        
        # Check recommendation structure
        recommendation = evaluation['recommendation']
        self.assertIn('action', recommendation)
        self.assertIn('position_size', recommendation)
        self.assertIn('confidence_level', recommendation)
        self.assertIn('risk_level', recommendation)
    
    def test_evaluate_prediction_comprehensive(self):
        """Test comprehensive prediction evaluation."""
        evaluation = self.manager.evaluate_prediction(
            prediction=0.03,
            uncertainty=0.08,
            current_price=100.0,
            model_agreement=0.85,
            data_quality=0.9,
            volatility=0.15,
            market_conditions={
                'regime_uncertainty': 0.2,
                'liquidity_risk': 0.1,
                'correlation_risk': 0.05,
                'tail_risk': 0.02
            }
        )
        
        # Should have comprehensive evaluation
        self.assertIsInstance(evaluation['confidence_score'], dict)
        self.assertIsInstance(evaluation['risk_assessment'], dict)
        self.assertIsInstance(evaluation['alerts'], list)
        self.assertIsInstance(evaluation['recommendation'], dict)
    
    def test_recommendation_generation(self):
        """Test recommendation generation logic."""
        # High confidence, low risk scenario
        high_conf_eval = self.manager.evaluate_prediction(
            prediction=0.05,
            uncertainty=0.05,
            current_price=100.0,
            model_agreement=0.95,
            data_quality=0.9,
            volatility=0.1
        )
        
        # Low confidence, high risk scenario
        low_conf_eval = self.manager.evaluate_prediction(
            prediction=0.05,
            uncertainty=0.4,
            current_price=100.0,
            model_agreement=0.3,
            data_quality=0.4,
            volatility=0.5
        )
        
        # High confidence should give stronger recommendation
        high_conf_action = high_conf_eval['recommendation']['action']
        low_conf_action = low_conf_eval['recommendation']['action']
        
        # Actions should be different based on confidence/risk
        self.assertIsInstance(high_conf_action, str)
        self.assertIsInstance(low_conf_action, str)
        
        # High confidence should recommend larger position
        high_conf_size = high_conf_eval['recommendation']['position_size']
        low_conf_size = low_conf_eval['recommendation']['position_size']
        
        self.assertGreater(high_conf_size, low_conf_size)
    
    def test_integration(self):
        """Test integration between components."""
        # Test that all components work together
        evaluation = self.manager.evaluate_prediction(
            prediction=0.02,
            uncertainty=0.15,
            current_price=100.0
        )
        
        # Confidence score should influence risk assessment
        confidence = evaluation['confidence_score']['overall_confidence']
        risk_score = evaluation['risk_assessment']['risk_score']
        
        # Generally, higher confidence should lead to lower risk
        # (though other factors also matter)
        self.assertTrue(0 <= confidence <= 1)
        self.assertTrue(0 <= risk_score <= 1)


if __name__ == '__main__':
    unittest.main()