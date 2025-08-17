"""
Unit tests for market regime analysis.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enhanced_timeseries.backtesting.regime_analysis import (
    MarketRegimeDetector, RegimePerformanceTracker, AdaptiveRegimeModel,
    MarketRegime, VolatilityRegime, TrendRegime, RegimeClassification, RegimePerformance
)


class TestMarketRegimeDetector(unittest.TestCase):
    """Test market regime detector."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic market data with different regimes
        np.random.seed(42)
        
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Create different market phases
        bull_phase = np.random.normal(0.0008, 0.015, n_days // 3)  # Bull market
        bear_phase = np.random.normal(-0.0005, 0.025, n_days // 3)  # Bear market
        sideways_phase = np.random.normal(0.0001, 0.01, n_days - 2 * (n_days // 3))  # Sideways
        
        returns = np.concatenate([bull_phase, bear_phase, sideways_phase])
        prices = 100 * np.cumprod(1 + returns)
        
        self.data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.3, n_days)
        }, index=dates)
        
        self.detector = MarketRegimeDetector(lookback_window=100, min_regime_length=10)
    
    def test_detector_creation(self):
        """Test detector creation."""
        self.assertEqual(self.detector.lookback_window, 100)
        self.assertEqual(self.detector.min_regime_length, 10)
        self.assertEqual(len(self.detector.regime_history), 0)
    
    def test_detect_regimes(self):
        """Test regime detection."""
        classifications = self.detector.detect_regimes(self.data, 'Close')
        
        self.assertEqual(len(classifications), len(self.data))
        
        # Check that all classifications are valid
        for classification in classifications:
            self.assertIsInstance(classification, RegimeClassification)
            self.assertIsInstance(classification.market_regime, MarketRegime)
            self.assertIsInstance(classification.volatility_regime, VolatilityRegime)
            self.assertIsInstance(classification.trend_regime, TrendRegime)
            self.assertTrue(0 <= classification.confidence <= 1)
            self.assertIsInstance(classification.regime_features, dict)
    
    def test_detect_market_regime(self):
        """Test market regime detection."""
        # Test bull market scenario
        bull_prices = pd.Series([100, 105, 110, 115, 120])
        bull_returns = bull_prices.pct_change().fillna(0)
        
        regime, confidence = self.detector._detect_market_regime(bull_prices, bull_returns)
        
        self.assertIsInstance(regime, MarketRegime)
        self.assertTrue(0 <= confidence <= 1)
        
        # Test bear market scenario
        bear_prices = pd.Series([100, 95, 90, 85, 80])
        bear_returns = bear_prices.pct_change().fillna(0)
        
        regime, confidence = self.detector._detect_market_regime(bear_prices, bear_returns)
        
        self.assertIsInstance(regime, MarketRegime)
        self.assertTrue(0 <= confidence <= 1)
    
    def test_detect_volatility_regime(self):
        """Test volatility regime detection."""
        # Low volatility scenario
        low_vol_returns = pd.Series(np.random.normal(0, 0.005, 100))
        vol_regime = self.detector._detect_volatility_regime(low_vol_returns)
        
        self.assertIsInstance(vol_regime, VolatilityRegime)
        
        # High volatility scenario
        high_vol_returns = pd.Series(np.random.normal(0, 0.05, 100))
        vol_regime = self.detector._detect_volatility_regime(high_vol_returns)
        
        self.assertIsInstance(vol_regime, VolatilityRegime)
    
    def test_detect_trend_regime(self):
        """Test trend regime detection."""
        # Strong uptrend
        uptrend_prices = pd.Series(np.exp(np.linspace(4.6, 5.0, 100)))  # Exponential growth
        trend_regime = self.detector._detect_trend_regime(uptrend_prices)
        
        self.assertIsInstance(trend_regime, TrendRegime)
        
        # Strong downtrend
        downtrend_prices = pd.Series(np.exp(np.linspace(5.0, 4.6, 100)))  # Exponential decay
        trend_regime = self.detector._detect_trend_regime(downtrend_prices)
        
        self.assertIsInstance(trend_regime, TrendRegime)
        
        # Sideways
        sideways_prices = pd.Series(100 + np.random.normal(0, 2, 100))
        trend_regime = self.detector._detect_trend_regime(sideways_prices)
        
        self.assertIsInstance(trend_regime, TrendRegime)
    
    def test_calculate_regime_features(self):
        """Test regime feature calculation."""
        prices = pd.Series([100, 102, 101, 105, 103, 108, 106])
        returns = prices.pct_change().fillna(0)
        
        features = self.detector._calculate_regime_features(prices, returns)
        
        expected_features = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'current_drawdown', 'skewness', 'kurtosis',
            'momentum_1m', 'momentum_3m', 'momentum_6m', 'volatility_percentile',
            'trend_strength'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            self.assertTrue(np.isfinite(features[feature]))
    
    def test_smooth_regime_transitions(self):
        """Test regime transition smoothing."""
        # Create classifications with rapid regime changes
        classifications = []
        regimes = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.BULL, MarketRegime.BEAR]
        
        for i, regime in enumerate(regimes * 10):  # Repeat pattern
            classification = RegimeClassification(
                timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                market_regime=regime,
                volatility_regime=VolatilityRegime.NORMAL,
                trend_regime=TrendRegime.SIDEWAYS,
                confidence=0.7,
                regime_features={}
            )
            classifications.append(classification)
        
        smoothed = self.detector._smooth_regime_transitions(classifications)
        
        self.assertEqual(len(smoothed), len(classifications))
        
        # Check that smoothing was applied
        for classification in smoothed:
            self.assertIsInstance(classification, RegimeClassification)
    
    def test_regime_classification_serialization(self):
        """Test regime classification serialization."""
        classification = RegimeClassification(
            timestamp=pd.Timestamp('2020-01-01'),
            market_regime=MarketRegime.BULL,
            volatility_regime=VolatilityRegime.NORMAL,
            trend_regime=TrendRegime.STRONG_UPTREND,
            confidence=0.8,
            regime_features={'volatility': 0.15, 'momentum': 0.05}
        )
        
        classification_dict = classification.to_dict()
        
        expected_keys = [
            'timestamp', 'market_regime', 'volatility_regime', 
            'trend_regime', 'confidence', 'regime_features'
        ]
        
        for key in expected_keys:
            self.assertIn(key, classification_dict)
        
        self.assertEqual(classification_dict['market_regime'], 'bull')
        self.assertEqual(classification_dict['confidence'], 0.8)


class TestRegimePerformanceTracker(unittest.TestCase):
    """Test regime performance tracker."""
    
    def setUp(self):
        """Set up test tracker."""
        self.tracker = RegimePerformanceTracker()
        
        # Create sample regime classifications
        self.bull_classification = RegimeClassification(
            timestamp=pd.Timestamp('2020-01-01'),
            market_regime=MarketRegime.BULL,
            volatility_regime=VolatilityRegime.NORMAL,
            trend_regime=TrendRegime.STRONG_UPTREND,
            confidence=0.8,
            regime_features={}
        )
        
        self.bear_classification = RegimeClassification(
            timestamp=pd.Timestamp('2020-06-01'),
            market_regime=MarketRegime.BEAR,
            volatility_regime=VolatilityRegime.HIGH,
            trend_regime=TrendRegime.STRONG_DOWNTREND,
            confidence=0.9,
            regime_features={}
        )
    
    def test_tracker_creation(self):
        """Test tracker creation."""
        self.assertEqual(len(self.tracker.regime_data), 0)
        self.assertEqual(len(self.tracker.performance_history), 0)
    
    def test_add_prediction(self):
        """Test adding predictions."""
        # Add bull market prediction
        self.tracker.add_prediction(
            timestamp=pd.Timestamp('2020-01-01'),
            prediction=0.05,
            actual=0.04,
            regime_classification=self.bull_classification
        )
        
        # Add bear market prediction
        self.tracker.add_prediction(
            timestamp=pd.Timestamp('2020-06-01'),
            prediction=-0.03,
            actual=-0.02,
            regime_classification=self.bear_classification
        )
        
        self.assertEqual(len(self.tracker.regime_data), 2)
        self.assertEqual(len(self.tracker.performance_history), 2)
        
        # Check bull market data
        bull_data = self.tracker.regime_data[MarketRegime.BULL]
        self.assertEqual(len(bull_data['predictions']), 1)
        self.assertEqual(bull_data['predictions'][0], 0.05)
        self.assertEqual(bull_data['actuals'][0], 0.04)
    
    def test_calculate_regime_performance(self):
        """Test regime performance calculation."""
        # Add multiple predictions for each regime
        for i in range(10):
            # Bull market predictions (generally accurate)
            self.tracker.add_prediction(
                timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                prediction=0.02 + np.random.normal(0, 0.005),
                actual=0.02 + np.random.normal(0, 0.01),
                regime_classification=self.bull_classification
            )
            
            # Bear market predictions (less accurate)
            self.tracker.add_prediction(
                timestamp=pd.Timestamp('2020-06-01') + pd.Timedelta(days=i),
                prediction=-0.01 + np.random.normal(0, 0.01),
                actual=-0.01 + np.random.normal(0, 0.02),
                regime_classification=self.bear_classification
            )
        
        performance = self.tracker.calculate_regime_performance()
        
        self.assertIn(MarketRegime.BULL, performance)
        self.assertIn(MarketRegime.BEAR, performance)
        
        # Check bull market performance
        bull_perf = performance[MarketRegime.BULL]
        self.assertIsInstance(bull_perf, RegimePerformance)
        self.assertEqual(bull_perf.regime, MarketRegime.BULL)
        self.assertEqual(bull_perf.total_predictions, 10)
        self.assertGreater(bull_perf.mae, 0)
        self.assertTrue(0 <= bull_perf.directional_accuracy <= 100)
        self.assertTrue(0 <= bull_perf.win_rate <= 1)
    
    def test_get_regime_distribution(self):
        """Test regime distribution calculation."""
        # Add predictions for different regimes
        regimes_and_counts = [
            (MarketRegime.BULL, 5),
            (MarketRegime.BEAR, 3),
            (MarketRegime.SIDEWAYS, 2)
        ]
        
        for regime, count in regimes_and_counts:
            classification = RegimeClassification(
                timestamp=pd.Timestamp('2020-01-01'),
                market_regime=regime,
                volatility_regime=VolatilityRegime.NORMAL,
                trend_regime=TrendRegime.SIDEWAYS,
                confidence=0.7,
                regime_features={}
            )
            
            for i in range(count):
                self.tracker.add_prediction(
                    timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                    prediction=0.01,
                    actual=0.01,
                    regime_classification=classification
                )
        
        distribution = self.tracker.get_regime_distribution()
        
        total_predictions = sum(count for _, count in regimes_and_counts)
        
        self.assertAlmostEqual(distribution['bull'], 5 / total_predictions * 100)
        self.assertAlmostEqual(distribution['bear'], 3 / total_predictions * 100)
        self.assertAlmostEqual(distribution['sideways'], 2 / total_predictions * 100)
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Add some predictions
        for i in range(5):
            self.tracker.add_prediction(
                timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                prediction=0.02,
                actual=0.015,
                regime_classification=self.bull_classification
            )
        
        summary = self.tracker.get_performance_summary()
        
        expected_keys = [
            'regime_performance', 'regime_distribution', 'overall_mae',
            'overall_directional_accuracy', 'total_predictions', 'n_regimes_encountered',
            'best_regime', 'worst_regime'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary['total_predictions'], 5)
        self.assertEqual(summary['n_regimes_encountered'], 1)
        self.assertIsNotNone(summary['best_regime'])
    
    def test_get_regime_transitions(self):
        """Test regime transition analysis."""
        # Create sequence with regime transitions
        regimes = [MarketRegime.BULL, MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.BEAR, MarketRegime.SIDEWAYS]
        
        for i, regime in enumerate(regimes):
            classification = RegimeClassification(
                timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                market_regime=regime,
                volatility_regime=VolatilityRegime.NORMAL,
                trend_regime=TrendRegime.SIDEWAYS,
                confidence=0.7,
                regime_features={}
            )
            
            self.tracker.add_prediction(
                timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                prediction=0.01,
                actual=0.01,
                regime_classification=classification
            )
        
        transitions = self.tracker.get_regime_transitions()
        
        # Should detect transitions from BULL to BEAR and BEAR to SIDEWAYS
        self.assertGreaterEqual(len(transitions), 0)
        
        for transition in transitions:
            expected_keys = [
                'timestamp', 'from_regime', 'to_regime', 
                'pre_transition_mae', 'post_transition_mae', 'performance_change'
            ]
            
            for key in expected_keys:
                self.assertIn(key, transition)


class MockPredictor:
    """Mock predictor for testing adaptive regime model."""
    
    def __init__(self, prediction_value=0.01):
        self.prediction_value = prediction_value
        self.is_trained = False
    
    def train(self, data):
        self.is_trained = True
        return {'mae': 0.01}
    
    def predict_with_uncertainty(self, data):
        return self.prediction_value, 0.1
    
    def predict(self, data):
        return self.prediction_value


class TestAdaptiveRegimeModel(unittest.TestCase):
    """Test adaptive regime model."""
    
    def setUp(self):
        """Set up test model."""
        self.predictor_factory = lambda: MockPredictor()
        self.detector = MarketRegimeDetector(lookback_window=50)
        self.adaptive_model = AdaptiveRegimeModel(self.predictor_factory, self.detector)
        
        # Create sample data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        self.data = pd.DataFrame({
            'Close': 100 + np.random.randn(len(dates)),
            'target': np.random.randn(len(dates)) * 0.02
        }, index=dates)
    
    def test_adaptive_model_creation(self):
        """Test adaptive model creation."""
        self.assertEqual(len(self.adaptive_model.regime_models), 0)
        self.assertEqual(self.adaptive_model.current_regime, MarketRegime.UNKNOWN)
    
    def test_train_regime_specific_models(self):
        """Test training regime-specific models."""
        # Create regime classifications
        classifications = []
        
        # Create enough samples for each regime
        for i in range(200):
            if i < 100:
                regime = MarketRegime.BULL
            else:
                regime = MarketRegime.BEAR
            
            classification = RegimeClassification(
                timestamp=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i % 365),
                market_regime=regime,
                volatility_regime=VolatilityRegime.NORMAL,
                trend_regime=TrendRegime.SIDEWAYS,
                confidence=0.7,
                regime_features={}
            )
            classifications.append(classification)
        
        # Extend data to match classifications
        extended_data = pd.concat([self.data] * (len(classifications) // len(self.data) + 1))
        extended_data = extended_data.head(len(classifications))
        
        self.adaptive_model.train_regime_specific_models(extended_data, classifications)
        
        # Should have trained models for regimes with sufficient data
        self.assertGreater(len(self.adaptive_model.regime_models), 0)
    
    def test_predict_with_regime_adaptation(self):
        """Test prediction with regime adaptation."""
        # Train a model for bull regime
        self.adaptive_model.regime_models[MarketRegime.BULL] = MockPredictor(0.05)
        
        # Test prediction with bull regime
        prediction, uncertainty = self.adaptive_model.predict_with_regime_adaptation(
            self.data.head(1), MarketRegime.BULL
        )
        
        self.assertEqual(prediction, 0.05)
        self.assertEqual(uncertainty, 0.1)
        self.assertEqual(self.adaptive_model.current_regime, MarketRegime.BULL)
        
        # Test prediction with unknown regime (should use fallback)
        prediction, uncertainty = self.adaptive_model.predict_with_regime_adaptation(
            self.data.head(1), MarketRegime.CRISIS
        )
        
        # Should use bull model as fallback
        self.assertEqual(prediction, 0.05)
    
    def test_export_regime_analysis(self):
        """Test regime analysis export."""
        import tempfile
        import os
        import json
        
        # Track some performance first
        self.tracker.track_performance(
            self.predictions, self.actuals, self.timestamps, self.market_data, 'Close'
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.tracker.export_regime_analysis(temp_filepath)
            
            # Check file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            expected_keys = ['regime_performance', 'regime_summary', 'regime_transitions']
            for key in expected_keys:
                self.assertIn(key, data)
        
        finally:
            # Clean up
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_regime_performance_serialization(self):
        """Test regime performance serialization."""
        performance = RegimePerformance(
            regime_type=MarketRegime.BULL,
            n_periods=1,
            total_predictions=100,
            mae=0.01,
            rmse=0.015,
            directional_accuracy=65.0,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            hit_rate=0.65,
            profit_factor=1.5,
            avg_return=0.001,
            return_volatility=0.02
        )
        
        perf_dict = performance.to_dict()
        
        expected_keys = [
            'regime_type', 'n_periods', 'total_predictions', 'mae', 'rmse',
            'directional_accuracy', 'sharpe_ratio', 'max_drawdown', 'hit_rate',
            'profit_factor', 'avg_return', 'return_volatility'
        ]
        
        for key in expected_keys:
            self.assertIn(key, perf_dict)
        
        self.assertEqual(perf_dict['regime_type'], MarketRegime.BULL)
        self.assertEqual(perf_dict['total_predictions'], 100)


if __name__ == '__main__':
    unittest.main()