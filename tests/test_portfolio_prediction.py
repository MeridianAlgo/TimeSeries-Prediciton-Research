"""
Unit tests for portfolio prediction and asset ranking system.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from enhanced_timeseries.multi_asset.portfolio_prediction import (
    AssetPrediction, PortfolioPrediction, AssetRanking,
    CovarianceEstimator, PortfolioOptimizer, AssetRankingSystem,
    PortfolioPredictionSystem
)


class TestAssetPrediction(unittest.TestCase):
    """Test AssetPrediction dataclass."""
    
    def test_asset_prediction_creation(self):
        """Test AssetPrediction creation and serialization."""
        prediction = AssetPrediction(
            symbol='AAPL',
            expected_return=0.12,
            return_std=0.20,
            confidence_score=0.85,
            prediction_horizon=252,
            model_ensemble_agreement=0.90,
            factor_exposure={'market': 0.8, 'tech': 0.9},
            sector='Technology',
            market_cap_rank=1
        )
        
        self.assertEqual(prediction.symbol, 'AAPL')
        self.assertEqual(prediction.expected_return, 0.12)
        self.assertEqual(prediction.confidence_score, 0.85)
        self.assertEqual(prediction.sector, 'Technology')
        
        # Test serialization
        pred_dict = prediction.to_dict()
        self.assertIsInstance(pred_dict, dict)
        self.assertEqual(pred_dict['symbol'], 'AAPL')
        self.assertEqual(pred_dict['expected_return'], 0.12)


class TestCovarianceEstimator(unittest.TestCase):
    """Test covariance estimation methods."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create correlated return data
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Generate factor-based returns
        market_factor = np.random.normal(0.0005, 0.02, n_days)
        
        self.returns_data = pd.DataFrame(index=dates)
        self.returns_data['ASSET_A'] = 0.8 * market_factor + np.random.normal(0, 0.01, n_days)
        self.returns_data['ASSET_B'] = 0.7 * market_factor + np.random.normal(0, 0.012, n_days)
        self.returns_data['ASSET_C'] = 0.6 * market_factor + np.random.normal(0, 0.015, n_days)
        
        self.estimator = CovarianceEstimator(method='ledoit_wolf', lookback_window=252)
    
    def test_estimator_creation(self):
        """Test covariance estimator creation."""
        self.assertEqual(self.estimator.method, 'ledoit_wolf')
        self.assertEqual(self.estimator.lookback_window, 252)
    
    def test_estimate_covariance_ledoit_wolf(self):
        """Test Ledoit-Wolf covariance estimation."""
        cov_matrix = self.estimator.estimate_covariance(self.returns_data)
        
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(cov_matrix.shape, (3, 3))
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix.values)
        self.assertTrue(np.all(eigenvalues >= -1e-8))  # Allow for small numerical errors
        
        # Check symmetry
        np.testing.assert_array_almost_equal(cov_matrix.values, cov_matrix.T.values)
    
    def test_estimate_covariance_empirical(self):
        """Test empirical covariance estimation."""
        empirical_estimator = CovarianceEstimator(method='empirical')
        cov_matrix = empirical_estimator.estimate_covariance(self.returns_data)
        
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(cov_matrix.shape, (3, 3))
    
    def test_estimate_correlation_matrix(self):
        """Test correlation matrix estimation."""
        corr_matrix = self.estimator.estimate_correlation_matrix(self.returns_data)
        
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (3, 3))
        
        # Check diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(3))
        
        # Check values are between -1 and 1
        self.assertTrue(np.all(corr_matrix.values >= -1))
        self.assertTrue(np.all(corr_matrix.values <= 1))
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        short_data = self.returns_data.head(10)
        
        cov_matrix = self.estimator.estimate_covariance(short_data)
        # Should handle gracefully
        self.assertIsInstance(cov_matrix, pd.DataFrame)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        cov_matrix = self.estimator.estimate_covariance(empty_data)
        self.assertTrue(cov_matrix.empty)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimization methods."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create expected returns and covariance matrix
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        self.expected_returns = pd.Series([0.12, 0.15, 0.10, 0.18], index=assets)
        
        # Create a realistic covariance matrix
        correlations = np.array([
            [1.0, 0.6, 0.7, 0.4],
            [0.6, 1.0, 0.5, 0.3],
            [0.7, 0.5, 1.0, 0.4],
            [0.4, 0.3, 0.4, 1.0]
        ])
        
        volatilities = np.array([0.20, 0.25, 0.18, 0.35])
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        self.covariance_matrix = pd.DataFrame(cov_matrix, index=assets, columns=assets)
        
        self.optimizer = PortfolioOptimizer(
            risk_aversion=1.0,
            max_weight=0.4,
            min_weight=0.05
        )
    
    def test_optimizer_creation(self):
        """Test portfolio optimizer creation."""
        self.assertEqual(self.optimizer.risk_aversion, 1.0)
        self.assertEqual(self.optimizer.max_weight, 0.4)
        self.assertEqual(self.optimizer.min_weight, 0.05)
    
    def test_optimize_max_sharpe(self):
        """Test maximum Sharpe ratio optimization."""
        result = self.optimizer.optimize_portfolio(
            expected_returns=self.expected_returns,
            covariance_matrix=self.covariance_matrix,
            objective='max_sharpe'
        )
        
        if result.get('optimization_success', False):
            weights = result['weights']
            
            # Check constraints
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)  # Weights sum to 1
            self.assertTrue(np.all(weights >= self.optimizer.min_weight - 1e-6))  # Min weight
            self.assertTrue(np.all(weights <= self.optimizer.max_weight + 1e-6))  # Max weight
            
            # Check that we have valid portfolio metrics
            self.assertIn('expected_return', result)
            self.assertIn('expected_volatility', result)
            self.assertIn('sharpe_ratio', result)
            
            self.assertGreater(result['expected_volatility'], 0)
            self.assertTrue(np.isfinite(result['sharpe_ratio']))
    
    def test_optimize_min_variance(self):
        """Test minimum variance optimization."""
        result = self.optimizer.optimize_portfolio(
            expected_returns=self.expected_returns,
            covariance_matrix=self.covariance_matrix,
            objective='min_variance'
        )
        
        if result.get('optimization_success', False):
            weights = result['weights']
            
            # Check constraints
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)
            self.assertTrue(np.all(weights >= self.optimizer.min_weight - 1e-6))
            self.assertTrue(np.all(weights <= self.optimizer.max_weight + 1e-6))
    
    def test_optimize_max_return(self):
        """Test maximum return optimization."""
        result = self.optimizer.optimize_portfolio(
            expected_returns=self.expected_returns,
            covariance_matrix=self.covariance_matrix,
            objective='max_return'
        )
        
        if result.get('optimization_success', False):
            weights = result['weights']
            
            # Check constraints
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)
            self.assertTrue(np.all(weights >= self.optimizer.min_weight - 1e-6))
            self.assertTrue(np.all(weights <= self.optimizer.max_weight + 1e-6))
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_returns = pd.Series(dtype=float)
        empty_cov = pd.DataFrame()
        
        result = self.optimizer.optimize_portfolio(
            expected_returns=empty_returns,
            covariance_matrix=empty_cov
        )
        
        self.assertEqual(len(result), 0)
    
    def test_misaligned_data(self):
        """Test handling of misaligned data."""
        # Returns for different assets than covariance matrix
        misaligned_returns = pd.Series([0.1, 0.12], index=['XYZ', 'ABC'])
        
        result = self.optimizer.optimize_portfolio(
            expected_returns=misaligned_returns,
            covariance_matrix=self.covariance_matrix
        )
        
        # Should handle gracefully
        self.assertEqual(len(result), 0)


class TestAssetRankingSystem(unittest.TestCase):
    """Test asset ranking system."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create asset predictions
        self.asset_predictions = {
            'AAPL': AssetPrediction(
                symbol='AAPL',
                expected_return=0.12,
                return_std=0.20,
                confidence_score=0.85,
                prediction_horizon=252,
                model_ensemble_agreement=0.90,
                factor_exposure={'market': 0.8, 'tech': 0.9},
                sector='Technology'
            ),
            'GOOGL': AssetPrediction(
                symbol='GOOGL',
                expected_return=0.15,
                return_std=0.25,
                confidence_score=0.75,
                prediction_horizon=252,
                model_ensemble_agreement=0.85,
                factor_exposure={'market': 0.7, 'tech': 0.8},
                sector='Technology'
            ),
            'JPM': AssetPrediction(
                symbol='JPM',
                expected_return=0.08,
                return_std=0.18,
                confidence_score=0.90,
                prediction_horizon=252,
                model_ensemble_agreement=0.95,
                factor_exposure={'market': 0.6, 'financial': 0.8},
                sector='Financial'
            )
        }
        
        # Create historical data
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        self.historical_data = pd.DataFrame(index=dates)
        
        # Generate realistic price data
        for symbol in self.asset_predictions.keys():
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = 100 * np.cumprod(1 + returns)
            self.historical_data[symbol] = prices
        
        self.ranking_system = AssetRankingSystem()
    
    def test_ranking_system_creation(self):
        """Test asset ranking system creation."""
        self.assertIsInstance(self.ranking_system.ranking_factors, dict)
        self.assertEqual(self.ranking_system.lookback_window, 252)
    
    def test_rank_assets(self):
        """Test asset ranking functionality."""
        rankings = self.ranking_system.rank_assets(
            self.asset_predictions, 
            self.historical_data
        )
        
        self.assertIsInstance(rankings, list)
        self.assertEqual(len(rankings), 3)
        
        # Check ranking structure
        for ranking in rankings:
            self.assertIsInstance(ranking, AssetRanking)
            self.assertIn(ranking.symbol, self.asset_predictions.keys())
            self.assertTrue(1 <= ranking.overall_rank <= 3)
            self.assertTrue(0 <= ranking.composite_score <= 1)
            self.assertIsInstance(ranking.ranking_factors, dict)
        
        # Check that rankings are sorted by composite score
        scores = [ranking.composite_score for ranking in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_calculate_return_scores(self):
        """Test return score calculation."""
        scores = self.ranking_system._calculate_return_scores(self.asset_predictions)
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        
        # All scores should be between 0 and 1
        for score in scores.values():
            self.assertTrue(0 <= score <= 1)
        
        # GOOGL should have highest return score (highest expected return)
        self.assertEqual(max(scores, key=scores.get), 'GOOGL')
    
    def test_calculate_risk_adjusted_scores(self):
        """Test risk-adjusted return score calculation."""
        scores = self.ranking_system._calculate_risk_adjusted_scores(self.asset_predictions)
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        
        # All scores should be between 0 and 1
        for score in scores.values():
            self.assertTrue(0 <= score <= 1)
    
    def test_calculate_momentum_scores(self):
        """Test momentum score calculation."""
        scores = self.ranking_system._calculate_momentum_scores(
            self.asset_predictions, self.historical_data
        )
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        
        # All scores should be between 0 and 1
        for score in scores.values():
            self.assertTrue(0 <= score <= 1)
    
    def test_calculate_quality_scores(self):
        """Test quality score calculation."""
        scores = self.ranking_system._calculate_quality_scores(
            self.asset_predictions, self.historical_data, None
        )
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        
        # All scores should be between 0 and 1
        for score in scores.values():
            self.assertTrue(0 <= score <= 1)
    
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        rankings = self.ranking_system.rank_assets({}, self.historical_data)
        
        self.assertEqual(len(rankings), 0)
    
    def test_insufficient_historical_data(self):
        """Test handling of insufficient historical data."""
        short_data = self.historical_data.head(10)
        
        rankings = self.ranking_system.rank_assets(
            self.asset_predictions, short_data
        )
        
        # Should still return rankings
        self.assertEqual(len(rankings), 3)


class TestPortfolioPredictionSystem(unittest.TestCase):
    """Test the main portfolio prediction system."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create asset predictions
        self.asset_predictions = {
            'AAPL': AssetPrediction(
                symbol='AAPL',
                expected_return=0.12,
                return_std=0.20,
                confidence_score=0.85,
                prediction_horizon=252,
                model_ensemble_agreement=0.90,
                factor_exposure={'market': 0.8, 'tech': 0.9},
                sector='Technology'
            ),
            'GOOGL': AssetPrediction(
                symbol='GOOGL',
                expected_return=0.15,
                return_std=0.25,
                confidence_score=0.75,
                prediction_horizon=252,
                model_ensemble_agreement=0.85,
                factor_exposure={'market': 0.7, 'tech': 0.8},
                sector='Technology'
            ),
            'JPM': AssetPrediction(
                symbol='JPM',
                expected_return=0.08,
                return_std=0.18,
                confidence_score=0.90,
                prediction_horizon=252,
                model_ensemble_agreement=0.95,
                factor_exposure={'market': 0.6, 'financial': 0.8},
                sector='Financial'
            ),
            'XOM': AssetPrediction(
                symbol='XOM',
                expected_return=0.06,
                return_std=0.22,
                confidence_score=0.70,
                prediction_horizon=252,
                model_ensemble_agreement=0.80,
                factor_exposure={'market': 0.4, 'energy': 0.7},
                sector='Energy'
            )
        }
        
        # Create historical data
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        self.historical_data = pd.DataFrame(index=dates)
        
        # Generate correlated return data
        market_factor = np.random.normal(0.0005, 0.02, n_days)
        
        for symbol, prediction in self.asset_predictions.items():
            # Create returns based on factor exposures
            market_exposure = prediction.factor_exposure.get('market', 0.5)
            idiosyncratic = np.random.normal(0, prediction.return_std * 0.5, n_days)
            
            returns = market_exposure * market_factor + idiosyncratic
            prices = 100 * np.cumprod(1 + returns)
            self.historical_data[symbol] = prices
        
        self.system = PortfolioPredictionSystem(
            covariance_method='ledoit_wolf',
            optimization_objective='max_sharpe',
            max_position_size=0.4,
            min_position_size=0.05
        )
    
    def test_system_creation(self):
        """Test portfolio prediction system creation."""
        self.assertIsNotNone(self.system.covariance_estimator)
        self.assertIsNotNone(self.system.portfolio_optimizer)
        self.assertIsNotNone(self.system.ranking_system)
        self.assertEqual(self.system.optimization_objective, 'max_sharpe')
    
    def test_generate_portfolio_prediction(self):
        """Test comprehensive portfolio prediction generation."""
        result = self.system.generate_portfolio_prediction(
            asset_predictions=self.asset_predictions,
            historical_data=self.historical_data
        )
        
        self.assertIsInstance(result, dict)
        
        # Check main sections
        expected_keys = [
            'timestamp', 'n_assets', 'optimization_objective', 'asset_rankings'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['n_assets'], 4)
        self.assertEqual(result['optimization_objective'], 'max_sharpe')
        
        # Check asset rankings
        rankings = result['asset_rankings']
        self.assertEqual(len(rankings), 4)
        
        for ranking in rankings:
            self.assertIn('symbol', ranking)
            self.assertIn('overall_rank', ranking)
            self.assertIn('composite_score', ranking)
        
        # Check if optimization succeeded
        if 'portfolio_optimization' in result:
            opt_result = result['portfolio_optimization']
            self.assertIn('optimal_weights', opt_result)
            self.assertIn('expected_return', opt_result)
            self.assertIn('expected_volatility', opt_result)
            
            # Check weights sum to 1
            weights = opt_result['optimal_weights']
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # Check portfolio prediction
        if 'portfolio_prediction' in result:
            portfolio_pred = result['portfolio_prediction']
            
            expected_portfolio_keys = [
                'expected_return', 'expected_volatility', 'sharpe_ratio',
                'max_drawdown_estimate', 'var_95', 'cvar_95'
            ]
            
            for key in expected_portfolio_keys:
                self.assertIn(key, portfolio_pred)
                self.assertTrue(np.isfinite(portfolio_pred[key]))
    
    def test_generate_portfolio_prediction_empty_data(self):
        """Test portfolio prediction with empty data."""
        result = self.system.generate_portfolio_prediction(
            asset_predictions={},
            historical_data=pd.DataFrame()
        )
        
        self.assertIn('error', result)
    
    def test_get_rebalancing_recommendations(self):
        """Test rebalancing recommendations."""
        current_weights = pd.Series([0.3, 0.3, 0.2, 0.2], 
                                  index=['AAPL', 'GOOGL', 'JPM', 'XOM'])
        target_weights = pd.Series([0.4, 0.25, 0.25, 0.1], 
                                 index=['AAPL', 'GOOGL', 'JPM', 'XOM'])
        
        recommendations = self.system.get_rebalancing_recommendations(
            current_weights, target_weights
        )
        
        self.assertIsInstance(recommendations, dict)
        
        expected_keys = [
            'rebalancing_needed', 'weight_differences', 
            'assets_to_buy', 'assets_to_sell', 'total_turnover'
        ]
        
        for key in expected_keys:
            self.assertIn(key, recommendations)
        
        self.assertIsInstance(recommendations['rebalancing_needed'], bool)
        self.assertGreaterEqual(recommendations['total_turnover'], 0)
    
    def test_export_portfolio_analysis(self):
        """Test portfolio analysis export."""
        import json
        
        # Generate a prediction first
        self.system.generate_portfolio_prediction(
            asset_predictions=self.asset_predictions,
            historical_data=self.historical_data
        )
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.system.export_portfolio_analysis(temp_filepath)
            
            # Check file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, dict)
            self.assertIn('prediction_history', data)
            self.assertIn('system_configuration', data)
            self.assertIn('export_timestamp', data)
        
        finally:
            # Clean up
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_prediction_history_management(self):
        """Test prediction history management."""
        initial_history_len = len(self.system.prediction_history)
        
        # Generate multiple predictions
        for _ in range(3):
            self.system.generate_portfolio_prediction(
                asset_predictions=self.asset_predictions,
                historical_data=self.historical_data
            )
        
        # Check history is updated
        self.assertEqual(len(self.system.prediction_history), initial_history_len + 3)


if __name__ == '__main__':
    unittest.main()