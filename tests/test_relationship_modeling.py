"""
Unit tests for cross-asset relationship modeling.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from enhanced_timeseries.multi_asset.relationship_modeling import (
    CorrelationAnalyzer, SectorClassifier, FactorAnalyzer, AdaptiveWeightingSystem,
    CrossAssetRelationshipModeler, CorrelationMetrics, SectorInfo, FactorExposure
)


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test correlation analyzer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create correlated asset returns
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Generate base factors
        market_factor = np.random.normal(0.0005, 0.02, n_days)
        sector_factor = np.random.normal(0.0002, 0.015, n_days)
        
        # Create assets with different exposures to factors
        self.returns_data = pd.DataFrame(index=dates)
        
        # Tech stocks (high correlation)
        self.returns_data['TECH_A'] = 0.8 * market_factor + 0.6 * sector_factor + np.random.normal(0, 0.01, n_days)
        self.returns_data['TECH_B'] = 0.7 * market_factor + 0.7 * sector_factor + np.random.normal(0, 0.01, n_days)
        
        # Financial stocks (medium correlation with tech)
        fin_factor = np.random.normal(0.0001, 0.018, n_days)
        self.returns_data['FIN_A'] = 0.6 * market_factor + 0.5 * fin_factor + np.random.normal(0, 0.012, n_days)
        self.returns_data['FIN_B'] = 0.5 * market_factor + 0.6 * fin_factor + np.random.normal(0, 0.012, n_days)
        
        # Utility (low correlation)
        self.returns_data['UTIL_A'] = 0.3 * market_factor + np.random.normal(0, 0.008, n_days)
        
        self.analyzer = CorrelationAnalyzer(rolling_window=63, min_periods=30)
    
    def test_analyzer_creation(self):
        """Test correlation analyzer creation."""
        self.assertEqual(self.analyzer.rolling_window, 63)
        self.assertEqual(self.analyzer.min_periods, 30)
        self.assertEqual(len(self.analyzer.correlation_history), 0)
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        corr_matrix = self.analyzer.calculate_correlation_matrix(self.returns_data)
        
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (5, 5))
        
        # Check diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(5))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.T.values)
        
        # Tech stocks should be highly correlated
        tech_corr = corr_matrix.loc['TECH_A', 'TECH_B']
        self.assertGreater(tech_corr, 0.5)
    
    def test_calculate_correlation_matrix_methods(self):
        """Test different correlation methods."""
        for method in ['pearson', 'spearman', 'kendall']:
            corr_matrix = self.analyzer.calculate_correlation_matrix(self.returns_data, method=method)
            self.assertIsInstance(corr_matrix, pd.DataFrame)
            self.assertEqual(corr_matrix.shape, (5, 5))
    
    def test_calculate_rolling_correlations(self):
        """Test rolling correlation calculation."""
        rolling_corrs = self.analyzer.calculate_rolling_correlations(self.returns_data)
        
        self.assertIsInstance(rolling_corrs, dict)
        self.assertGreater(len(rolling_corrs), 0)
        
        # Check that we have correlations for asset pairs
        for (asset1, asset2), corr_series in rolling_corrs.items():
            self.assertIsInstance(corr_series, pd.Series)
            self.assertGreater(len(corr_series), 0)
            self.assertIn(asset1, self.returns_data.columns)
            self.assertIn(asset2, self.returns_data.columns)
    
    def test_analyze_correlation_pair(self):
        """Test pairwise correlation analysis."""
        asset1_data = self.returns_data['TECH_A']
        asset2_data = self.returns_data['TECH_B']
        
        metrics = self.analyzer.analyze_correlation_pair(asset1_data, asset2_data)
        
        self.assertIsInstance(metrics, CorrelationMetrics)
        self.assertTrue(-1 <= metrics.pearson_correlation <= 1)
        self.assertTrue(-1 <= metrics.spearman_correlation <= 1)
        self.assertTrue(-1 <= metrics.kendall_correlation <= 1)
        self.assertTrue(0 <= metrics.correlation_stability <= 1)
        self.assertGreaterEqual(metrics.rolling_correlation_std, 0)
    
    def test_analyze_correlation_pair_insufficient_data(self):
        """Test correlation analysis with insufficient data."""
        short_data = self.returns_data.head(10)
        asset1_data = short_data['TECH_A']
        asset2_data = short_data['TECH_B']
        
        metrics = self.analyzer.analyze_correlation_pair(asset1_data, asset2_data)
        
        # Should return default metrics
        self.assertEqual(metrics.pearson_correlation, 0.0)
        self.assertEqual(metrics.correlation_stability, 0.0)
    
    def test_detect_correlation_regimes(self):
        """Test correlation regime detection."""
        rolling_corrs = self.analyzer.calculate_rolling_correlations(self.returns_data)
        
        if rolling_corrs:  # Only test if we have rolling correlations
            regimes = self.analyzer.detect_correlation_regimes(rolling_corrs, n_regimes=3)
            
            self.assertIsInstance(regimes, dict)
            
            for pair, regime_info in regimes.items():
                self.assertIn('regimes', regime_info)
                self.assertIn('regime_quality', regime_info)
                self.assertIn('n_regimes', regime_info)
                self.assertTrue(0 <= regime_info['regime_quality'] <= 1)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        corr_matrix = self.analyzer.calculate_correlation_matrix(empty_data)
        self.assertTrue(corr_matrix.empty)
        
        rolling_corrs = self.analyzer.calculate_rolling_correlations(empty_data)
        self.assertEqual(len(rolling_corrs), 0)


class TestSectorClassifier(unittest.TestCase):
    """Test sector classifier."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sector-specific returns
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        self.returns_data = pd.DataFrame(index=dates)
        
        # Technology sector (high correlation within sector)
        tech_factor = np.random.normal(0.001, 0.025, n_days)
        self.returns_data['AAPL'] = tech_factor + np.random.normal(0, 0.01, n_days)
        self.returns_data['GOOGL'] = tech_factor + np.random.normal(0, 0.012, n_days)
        self.returns_data['MSFT'] = tech_factor + np.random.normal(0, 0.011, n_days)
        
        # Financial sector
        fin_factor = np.random.normal(0.0005, 0.02, n_days)
        self.returns_data['JPM'] = fin_factor + np.random.normal(0, 0.015, n_days)
        self.returns_data['BAC'] = fin_factor + np.random.normal(0, 0.016, n_days)
        
        # Utility sector (lower volatility)
        util_factor = np.random.normal(0.0002, 0.012, n_days)
        self.returns_data['XEL'] = util_factor + np.random.normal(0, 0.008, n_days)
        
        self.classifier = SectorClassifier(min_cluster_size=2, max_clusters=5)
    
    def test_classifier_creation(self):
        """Test sector classifier creation."""
        self.assertEqual(self.classifier.min_cluster_size, 2)
        self.assertEqual(self.classifier.max_clusters, 5)
    
    def test_classify_sectors_hierarchical(self):
        """Test hierarchical sector classification."""
        sectors = self.classifier.classify_sectors(self.returns_data, method='hierarchical')
        
        self.assertIsInstance(sectors, dict)
        
        # Check that all assets are classified
        for asset in self.returns_data.columns:
            if asset in sectors:
                sector_info = sectors[asset]
                self.assertIsInstance(sector_info, SectorInfo)
                self.assertTrue(0 <= sector_info.confidence <= 1)
                self.assertIsInstance(sector_info.characteristics, dict)
                self.assertIsInstance(sector_info.similar_assets, list)
    
    def test_classify_sectors_kmeans(self):
        """Test K-means sector classification."""
        sectors = self.classifier.classify_sectors(self.returns_data, method='kmeans')
        
        self.assertIsInstance(sectors, dict)
        
        # Should have some sector assignments
        if sectors:
            for asset, sector_info in sectors.items():
                self.assertIsInstance(sector_info, SectorInfo)
                self.assertIn('n_assets', sector_info.characteristics)
    
    def test_classify_sectors_correlation(self):
        """Test correlation-based sector classification."""
        sectors = self.classifier.classify_sectors(self.returns_data, method='correlation')
        
        self.assertIsInstance(sectors, dict)
        
        # Check sector structure
        if sectors:
            sector_names = set(info.sector_name for info in sectors.values())
            self.assertGreater(len(sector_names), 0)
    
    def test_calculate_sector_momentum(self):
        """Test sector momentum calculation."""
        sectors = self.classifier.classify_sectors(self.returns_data, method='hierarchical')
        
        if sectors:
            momentum = self.classifier.calculate_sector_momentum(self.returns_data, sectors)
            
            self.assertIsInstance(momentum, dict)
            
            for sector_name, mom_value in momentum.items():
                self.assertIsInstance(mom_value, float)
                self.assertTrue(np.isfinite(mom_value))
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        sectors = self.classifier.classify_sectors(empty_data)
        self.assertEqual(len(sectors), 0)
    
    def test_insufficient_assets(self):
        """Test classification with insufficient assets."""
        small_data = self.returns_data[['AAPL']]  # Only one asset
        
        sectors = self.classifier.classify_sectors(small_data)
        # Should handle gracefully (might return empty or single sector)
        self.assertIsInstance(sectors, dict)


class TestFactorAnalyzer(unittest.TestCase):
    """Test factor analyzer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create factor-driven returns
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Generate factors
        market_factor = np.random.normal(0.0005, 0.02, n_days)
        size_factor = np.random.normal(0.0002, 0.015, n_days)
        value_factor = np.random.normal(0.0001, 0.012, n_days)
        
        self.returns_data = pd.DataFrame(index=dates)
        
        # Assets with different factor exposures
        self.returns_data['ASSET_1'] = (0.8 * market_factor + 0.3 * size_factor + 0.1 * value_factor + 
                                       np.random.normal(0, 0.01, n_days))
        self.returns_data['ASSET_2'] = (0.7 * market_factor + 0.5 * size_factor + 0.2 * value_factor + 
                                       np.random.normal(0, 0.012, n_days))
        self.returns_data['ASSET_3'] = (0.9 * market_factor + 0.1 * size_factor + 0.4 * value_factor + 
                                       np.random.normal(0, 0.011, n_days))
        self.returns_data['ASSET_4'] = (0.6 * market_factor + 0.2 * size_factor + 0.6 * value_factor + 
                                       np.random.normal(0, 0.013, n_days))
        self.returns_data['ASSET_5'] = (0.5 * market_factor + 0.4 * size_factor + 0.3 * value_factor + 
                                       np.random.normal(0, 0.014, n_days))
        
        self.analyzer = FactorAnalyzer(n_factors=3, method='pca')
    
    def test_analyzer_creation(self):
        """Test factor analyzer creation."""
        self.assertEqual(self.analyzer.n_factors, 3)
        self.assertEqual(self.analyzer.method, 'pca')
        self.assertIsNone(self.analyzer.factor_model)
    
    def test_extract_factors_pca(self):
        """Test factor extraction using PCA."""
        factor_exposures = self.analyzer.extract_factors(self.returns_data)
        
        self.assertIsInstance(factor_exposures, dict)
        
        if factor_exposures:  # Only test if extraction succeeded
            for asset, exposure in factor_exposures.items():
                self.assertIsInstance(exposure, FactorExposure)
                self.assertIsInstance(exposure.factor_loadings, dict)
                self.assertEqual(len(exposure.factor_loadings), 3)  # n_factors
                self.assertTrue(0 <= exposure.explained_variance <= 1)
                self.assertGreaterEqual(exposure.residual_variance, 0)
                self.assertTrue(0 <= exposure.factor_stability <= 1)
                self.assertIsInstance(exposure.factor_returns, pd.Series)
    
    def test_extract_factors_factor_analysis(self):
        """Test factor extraction using Factor Analysis."""
        fa_analyzer = FactorAnalyzer(n_factors=3, method='factor_analysis')
        factor_exposures = fa_analyzer.extract_factors(self.returns_data)
        
        self.assertIsInstance(factor_exposures, dict)
        
        if factor_exposures:
            for asset, exposure in factor_exposures.items():
                self.assertIsInstance(exposure, FactorExposure)
                self.assertIn('factor_1', exposure.factor_loadings)
    
    def test_insufficient_data(self):
        """Test factor analysis with insufficient data."""
        short_data = self.returns_data.head(20)  # Very short time series
        
        factor_exposures = self.analyzer.extract_factors(short_data)
        # Should handle gracefully
        self.assertIsInstance(factor_exposures, dict)
    
    def test_insufficient_assets(self):
        """Test factor analysis with fewer assets than factors."""
        small_data = self.returns_data[['ASSET_1', 'ASSET_2']]  # Only 2 assets, 3 factors
        
        factor_exposures = self.analyzer.extract_factors(small_data)
        # Should handle gracefully
        self.assertIsInstance(factor_exposures, dict)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        factor_exposures = self.analyzer.extract_factors(empty_data)
        self.assertEqual(len(factor_exposures), 0)


class TestAdaptiveWeightingSystem(unittest.TestCase):
    """Test adaptive weighting system."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Create returns with changing correlations
        self.returns_data = pd.DataFrame(index=dates)
        
        # Target asset
        self.returns_data['TARGET'] = np.random.normal(0.0005, 0.02, n_days)
        
        # Correlated assets with time-varying correlation
        base_corr = 0.6
        for i, asset in enumerate(['ASSET_A', 'ASSET_B', 'ASSET_C']):
            # Create time-varying correlation
            correlation_series = base_corr + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 252)
            
            correlated_returns = []
            for j, target_return in enumerate(self.returns_data['TARGET']):
                corr = correlation_series[j]
                noise = np.random.normal(0, 0.015)
                correlated_return = corr * target_return + np.sqrt(1 - corr**2) * noise
                correlated_returns.append(correlated_return)
            
            self.returns_data[asset] = correlated_returns
        
        self.weighting_system = AdaptiveWeightingSystem(lookback_window=63, adaptation_rate=0.1)
    
    def test_weighting_system_creation(self):
        """Test adaptive weighting system creation."""
        self.assertEqual(self.weighting_system.lookback_window, 63)
        self.assertEqual(self.weighting_system.adaptation_rate, 0.1)
        self.assertEqual(len(self.weighting_system.current_weights), 0)
    
    def test_calculate_adaptive_weights(self):
        """Test adaptive weight calculation."""
        weights = self.weighting_system.calculate_adaptive_weights(self.returns_data, 'TARGET')
        
        self.assertIsInstance(weights, dict)
        
        # Should have weights for other assets
        expected_assets = ['ASSET_A', 'ASSET_B', 'ASSET_C']
        for asset in expected_assets:
            self.assertIn(asset, weights)
            self.assertTrue(0 <= weights[asset] <= 1)
        
        # Weights should sum to 1 (normalized)
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_calculate_adaptive_weights_nonexistent_target(self):
        """Test weight calculation with non-existent target."""
        weights = self.weighting_system.calculate_adaptive_weights(self.returns_data, 'NONEXISTENT')
        
        self.assertEqual(len(weights), 0)
    
    def test_update_correlation_history(self):
        """Test correlation history update."""
        initial_history_len = len(self.weighting_system.correlation_history)
        
        self.weighting_system.update_correlation_history(self.returns_data)
        
        self.assertEqual(len(self.weighting_system.correlation_history), initial_history_len + 1)
    
    def test_detect_correlation_regime_changes(self):
        """Test correlation regime change detection."""
        # Add some history first
        for _ in range(5):
            self.weighting_system.update_correlation_history(self.returns_data)
        
        regime_changes = self.weighting_system.detect_correlation_regime_changes(threshold=0.3)
        
        self.assertIsInstance(regime_changes, dict)
        
        # Check structure of regime changes
        for pair, changed in regime_changes.items():
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(changed, bool)
    
    def test_weight_adaptation(self):
        """Test that weights adapt over time."""
        # Calculate initial weights
        initial_weights = self.weighting_system.calculate_adaptive_weights(self.returns_data, 'TARGET')
        
        # Calculate weights again (should be adapted)
        adapted_weights = self.weighting_system.calculate_adaptive_weights(self.returns_data, 'TARGET')
        
        # Weights should be similar but potentially different due to adaptation
        for asset in initial_weights:
            if asset in adapted_weights:
                # Should be within reasonable range
                self.assertTrue(0 <= adapted_weights[asset] <= 1)


class TestCrossAssetRelationshipModeler(unittest.TestCase):
    """Test the main cross-asset relationship modeler."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create comprehensive test dataset
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_days = len(dates)
        
        # Generate market factors
        market_factor = np.random.normal(0.0005, 0.02, n_days)
        tech_factor = np.random.normal(0.0003, 0.025, n_days)
        fin_factor = np.random.normal(0.0002, 0.018, n_days)
        
        self.returns_data = pd.DataFrame(index=dates)
        
        # Technology sector
        self.returns_data['AAPL'] = 0.8 * market_factor + 0.7 * tech_factor + np.random.normal(0, 0.01, n_days)
        self.returns_data['GOOGL'] = 0.7 * market_factor + 0.8 * tech_factor + np.random.normal(0, 0.012, n_days)
        self.returns_data['MSFT'] = 0.75 * market_factor + 0.75 * tech_factor + np.random.normal(0, 0.011, n_days)
        
        # Financial sector
        self.returns_data['JPM'] = 0.6 * market_factor + 0.8 * fin_factor + np.random.normal(0, 0.015, n_days)
        self.returns_data['BAC'] = 0.5 * market_factor + 0.7 * fin_factor + np.random.normal(0, 0.016, n_days)
        
        # Mixed/Other
        self.returns_data['XOM'] = 0.4 * market_factor + np.random.normal(0, 0.02, n_days)
        
        self.modeler = CrossAssetRelationshipModeler(
            correlation_window=63,
            min_correlation_periods=30,
            n_factors=3,
            max_sectors=5
        )
    
    def test_modeler_creation(self):
        """Test relationship modeler creation."""
        self.assertIsNotNone(self.modeler.correlation_analyzer)
        self.assertIsNotNone(self.modeler.sector_classifier)
        self.assertIsNotNone(self.modeler.factor_analyzer)
        self.assertIsNotNone(self.modeler.adaptive_weighting)
        self.assertEqual(len(self.modeler.relationship_cache), 0)
    
    def test_analyze_cross_asset_relationships(self):
        """Test comprehensive cross-asset relationship analysis."""
        results = self.modeler.analyze_cross_asset_relationships(self.returns_data)
        
        self.assertIsInstance(results, dict)
        
        # Check main sections
        expected_sections = [
            'timestamp', 'n_assets', 'analysis_period',
            'correlation_analysis', 'sector_analysis', 
            'factor_analysis', 'adaptive_weighting'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results)
        
        # Check analysis period
        self.assertEqual(results['n_assets'], 6)
        self.assertIn('start', results['analysis_period'])
        self.assertIn('end', results['analysis_period'])
        
        # Check correlation analysis
        corr_analysis = results['correlation_analysis']
        if 'error' not in corr_analysis:
            self.assertIn('correlation_matrix', corr_analysis)
            self.assertIn('avg_correlation', corr_analysis)
        
        # Check sector analysis
        sector_analysis = results['sector_analysis']
        if 'error' not in sector_analysis:
            self.assertIn('sector_assignments', sector_analysis)
            self.assertIn('n_sectors', sector_analysis)
        
        # Check factor analysis
        factor_analysis = results['factor_analysis']
        if 'error' not in factor_analysis:
            self.assertIn('factor_exposures', factor_analysis)
            self.assertIn('n_factors', factor_analysis)
        
        # Check adaptive weighting
        adaptive_analysis = results['adaptive_weighting']
        if 'error' not in adaptive_analysis:
            self.assertIn('adaptive_weights', adaptive_analysis)
    
    def test_get_relationship_summary(self):
        """Test relationship summary generation."""
        # First run analysis
        self.modeler.analyze_cross_asset_relationships(self.returns_data)
        
        summary = self.modeler.get_relationship_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('last_update', summary)
        self.assertIn('n_assets', summary)
        self.assertEqual(summary['n_assets'], 6)
    
    def test_export_relationship_analysis(self):
        """Test exporting relationship analysis."""
        import json
        
        # Run analysis first
        self.modeler.analyze_cross_asset_relationships(self.returns_data)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.modeler.export_relationship_analysis(temp_filepath)
            
            # Check file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, dict)
            self.assertIn('timestamp', data)
            self.assertIn('n_assets', data)
        
        finally:
            # Clean up
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        results = self.modeler.analyze_cross_asset_relationships(empty_data)
        
        self.assertEqual(len(results), 0)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        short_data = self.returns_data.head(10)
        
        results = self.modeler.analyze_cross_asset_relationships(short_data)
        
        # Should handle gracefully and return results structure
        self.assertIsInstance(results, dict)
        self.assertIn('n_assets', results)


if __name__ == '__main__':
    unittest.main()