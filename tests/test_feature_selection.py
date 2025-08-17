"""
Unit tests for automatic feature selection and importance ranking.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from enhanced_timeseries.features.feature_selection import (
    BaseFeatureSelector, MutualInformationSelector, RecursiveFeatureElimination,
    PermutationImportanceSelector, UnivariateSelector, EnsembleFeatureSelector,
    CorrelationAnalyzer, FeatureSelectionPipeline
)


class TestMutualInformationSelector(unittest.TestCase):
    """Test MutualInformationSelector class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic regression data
        self.X, self.y = make_regression(
            n_samples=200, 
            n_features=20, 
            n_informative=10, 
            noise=0.1, 
            random_state=42
        )
        
        self.feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        # Add some correlated features
        self.X[:, -1] = self.X[:, 0] + np.random.normal(0, 0.01, self.X.shape[0])  # Highly correlated
        self.X[:, -2] = self.X[:, 1] * 2 + np.random.normal(0, 0.1, self.X.shape[0])  # Correlated
        
        self.selector = MutualInformationSelector(k=10)
    
    def test_selector_creation(self):
        """Test feature selector creation."""
        self.assertEqual(self.selector.k, 10)
        self.assertIsNone(self.selector.scores_)
        self.assertIsNone(self.selector.selected_features_)
    
    def test_fit_selector(self):
        """Test fitting the feature selector."""
        self.selector.fit(self.X, self.y)
        
        # Should have feature scores
        self.assertIsNotNone(self.selector.scores_)
        self.assertEqual(len(self.selector.scores_), self.X.shape[1])
        
        # Should have selected features
        self.assertIsNotNone(self.selector.selected_features_)
        self.assertEqual(len(self.selector.selected_features_), self.selector.k)
    
    def test_transform_data(self):
        """Test transforming data with selected features."""
        self.selector.fit(self.X, self.y)
        X_transformed = self.selector.transform(self.X)
        
        # Should have reduced number of features
        self.assertEqual(X_transformed.shape[1], self.selector.k)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X_transformed = self.selector.fit_transform(self.X, self.y)
        
        self.assertEqual(X_transformed.shape[1], self.selector.k)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
    
    def test_get_feature_importance(self):
        """Test getting feature importance scores."""
        self.selector.fit(self.X, self.y)
        
        importance_scores = self.selector.get_feature_importance()
        self.assertIsInstance(importance_scores, np.ndarray)
        self.assertEqual(len(importance_scores), self.X.shape[1])
        
        # All importance scores should be non-negative
        for score in importance_scores:
            self.assertGreaterEqual(score, 0)
    
    def test_get_selected_features(self):
        """Test getting selected features."""
        self.selector.fit(self.X, self.y)
        selected = self.selector.get_selected_features()
        
        self.assertIsInstance(selected, np.ndarray)
        self.assertEqual(len(selected), self.selector.k)
        
        # All selected features should be valid indices
        for idx in selected:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.X.shape[1])
    
    def test_individual_selection_methods(self):
        """Test individual selection methods."""
        # Test with specific methods
        methods = ['mutual_info', 'correlation', 'random_forest']
        selector = FeatureSelector(max_features=5, selection_methods=methods)
        
        selector.fit(self.X, self.y, self.feature_names)
        
        # Should have scores for each method
        for method in methods:
            if method in selector.feature_scores_:
                self.assertGreater(len(selector.feature_scores_[method]), 0)
    
    def test_clean_data_with_nans(self):
        """Test data cleaning with NaN values."""
        # Add some NaN values
        X_with_nans = self.X.copy()
        X_with_nans[0, 0] = np.nan
        X_with_nans[1, 1] = np.inf
        
        y_with_nans = self.y.copy()
        y_with_nans[2] = np.nan
        
        selector = FeatureSelector(max_features=5)
        
        # Should handle NaN values gracefully
        try:
            selector.fit(X_with_nans, y_with_nans, self.feature_names)
            self.assertTrue(True)  # If no exception, test passes
        except ValueError as e:
            # It's acceptable to raise ValueError for insufficient clean data
            self.assertIn("No valid samples", str(e))
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_X = np.array([]).reshape(0, 5)
        empty_y = np.array([])
        
        selector = FeatureSelector(max_features=3)
        
        with self.assertRaises(ValueError):
            selector.fit(empty_X, empty_y)


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test CorrelationAnalyzer class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create data with known correlations
        n_samples = 100
        self.X = np.random.randn(n_samples, 5)
        
        # Make feature 1 highly correlated with feature 0
        self.X[:, 1] = self.X[:, 0] + np.random.normal(0, 0.01, n_samples)
        
        # Make feature 4 moderately correlated with feature 2
        self.X[:, 4] = self.X[:, 2] * 0.8 + np.random.normal(0, 0.3, n_samples)
        
        self.feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        self.analyzer = CorrelationAnalyzer(correlation_threshold=0.9)
    
    def test_analyzer_creation(self):
        """Test correlation analyzer creation."""
        self.assertEqual(self.analyzer.correlation_threshold, 0.9)
        self.assertIsNone(self.analyzer.correlation_matrix_)
    
    def test_fit_analyzer(self):
        """Test fitting the correlation analyzer."""
        self.analyzer.fit(self.X, self.feature_names)
        
        # Should have correlation matrix
        self.assertIsNotNone(self.analyzer.correlation_matrix_)
        self.assertEqual(self.analyzer.correlation_matrix_.shape, (5, 5))
        
        # Should detect high correlations
        high_corr = self.analyzer.get_high_correlations()
        self.assertGreater(len(high_corr), 0)
        
        # Should identify multicollinear features
        multicollinear = self.analyzer.get_multicollinear_features()
        self.assertGreater(len(multicollinear), 0)
    
    def test_get_correlation_matrix(self):
        """Test getting correlation matrix."""
        self.analyzer.fit(self.X, self.feature_names)
        corr_matrix = self.analyzer.get_correlation_matrix()
        
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (5, 5))
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix.values), 1.0)
    
    def test_remove_multicollinear_features(self):
        """Test removing multicollinear features."""
        self.analyzer.fit(self.X, self.feature_names)
        
        X_filtered, filtered_names = self.analyzer.remove_multicollinear_features(
            self.X, self.feature_names
        )
        
        # Should have fewer features
        self.assertLess(len(filtered_names), len(self.feature_names))
        self.assertEqual(X_filtered.shape[1], len(filtered_names))
        self.assertEqual(X_filtered.shape[0], self.X.shape[0])
    
    def test_high_correlation_detection(self):
        """Test detection of high correlations."""
        self.analyzer.fit(self.X, self.feature_names)
        high_corr_pairs = self.analyzer.get_high_correlations()
        
        # Should find the highly correlated pair (feature_0 and feature_1)
        found_high_corr = False
        for pair in high_corr_pairs:
            if (pair['feature1'] == 'feature_0' and pair['feature2'] == 'feature_1') or \
               (pair['feature1'] == 'feature_1' and pair['feature2'] == 'feature_0'):
                found_high_corr = True
                self.assertGreater(pair['correlation'], 0.9)
                break
        
        self.assertTrue(found_high_corr, "Should detect high correlation between feature_0 and feature_1")
    
    def test_different_threshold(self):
        """Test with different correlation threshold."""
        # Use lower threshold
        analyzer_low = CorrelationAnalyzer(correlation_threshold=0.5)
        analyzer_low.fit(self.X, self.feature_names)
        
        # Use higher threshold
        analyzer_high = CorrelationAnalyzer(correlation_threshold=0.99)
        analyzer_high.fit(self.X, self.feature_names)
        
        # Lower threshold should find more correlations
        low_corr = analyzer_low.get_high_correlations()
        high_corr = analyzer_high.get_high_correlations()
        
        self.assertGreaterEqual(len(low_corr), len(high_corr))



class TestFeatureSelectionPipeline(unittest.TestCase):
    """Test FeatureSelectionPipeline class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic data with correlations
        self.X, self.y = make_regression(
            n_samples=150, 
            n_features=15, 
            n_informative=8, 
            noise=0.1, 
            random_state=42
        )
        
        # Add highly correlated features
        self.X[:, -1] = self.X[:, 0] + np.random.normal(0, 0.01, self.X.shape[0])
        self.X[:, -2] = self.X[:, 1] * 0.95 + np.random.normal(0, 0.05, self.X.shape[0])
        
        self.feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        self.pipeline = FeatureSelectionPipeline(
            max_features=8,
            correlation_threshold=0.9
        )
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        self.assertEqual(self.pipeline.max_features, 8)
        self.assertEqual(self.pipeline.correlation_threshold, 0.9)
        self.assertFalse(self.pipeline.pipeline_fitted_)
    
    def test_fit_pipeline(self):
        """Test fitting the pipeline."""
        self.pipeline.fit(self.X, self.y, self.feature_names)
        
        self.assertTrue(self.pipeline.pipeline_fitted_)
        self.assertGreater(len(self.pipeline.final_features_), 0)
        self.assertLessEqual(len(self.pipeline.final_features_), self.pipeline.max_features)
    
    def test_transform_pipeline(self):
        """Test transforming data with pipeline."""
        self.pipeline.fit(self.X, self.y, self.feature_names)
        X_transformed = self.pipeline.transform(self.X, self.feature_names)
        
        # Should have reduced features
        self.assertLessEqual(X_transformed.shape[1], self.pipeline.max_features)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
    
    def test_fit_transform_pipeline(self):
        """Test fit_transform method."""
        X_transformed = self.pipeline.fit_transform(self.X, self.y, self.feature_names)
        
        self.assertLessEqual(X_transformed.shape[1], self.pipeline.max_features)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
    
    def test_get_selection_summary(self):
        """Test getting selection summary."""
        self.pipeline.fit(self.X, self.y, self.feature_names)
        summary = self.pipeline.get_selection_summary()
        
        expected_keys = [
            'original_features', 'multicollinear_removed', 'final_selected',
            'selected_features', 'multicollinear_features', 'high_correlations'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check logical relationships
        self.assertEqual(summary['original_features'], len(self.feature_names))
        self.assertEqual(summary['final_selected'], len(self.pipeline.final_features_))
        self.assertLessEqual(summary['final_selected'], self.pipeline.max_features)
    
    def test_get_selected_features(self):
        """Test getting selected features."""
        self.pipeline.fit(self.X, self.y, self.feature_names)
        selected = self.pipeline.get_selected_features()
        
        self.assertIsInstance(selected, list)
        self.assertEqual(len(selected), len(self.pipeline.final_features_))
        
        # All selected features should be from original features
        for feature in selected:
            self.assertIn(feature, self.feature_names)
    
    def test_pipeline_without_fit(self):
        """Test using pipeline without fitting."""
        with self.assertRaises(ValueError):
            self.pipeline.transform(self.X, self.feature_names)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        self.X, self.y = make_regression(
            n_samples=100, 
            n_features=12, 
            n_informative=6, 
            noise=0.1, 
            random_state=42
        )
        
        self.feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
    
    def test_select_features_for_timeseries(self):
        """Test quick feature selection utility."""
        X_selected, selected_features = select_features_for_timeseries(
            self.X, self.y, self.feature_names, max_features=6
        )
        
        # Should return reduced feature set
        self.assertLessEqual(X_selected.shape[1], 6)
        self.assertEqual(X_selected.shape[0], self.X.shape[0])
        self.assertEqual(len(selected_features), X_selected.shape[1])
        
        # Selected features should be from original set
        for feature in selected_features:
            self.assertIn(feature, self.feature_names)
    
    def test_analyze_feature_correlations(self):
        """Test correlation analysis utility."""
        # Add some correlated features
        X_with_corr = self.X.copy()
        X_with_corr[:, -1] = X_with_corr[:, 0] + np.random.normal(0, 0.01, self.X.shape[0])
        
        analysis = analyze_feature_correlations(
            X_with_corr, self.feature_names, threshold=0.8
        )
        
        expected_keys = [
            'correlation_matrix', 'high_correlations', 'multicollinear_features',
            'n_high_correlations', 'n_multicollinear'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Should detect the high correlation we added
        self.assertGreater(analysis['n_high_correlations'], 0)
        self.assertGreater(analysis['n_multicollinear'], 0)
    
    def test_feature_selection_with_no_correlations(self):
        """Test feature selection with uncorrelated features."""
        # Create completely uncorrelated features
        X_uncorr = np.random.randn(100, 8)
        feature_names = [f'uncorr_feature_{i}' for i in range(8)]
        
        X_selected, selected_features = select_features_for_timeseries(
            X_uncorr, self.y[:100], feature_names, max_features=5
        )
        
        # Should still work and select features
        self.assertLessEqual(X_selected.shape[1], 5)
        self.assertEqual(len(selected_features), X_selected.shape[1])
    
    def test_correlation_analysis_with_perfect_correlations(self):
        """Test correlation analysis with perfect correlations."""
        # Create data with perfect correlation
        X_perfect = np.random.randn(50, 4)
        X_perfect[:, 1] = X_perfect[:, 0]  # Perfect correlation
        X_perfect[:, 3] = X_perfect[:, 2] * 2  # Perfect linear relationship
        
        feature_names = [f'perfect_feature_{i}' for i in range(4)]
        
        analysis = analyze_feature_correlations(
            X_perfect, feature_names, threshold=0.95
        )
        
        # Should detect perfect correlations
        self.assertGreater(analysis['n_high_correlations'], 0)
        self.assertGreater(analysis['n_multicollinear'], 0)


if __name__ == '__main__':
    unittest.main()