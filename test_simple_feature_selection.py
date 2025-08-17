#!/usr/bin/env python3

import sys
sys.path.append('.')

from enhanced_timeseries.features.feature_selection import MutualInformationSelector
from sklearn.datasets import make_regression
import numpy as np

def test_mutual_info_selector():
    """Simple test for MutualInformationSelector."""
    # Create test data
    X, y = make_regression(n_samples=100, n_features=15, n_informative=8, random_state=42)
    
    # Create selector
    selector = MutualInformationSelector(k=8, random_state=42)
    
    # Test fit_transform
    X_selected = selector.fit_transform(X, y)
    
    assert X_selected.shape == (100, 8), f"Expected (100, 8), got {X_selected.shape}"
    
    # Test get_feature_importance
    importance = selector.get_feature_importance()
    assert len(importance) == 15, f"Expected 15 importance scores, got {len(importance)}"
    assert np.all(importance >= 0), "All importance scores should be non-negative"
    
    # Test get_selected_features
    selected = selector.get_selected_features()
    assert len(selected) == 8, f"Expected 8 selected features, got {len(selected)}"
    
    print("✓ MutualInformationSelector test passed")

if __name__ == "__main__":
    test_mutual_info_selector()
    print("All tests passed!")