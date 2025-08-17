"""
Automatic feature selection and importance ranking for time series forecasting.
Implements various feature selection algorithms including mutual information,
recursive feature elimination, and correlation analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from sklearn.feature_selection import (
    mutual_info_regression, RFE, RFECV, SelectKBest, f_regression
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

warnings.filterwarnings('ignore')


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selection methods."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseFeatureSelector':
        """Fit the feature selector on training data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using the fitted selector."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        pass
    
    @abstractmethod
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        pass
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the selector and transform the data."""
        return self.fit(X, y).transform(X)


class MutualInformationSelector(BaseFeatureSelector):
    """Feature selection based on mutual information."""
    
    def __init__(self, k: int = 10, discrete_features: str = 'auto', 
                 random_state: int = 42):
        """
        Initialize mutual information selector.
        
        Args:
            k: Number of top features to select
            discrete_features: How to handle discrete features
            random_state: Random state for reproducibility
        """
        self.k = k
        self.discrete_features = discrete_features
        self.random_state = random_state
        self.scores_ = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MutualInformationSelector':
        """Fit the mutual information selector."""
        # Calculate mutual information scores
        self.scores_ = mutual_info_regression(
            X, y.ravel(), 
            discrete_features=self.discrete_features,
            random_state=self.random_state
        )
        
        # Select top k features
        self.selected_features_ = np.argsort(self.scores_)[-self.k:]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using selected indices."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before transform")
        return X[:, self.selected_features_]
    
    def get_feature_importance(self) -> np.ndarray:
        """Get mutual information scores."""
        if self.scores_ is None:
            raise ValueError("Selector must be fitted first")
        return self.scores_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_


class RecursiveFeatureElimination(BaseFeatureSelector):
    """Recursive Feature Elimination with cross-validation."""
    
    def __init__(self, estimator: Optional[object] = None, n_features: Optional[int] = None,
                 step: int = 1, cv: int = 5, scoring: str = 'neg_mean_squared_error',
                 random_state: int = 42):
        """
        Initialize RFE selector.
        
        Args:
            estimator: Base estimator for feature ranking
            n_features: Number of features to select (if None, uses CV)
            step: Number of features to remove at each iteration
            cv: Number of cross-validation folds
            scoring: Scoring metric for CV
            random_state: Random state for reproducibility
        """
        self.estimator = estimator or RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.n_features = n_features
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        
        self.selector_ = None
        self.selected_features_ = None
        self.feature_importance_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RecursiveFeatureElimination':
        """Fit the RFE selector."""
        if self.n_features is None:
            # Use cross-validation to find optimal number of features
            self.selector_ = RFECV(
                estimator=self.estimator,
                step=self.step,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1
            )
        else:
            # Use fixed number of features
            self.selector_ = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features,
                step=self.step
            )
        
        # Fit the selector
        self.selector_.fit(X, y.ravel())
        
        # Get selected features
        self.selected_features_ = np.where(self.selector_.support_)[0]
        
        # Get feature importance if available
        if hasattr(self.selector_.estimator_, 'feature_importances_'):
            self.feature_importance_ = self.selector_.estimator_.feature_importances_
        elif hasattr(self.selector_.estimator_, 'coef_'):
            self.feature_importance_ = np.abs(self.selector_.estimator_.coef_)
        else:
            # Use ranking as importance (inverted)
            self.feature_importance_ = 1.0 / self.selector_.ranking_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using RFE selector."""
        if self.selector_ is None:
            raise ValueError("Selector must be fitted before transform")
        return self.selector_.transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Selector must be fitted first")
        return self.feature_importance_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_


class PermutationImportanceSelector(BaseFeatureSelector):
    """Feature selection based on permutation importance."""
    
    def __init__(self, estimator: Optional[object] = None, n_features: int = 10,
                 n_repeats: int = 10, random_state: int = 42):
        """
        Initialize permutation importance selector.
        
        Args:
            estimator: Base estimator for importance calculation
            n_features: Number of features to select
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
        """
        self.estimator = estimator or RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.n_features = n_features
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        self.importance_scores_ = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PermutationImportanceSelector':
        """Fit the permutation importance selector."""
        # Fit the base estimator
        self.estimator.fit(X, y.ravel())
        
        # Calculate baseline score
        baseline_score = self.estimator.score(X, y.ravel())
        
        # Calculate permutation importance
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)
        
        np.random.seed(self.random_state)
        
        for feature_idx in range(n_features):
            scores = []
            
            for _ in range(self.n_repeats):
                # Create permuted copy
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
                
                # Calculate score with permuted feature
                permuted_score = self.estimator.score(X_permuted, y.ravel())
                
                # Importance is the decrease in score
                importance = baseline_score - permuted_score
                scores.append(importance)
            
            importance_scores[feature_idx] = np.mean(scores)
        
        self.importance_scores_ = importance_scores
        
        # Select top features
        self.selected_features_ = np.argsort(importance_scores)[-self.n_features:]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using selected indices."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before transform")
        return X[:, self.selected_features_]
    
    def get_feature_importance(self) -> np.ndarray:
        """Get permutation importance scores."""
        if self.importance_scores_ is None:
            raise ValueError("Selector must be fitted first")
        return self.importance_scores_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_


class CorrelationAnalyzer:
    """Correlation analysis and multicollinearity detection."""
    
    def __init__(self, threshold: float = 0.95, method: str = 'pearson'):
        """
        Initialize correlation analyzer.
        
        Args:
            threshold: Correlation threshold for multicollinearity detection
            method: Correlation method ('pearson' or 'spearman')
        """
        self.threshold = threshold
        self.method = method
        self.correlation_matrix_ = None
        self.high_corr_pairs_ = None
        self.features_to_remove_ = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'CorrelationAnalyzer':
        """Analyze correlations in the feature matrix."""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        
        # Calculate correlation matrix
        if self.method == 'pearson':
            self.correlation_matrix_ = np.corrcoef(X.T)
        elif self.method == 'spearman':
            from scipy.stats import spearmanr
            self.correlation_matrix_, _ = spearmanr(X, axis=0)
        else:
            raise ValueError(f"Unknown correlation method: {self.method}")
        
        # Find highly correlated pairs
        self.high_corr_pairs_ = []
        n_features = len(feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(self.correlation_matrix_[i, j])
                if corr > self.threshold:
                    self.high_corr_pairs_.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr,
                        'indices': (i, j)
                    })
        
        # Determine features to remove (keep first occurrence)
        features_to_remove = set()
        for pair in self.high_corr_pairs_:
            # Remove the second feature in each highly correlated pair
            features_to_remove.add(pair['indices'][1])
        
        self.features_to_remove_ = sorted(list(features_to_remove))
        
        return self
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix."""
        if self.correlation_matrix_ is None:
            raise ValueError("Analyzer must be fitted first")
        return self.correlation_matrix_
    
    def get_high_correlation_pairs(self) -> List[Dict]:
        """Get pairs of highly correlated features."""
        if self.high_corr_pairs_ is None:
            raise ValueError("Analyzer must be fitted first")
        return self.high_corr_pairs_
    
    def get_features_to_remove(self) -> List[int]:
        """Get indices of features to remove due to multicollinearity."""
        if self.features_to_remove_ is None:
            raise ValueError("Analyzer must be fitted first")
        return self.features_to_remove_
    
    def remove_correlated_features(self, X: np.ndarray) -> np.ndarray:
        """Remove highly correlated features from the dataset."""
        if self.features_to_remove_ is None:
            raise ValueError("Analyzer must be fitted first")
        
        features_to_keep = [i for i in range(X.shape[1]) if i not in self.features_to_remove_]
        return X[:, features_to_keep]


class UnivariateSelector(BaseFeatureSelector):
    """Univariate feature selection using statistical tests."""
    
    def __init__(self, score_func: Callable = f_regression, k: int = 10):
        """
        Initialize univariate selector.
        
        Args:
            score_func: Scoring function for feature evaluation
            k: Number of top features to select
        """
        self.score_func = score_func
        self.k = k
        self.selector_ = None
        self.scores_ = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'UnivariateSelector':
        """Fit the univariate selector."""
        self.selector_ = SelectKBest(score_func=self.score_func, k=self.k)
        self.selector_.fit(X, y.ravel())
        
        self.scores_ = self.selector_.scores_
        self.selected_features_ = self.selector_.get_support(indices=True)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using univariate selector."""
        if self.selector_ is None:
            raise ValueError("Selector must be fitted before transform")
        return self.selector_.transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get univariate scores."""
        if self.scores_ is None:
            raise ValueError("Selector must be fitted first")
        return self.scores_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_


class EnsembleFeatureSelector:
    """Ensemble feature selector combining multiple selection methods."""
    
    def __init__(self, selectors: List[BaseFeatureSelector], 
                 voting: str = 'soft', weights: Optional[List[float]] = None):
        """
        Initialize ensemble feature selector.
        
        Args:
            selectors: List of feature selectors
            voting: Voting method ('soft' or 'hard')
            weights: Weights for each selector (if None, equal weights)
        """
        self.selectors = selectors
        self.voting = voting
        self.weights = weights or [1.0] * len(selectors)
        
        self.ensemble_scores_ = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, n_features: int = 10) -> 'EnsembleFeatureSelector':
        """Fit all selectors and combine their results."""
        # Fit all selectors
        for selector in self.selectors:
            selector.fit(X, y)
        
        n_total_features = X.shape[1]
        
        if self.voting == 'soft':
            # Combine importance scores
            ensemble_scores = np.zeros(n_total_features)
            
            for i, selector in enumerate(self.selectors):
                scores = selector.get_feature_importance()
                # Normalize scores to [0, 1]
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                ensemble_scores += self.weights[i] * scores_norm
            
            self.ensemble_scores_ = ensemble_scores
            self.selected_features_ = np.argsort(ensemble_scores)[-n_features:]
            
        elif self.voting == 'hard':
            # Count votes for each feature
            feature_votes = np.zeros(n_total_features)
            
            for i, selector in enumerate(self.selectors):
                selected = selector.get_selected_features()
                feature_votes[selected] += self.weights[i]
            
            self.ensemble_scores_ = feature_votes
            self.selected_features_ = np.argsort(feature_votes)[-n_features:]
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using ensemble selection."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before transform")
        return X[:, self.selected_features_]
    
    def get_feature_importance(self) -> np.ndarray:
        """Get ensemble importance scores."""
        if self.ensemble_scores_ is None:
            raise ValueError("Selector must be fitted first")
        return self.ensemble_scores_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_


class FeatureSelectionPipeline:
    """Complete feature selection pipeline with multiple stages."""
    
    def __init__(self, 
                 correlation_threshold: float = 0.95,
                 n_features_mutual_info: int = 50,
                 n_features_rfe: int = 30,
                 n_features_final: int = 20,
                 random_state: int = 42):
        """
        Initialize feature selection pipeline.
        
        Args:
            correlation_threshold: Threshold for correlation analysis
            n_features_mutual_info: Features to keep after mutual information
            n_features_rfe: Features to keep after RFE
            n_features_final: Final number of features
            random_state: Random state for reproducibility
        """
        self.correlation_threshold = correlation_threshold
        self.n_features_mutual_info = n_features_mutual_info
        self.n_features_rfe = n_features_rfe
        self.n_features_final = n_features_final
        self.random_state = random_state
        
        # Pipeline components
        self.correlation_analyzer = None
        self.mutual_info_selector = None
        self.rfe_selector = None
        self.final_selector = None
        
        # Results
        self.selected_features_ = None
        self.feature_importance_ = None
        self.pipeline_results_ = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'FeatureSelectionPipeline':
        """Fit the complete feature selection pipeline."""
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.original_feature_names = feature_names
        current_X = X.copy()
        current_features = np.arange(X.shape[1])
        
        # Stage 1: Correlation analysis and multicollinearity removal
        print(f"Stage 1: Correlation analysis (starting with {current_X.shape[1]} features)")
        self.correlation_analyzer = CorrelationAnalyzer(threshold=self.correlation_threshold)
        self.correlation_analyzer.fit(current_X, 
                                    [feature_names[i] for i in current_features])
        
        # Remove highly correlated features
        features_to_remove = self.correlation_analyzer.get_features_to_remove()
        features_to_keep = [i for i in range(current_X.shape[1]) if i not in features_to_remove]
        
        current_X = current_X[:, features_to_keep]
        current_features = current_features[features_to_keep]
        
        self.pipeline_results_['after_correlation'] = {
            'n_features': current_X.shape[1],
            'removed_features': len(features_to_remove),
            'feature_indices': current_features.copy()
        }
        
        print(f"  Removed {len(features_to_remove)} correlated features, {current_X.shape[1]} remaining")
        
        # Stage 2: Mutual information selection
        if current_X.shape[1] > self.n_features_mutual_info:
            print(f"Stage 2: Mutual information selection (target: {self.n_features_mutual_info} features)")
            self.mutual_info_selector = MutualInformationSelector(
                k=min(self.n_features_mutual_info, current_X.shape[1]),
                random_state=self.random_state
            )
            self.mutual_info_selector.fit(current_X, y)
            
            selected_indices = self.mutual_info_selector.get_selected_features()
            current_X = current_X[:, selected_indices]
            current_features = current_features[selected_indices]
            
            self.pipeline_results_['after_mutual_info'] = {
                'n_features': current_X.shape[1],
                'feature_indices': current_features.copy()
            }
            
            print(f"  Selected {current_X.shape[1]} features using mutual information")
        
        # Stage 3: Recursive Feature Elimination
        if current_X.shape[1] > self.n_features_rfe:
            print(f"Stage 3: Recursive Feature Elimination (target: {self.n_features_rfe} features)")
            self.rfe_selector = RecursiveFeatureElimination(
                n_features=min(self.n_features_rfe, current_X.shape[1]),
                random_state=self.random_state
            )
            self.rfe_selector.fit(current_X, y)
            
            selected_indices = self.rfe_selector.get_selected_features()
            current_X = current_X[:, selected_indices]
            current_features = current_features[selected_indices]
            
            self.pipeline_results_['after_rfe'] = {
                'n_features': current_X.shape[1],
                'feature_indices': current_features.copy()
            }
            
            print(f"  Selected {current_X.shape[1]} features using RFE")
        
        # Stage 4: Final ensemble selection
        if current_X.shape[1] > self.n_features_final:
            print(f"Stage 4: Final ensemble selection (target: {self.n_features_final} features)")
            
            # Create ensemble of different selectors
            selectors = [
                PermutationImportanceSelector(
                    n_features=min(self.n_features_final, current_X.shape[1]),
                    random_state=self.random_state
                ),
                UnivariateSelector(k=min(self.n_features_final, current_X.shape[1]))
            ]
            
            self.final_selector = EnsembleFeatureSelector(
                selectors=selectors,
                voting='soft',
                weights=[0.6, 0.4]  # Prefer permutation importance
            )
            
            self.final_selector.fit(current_X, y, n_features=self.n_features_final)
            
            selected_indices = self.final_selector.get_selected_features()
            current_X = current_X[:, selected_indices]
            current_features = current_features[selected_indices]
            
            print(f"  Selected {current_X.shape[1]} features using ensemble method")
        
        # Store final results
        self.selected_features_ = current_features
        
        # Calculate final feature importance
        if self.final_selector is not None:
            self.feature_importance_ = self.final_selector.get_feature_importance()
        elif self.rfe_selector is not None:
            importance = self.rfe_selector.get_feature_importance()
            selected = self.rfe_selector.get_selected_features()
            self.feature_importance_ = importance[selected]
        else:
            self.feature_importance_ = np.ones(len(self.selected_features_))
        
        self.pipeline_results_['final'] = {
            'n_features': len(self.selected_features_),
            'feature_indices': self.selected_features_.copy(),
            'feature_names': [self.original_feature_names[i] for i in self.selected_features_]
        }
        
        print(f"Pipeline complete: {X.shape[1]} -> {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using the fitted pipeline."""
        if self.selected_features_ is None:
            raise ValueError("Pipeline must be fitted before transform")
        return X[:, self.selected_features_]
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of final selected features."""
        if self.selected_features_ is None:
            raise ValueError("Pipeline must be fitted first")
        return self.selected_features_
    
    def get_feature_importance(self) -> np.ndarray:
        """Get importance scores for selected features."""
        if self.feature_importance_ is None:
            raise ValueError("Pipeline must be fitted first")
        return self.feature_importance_
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline results."""
        if not self.pipeline_results_:
            raise ValueError("Pipeline must be fitted first")
        return self.pipeline_results_
    
    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Pipeline must be fitted first")
        return [self.original_feature_names[i] for i in self.selected_features_]


# Utility functions
def evaluate_feature_selection(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              selector: BaseFeatureSelector,
                              estimator: Optional[object] = None) -> Dict[str, float]:
    """
    Evaluate feature selection performance.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        selector: Feature selector to evaluate
        estimator: Estimator for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit selector and transform data
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Train and evaluate model
    estimator.fit(X_train_selected, y_train.ravel())
    
    # Predictions
    y_train_pred = estimator.predict(X_train_selected)
    y_test_pred = estimator.predict(X_test_selected)
    
    # Metrics
    results = {
        'n_features_original': X_train.shape[1],
        'n_features_selected': X_train_selected.shape[1],
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return results


def compare_feature_selectors(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             selectors: Dict[str, BaseFeatureSelector],
                             estimator: Optional[object] = None) -> pd.DataFrame:
    """
    Compare multiple feature selectors.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        selectors: Dictionary of selectors to compare
        estimator: Estimator for evaluation
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, selector in selectors.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_feature_selection(
            X_train, y_train, X_test, y_test, selector, estimator
        )
        metrics['selector'] = name
        results.append(metrics)
    
    return pd.DataFrame(results)


def create_feature_selection_report(pipeline: FeatureSelectionPipeline,
                                   feature_names: List[str]) -> Dict:
    """
    Create a comprehensive feature selection report.
    
    Args:
        pipeline: Fitted feature selection pipeline
        feature_names: Original feature names
        
    Returns:
        Dictionary containing the report
    """
    if pipeline.selected_features_ is None:
        raise ValueError("Pipeline must be fitted first")
    
    selected_names = pipeline.get_selected_feature_names()
    importance_scores = pipeline.get_feature_importance()
    
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    report = {
        'summary': {
            'original_features': len(feature_names),
            'selected_features': len(selected_names),
            'reduction_ratio': 1 - len(selected_names) / len(feature_names)
        },
        'pipeline_stages': pipeline.get_pipeline_summary(),
        'selected_features': [
            {
                'rank': i + 1,
                'name': selected_names[sorted_indices[i]],
                'importance': importance_scores[sorted_indices[i]],
                'original_index': pipeline.selected_features_[sorted_indices[i]]
            }
            for i in range(len(selected_names))
        ],
        'correlation_analysis': {
            'high_correlation_pairs': len(pipeline.correlation_analyzer.get_high_correlation_pairs()),
            'features_removed': len(pipeline.correlation_analyzer.get_features_to_remove())
        }
    }
    
    return report