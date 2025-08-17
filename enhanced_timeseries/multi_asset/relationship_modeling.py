"""
Cross-asset relationship modeling for multi-asset prediction systems.
Implements correlation analysis, sector classification, factor analysis, and adaptive weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
from collections import defaultdict
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class CorrelationMetrics:
    """Container for correlation analysis results."""
    pearson_correlation: float
    spearman_correlation: float
    kendall_correlation: float
    rolling_correlation_mean: float
    rolling_correlation_std: float
    correlation_stability: float
    max_correlation: float
    min_correlation: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'pearson': self.pearson_correlation,
            'spearman': self.spearman_correlation,
            'kendall': self.kendall_correlation,
            'rolling_mean': self.rolling_correlation_mean,
            'rolling_std': self.rolling_correlation_std,
            'stability': self.correlation_stability,
            'max_corr': self.max_correlation,
            'min_corr': self.min_correlation
        }


@dataclass
class SectorInfo:
    """Information about asset sector classification."""
    sector_name: str
    confidence: float
    characteristics: Dict[str, float]
    similar_assets: List[str]
    sector_momentum: float
    sector_volatility: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sector_name': self.sector_name,
            'confidence': self.confidence,
            'characteristics': self.characteristics,
            'similar_assets': self.similar_assets,
            'sector_momentum': self.sector_momentum,
            'sector_volatility': self.sector_volatility
        }


@dataclass
class FactorExposure:
    """Factor exposure analysis results."""
    factor_loadings: Dict[str, float]
    explained_variance: float
    factor_returns: pd.Series
    residual_variance: float
    factor_stability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'factor_loadings': self.factor_loadings,
            'explained_variance': self.explained_variance,
            'residual_variance': self.residual_variance,
            'factor_stability': self.factor_stability
        }


class CorrelationAnalyzer:
    """Advanced correlation analysis for cross-asset relationships."""
    
    def __init__(self, rolling_window: int = 63, min_periods: int = 30):
        """
        Initialize correlation analyzer.
        
        Args:
            rolling_window: Window for rolling correlation calculation
            min_periods: Minimum periods required for correlation calculation
        """
        self.rolling_window = rolling_window
        self.min_periods = min_periods
        self.correlation_history = {}
        
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                   method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for asset returns.
        
        Args:
            data: DataFrame with asset returns (assets as columns)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix
        """
        if data.empty:
            return pd.DataFrame()
        
        # Calculate returns if not already returns
        if data.max().max() > 10:  # Likely prices, not returns
            returns = data.pct_change().dropna()
        else:
            returns = data.dropna()
        
        if len(returns) < self.min_periods:
            logger.warning(f"Insufficient data for correlation calculation: {len(returns)} < {self.min_periods}")
            return pd.DataFrame()
        
        try:
            if method == 'pearson':
                corr_matrix = returns.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = returns.corr(method='spearman')
            elif method == 'kendall':
                corr_matrix = returns.corr(method='kendall')
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            return corr_matrix.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_correlations(self, data: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
        """
        Calculate rolling correlations between all asset pairs.
        
        Args:
            data: DataFrame with asset data
            
        Returns:
            Dictionary of rolling correlations for each asset pair
        """
        if data.empty or len(data.columns) < 2:
            return {}
        
        # Calculate returns
        if data.max().max() > 10:
            returns = data.pct_change().dropna()
        else:
            returns = data.dropna()
        
        rolling_correlations = {}
        
        for i, asset1 in enumerate(returns.columns):
            for j, asset2 in enumerate(returns.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    try:
                        rolling_corr = returns[asset1].rolling(
                            window=self.rolling_window, 
                            min_periods=self.min_periods
                        ).corr(returns[asset2])
                        
                        rolling_correlations[(asset1, asset2)] = rolling_corr.dropna()
                        
                    except Exception as e:
                        logger.warning(f"Failed to calculate rolling correlation for {asset1}-{asset2}: {e}")
        
        return rolling_correlations
    
    def analyze_correlation_pair(self, asset1_data: pd.Series, 
                               asset2_data: pd.Series) -> CorrelationMetrics:
        """
        Comprehensive correlation analysis between two assets.
        
        Args:
            asset1_data: Time series data for first asset
            asset2_data: Time series data for second asset
            
        Returns:
            Correlation metrics
        """
        # Align data
        aligned_data = pd.DataFrame({
            'asset1': asset1_data,
            'asset2': asset2_data
        }).dropna()
        
        if len(aligned_data) < self.min_periods:
            # Return default metrics for insufficient data
            return CorrelationMetrics(
                pearson_correlation=0.0,
                spearman_correlation=0.0,
                kendall_correlation=0.0,
                rolling_correlation_mean=0.0,
                rolling_correlation_std=0.0,
                correlation_stability=0.0,
                max_correlation=0.0,
                min_correlation=0.0
            )
        
        asset1_aligned = aligned_data['asset1']
        asset2_aligned = aligned_data['asset2']
        
        # Calculate different correlation measures
        pearson_corr = asset1_aligned.corr(asset2_aligned, method='pearson')
        spearman_corr = asset1_aligned.corr(asset2_aligned, method='spearman')
        kendall_corr = asset1_aligned.corr(asset2_aligned, method='kendall')
        
        # Rolling correlation analysis
        rolling_corr = asset1_aligned.rolling(
            window=self.rolling_window, 
            min_periods=self.min_periods
        ).corr(asset2_aligned).dropna()
        
        if len(rolling_corr) > 0:
            rolling_mean = rolling_corr.mean()
            rolling_std = rolling_corr.std()
            max_corr = rolling_corr.max()
            min_corr = rolling_corr.min()
            
            # Correlation stability (inverse of coefficient of variation)
            if abs(rolling_mean) > 1e-6:
                stability = 1 - (rolling_std / abs(rolling_mean))
            else:
                stability = 0.0
        else:
            rolling_mean = pearson_corr
            rolling_std = 0.0
            max_corr = pearson_corr
            min_corr = pearson_corr
            stability = 1.0
        
        return CorrelationMetrics(
            pearson_correlation=pearson_corr if not pd.isna(pearson_corr) else 0.0,
            spearman_correlation=spearman_corr if not pd.isna(spearman_corr) else 0.0,
            kendall_correlation=kendall_corr if not pd.isna(kendall_corr) else 0.0,
            rolling_correlation_mean=rolling_mean if not pd.isna(rolling_mean) else 0.0,
            rolling_correlation_std=rolling_std if not pd.isna(rolling_std) else 0.0,
            correlation_stability=max(0.0, min(1.0, stability)) if not pd.isna(stability) else 0.0,
            max_correlation=max_corr if not pd.isna(max_corr) else 0.0,
            min_correlation=min_corr if not pd.isna(min_corr) else 0.0
        )
    
    def detect_correlation_regimes(self, rolling_correlations: Dict[Tuple[str, str], pd.Series],
                                 n_regimes: int = 3) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Detect correlation regimes using clustering.
        
        Args:
            rolling_correlations: Rolling correlations from calculate_rolling_correlations
            n_regimes: Number of correlation regimes to detect
            
        Returns:
            Dictionary with regime information for each asset pair
        """
        regime_results = {}
        
        for pair, corr_series in rolling_correlations.items():
            if len(corr_series) < n_regimes * 5:  # Need sufficient data
                continue
            
            try:
                # Prepare data for clustering
                corr_values = corr_series.values.reshape(-1, 1)
                
                # Use KMeans clustering to identify regimes
                kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                regime_labels = kmeans.fit_predict(corr_values)
                
                # Calculate regime statistics
                regime_stats = {}
                for regime_id in range(n_regimes):
                    regime_mask = regime_labels == regime_id
                    regime_corrs = corr_values[regime_mask]
                    
                    if len(regime_corrs) > 0:
                        regime_stats[f'regime_{regime_id}'] = {
                            'mean_correlation': float(np.mean(regime_corrs)),
                            'std_correlation': float(np.std(regime_corrs)),
                            'frequency': float(np.sum(regime_mask) / len(regime_labels)),
                            'center': float(kmeans.cluster_centers_[regime_id][0])
                        }
                
                # Calculate silhouette score for regime quality
                if len(set(regime_labels)) > 1:
                    silhouette = silhouette_score(corr_values, regime_labels)
                else:
                    silhouette = 0.0
                
                regime_results[pair] = {
                    'regimes': regime_stats,
                    'regime_labels': regime_labels.tolist(),
                    'regime_quality': silhouette,
                    'n_regimes': n_regimes
                }
                
            except Exception as e:
                logger.warning(f"Failed to detect correlation regimes for {pair}: {e}")
        
        return regime_results


class SectorClassifier:
    """Classify assets into sectors based on return patterns and correlations."""
    
    def __init__(self, min_cluster_size: int = 2, max_clusters: int = 10):
        """
        Initialize sector classifier.
        
        Args:
            min_cluster_size: Minimum assets per sector
            max_clusters: Maximum number of sectors to identify
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.sector_assignments = {}
        self.sector_characteristics = {}
        
    def classify_sectors(self, returns_data: pd.DataFrame, 
                        method: str = 'hierarchical') -> Dict[str, SectorInfo]:
        """
        Classify assets into sectors based on return patterns.
        
        Args:
            returns_data: DataFrame with asset returns
            method: Clustering method ('hierarchical', 'kmeans', 'correlation')
            
        Returns:
            Dictionary mapping assets to sector information
        """
        if returns_data.empty or len(returns_data.columns) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = returns_data.corr().fillna(0)
        
        if method == 'hierarchical':
            return self._hierarchical_clustering(returns_data, corr_matrix)
        elif method == 'kmeans':
            return self._kmeans_clustering(returns_data, corr_matrix)
        elif method == 'correlation':
            return self._correlation_clustering(returns_data, corr_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _hierarchical_clustering(self, returns_data: pd.DataFrame, 
                               corr_matrix: pd.DataFrame) -> Dict[str, SectorInfo]:
        """Perform hierarchical clustering for sector classification."""
        try:
            # Convert correlation to distance
            distance_matrix = 1 - corr_matrix.abs()
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine optimal number of clusters
            best_n_clusters = self._find_optimal_clusters(
                distance_matrix.values, linkage_matrix, returns_data.columns
            )
            
            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, best_n_clusters, criterion='maxclust')
            
            return self._create_sector_info(returns_data, corr_matrix, cluster_labels, returns_data.columns)
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            return {}
    
    def _kmeans_clustering(self, returns_data: pd.DataFrame, 
                          corr_matrix: pd.DataFrame) -> Dict[str, SectorInfo]:
        """Perform K-means clustering for sector classification."""
        try:
            # Use return statistics as features
            features = self._extract_return_features(returns_data)
            
            if features.empty:
                return {}
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Find optimal number of clusters
            best_n_clusters = self._find_optimal_kmeans_clusters(features_scaled)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            return self._create_sector_info(returns_data, corr_matrix, cluster_labels, features.index)
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return {}
    
    def _correlation_clustering(self, returns_data: pd.DataFrame, 
                              corr_matrix: pd.DataFrame) -> Dict[str, SectorInfo]:
        """Perform correlation-based clustering."""
        try:
            # Use correlation threshold to form clusters
            threshold = 0.5  # Assets with correlation > threshold are in same sector
            
            # Create graph from correlation matrix
            G = nx.Graph()
            assets = corr_matrix.columns.tolist()
            G.add_nodes_from(assets)
            
            # Add edges for highly correlated assets
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j and abs(corr_matrix.iloc[i, j]) > threshold:
                        G.add_edge(asset1, asset2, weight=abs(corr_matrix.iloc[i, j]))
            
            # Find connected components (sectors)
            sectors = list(nx.connected_components(G))
            
            # Create cluster labels
            cluster_labels = np.zeros(len(assets))
            for sector_id, sector_assets in enumerate(sectors):
                for asset in sector_assets:
                    asset_idx = assets.index(asset)
                    cluster_labels[asset_idx] = sector_id
            
            return self._create_sector_info(returns_data, corr_matrix, cluster_labels, assets)
            
        except Exception as e:
            logger.error(f"Correlation clustering failed: {e}")
            return {}
    
    def _extract_return_features(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from return data for clustering."""
        features = pd.DataFrame(index=returns_data.columns)
        
        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()
            
            if len(asset_returns) > 10:
                features.loc[asset, 'mean_return'] = asset_returns.mean()
                features.loc[asset, 'volatility'] = asset_returns.std()
                features.loc[asset, 'skewness'] = asset_returns.skew()
                features.loc[asset, 'kurtosis'] = asset_returns.kurt()
                features.loc[asset, 'sharpe_ratio'] = (
                    asset_returns.mean() / asset_returns.std() if asset_returns.std() > 0 else 0
                )
                
                # Rolling statistics
                rolling_vol = asset_returns.rolling(21).std()
                features.loc[asset, 'vol_of_vol'] = rolling_vol.std() if len(rolling_vol.dropna()) > 0 else 0
                
                # Autocorrelation
                features.loc[asset, 'autocorr_1'] = asset_returns.autocorr(lag=1) if len(asset_returns) > 1 else 0
                features.loc[asset, 'autocorr_5'] = asset_returns.autocorr(lag=5) if len(asset_returns) > 5 else 0
        
        return features.fillna(0)
    
    def _find_optimal_clusters(self, distance_matrix: np.ndarray, 
                             linkage_matrix: np.ndarray, 
                             asset_names: List[str]) -> int:
        """Find optimal number of clusters for hierarchical clustering."""
        max_clusters = min(self.max_clusters, len(asset_names) // self.min_cluster_size)
        
        if max_clusters < 2:
            return 1
        
        # Try different numbers of clusters and evaluate
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # Check minimum cluster size constraint
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                if np.min(counts) < self.min_cluster_size:
                    continue
                
                # Calculate silhouette score
                if len(unique_labels) > 1:
                    score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
            except Exception:
                continue
        
        return best_n_clusters
    
    def _find_optimal_kmeans_clusters(self, features: np.ndarray) -> int:
        """Find optimal number of clusters for K-means."""
        max_clusters = min(self.max_clusters, len(features) // self.min_cluster_size)
        
        if max_clusters < 2:
            return 1
        
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # Check minimum cluster size constraint
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                if np.min(counts) < self.min_cluster_size:
                    continue
                
                # Calculate silhouette score
                score = silhouette_score(features, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    
            except Exception:
                continue
        
        return best_n_clusters
    
    def _create_sector_info(self, returns_data: pd.DataFrame, 
                          corr_matrix: pd.DataFrame,
                          cluster_labels: np.ndarray, 
                          asset_names: List[str]) -> Dict[str, SectorInfo]:
        """Create SectorInfo objects from clustering results."""
        sector_info = {}
        
        # Group assets by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(asset_names[i])
        
        for cluster_id, cluster_assets in clusters.items():
            if len(cluster_assets) < self.min_cluster_size:
                continue
            
            sector_name = f"Sector_{cluster_id}"
            
            # Calculate sector characteristics
            sector_returns = returns_data[cluster_assets].mean(axis=1)
            sector_momentum = sector_returns.rolling(21).mean().iloc[-1] if len(sector_returns) > 21 else sector_returns.mean()
            sector_volatility = sector_returns.std()
            
            # Calculate intra-sector correlations
            if len(cluster_assets) > 1:
                sector_corr_matrix = corr_matrix.loc[cluster_assets, cluster_assets]
                avg_intra_correlation = sector_corr_matrix.values[np.triu_indices_from(sector_corr_matrix.values, k=1)].mean()
            else:
                avg_intra_correlation = 1.0
            
            # Sector characteristics
            characteristics = {
                'avg_return': float(sector_returns.mean()),
                'volatility': float(sector_volatility),
                'avg_intra_correlation': float(avg_intra_correlation),
                'n_assets': len(cluster_assets)
            }
            
            # Confidence based on intra-sector correlation
            confidence = max(0.0, min(1.0, avg_intra_correlation))
            
            # Create SectorInfo for each asset in the sector
            for asset in cluster_assets:
                sector_info[asset] = SectorInfo(
                    sector_name=sector_name,
                    confidence=confidence,
                    characteristics=characteristics,
                    similar_assets=[a for a in cluster_assets if a != asset],
                    sector_momentum=float(sector_momentum) if not pd.isna(sector_momentum) else 0.0,
                    sector_volatility=float(sector_volatility) if not pd.isna(sector_volatility) else 0.0
                )
        
        return sector_info
    
    def calculate_sector_momentum(self, returns_data: pd.DataFrame, 
                                sector_assignments: Dict[str, SectorInfo],
                                window: int = 21) -> Dict[str, float]:
        """Calculate momentum for each sector."""
        sector_momentum = {}
        
        # Group assets by sector
        sectors = defaultdict(list)
        for asset, sector_info in sector_assignments.items():
            sectors[sector_info.sector_name].append(asset)
        
        for sector_name, sector_assets in sectors.items():
            if len(sector_assets) == 0:
                continue
            
            # Calculate sector return as equal-weighted average
            sector_returns = returns_data[sector_assets].mean(axis=1)
            
            # Calculate momentum as rolling mean
            momentum = sector_returns.rolling(window).mean().iloc[-1] if len(sector_returns) >= window else sector_returns.mean()
            
            sector_momentum[sector_name] = float(momentum) if not pd.isna(momentum) else 0.0
        
        return sector_momentum


class FactorAnalyzer:
    """Market-wide factor analysis and principal component extraction."""
    
    def __init__(self, n_factors: int = 5, method: str = 'pca'):
        """
        Initialize factor analyzer.
        
        Args:
            n_factors: Number of factors to extract
            method: Factor extraction method ('pca', 'factor_analysis')
        """
        self.n_factors = n_factors
        self.method = method
        self.factor_model = None
        self.scaler = StandardScaler()
        
    def extract_factors(self, returns_data: pd.DataFrame) -> Dict[str, FactorExposure]:
        """
        Extract market factors from asset returns.
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            Dictionary mapping assets to their factor exposures
        """
        if returns_data.empty or len(returns_data.columns) < self.n_factors:
            return {}
        
        # Clean data
        clean_data = returns_data.dropna()
        if len(clean_data) < 30:  # Need sufficient observations
            logger.warning("Insufficient data for factor analysis")
            return {}
        
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(clean_data)
        
        try:
            if self.method == 'pca':
                self.factor_model = PCA(n_components=self.n_factors)
            elif self.method == 'factor_analysis':
                self.factor_model = FactorAnalysis(n_components=self.n_factors, random_state=42)
            else:
                raise ValueError(f"Unknown factor method: {self.method}")
            
            # Fit factor model
            factor_scores = self.factor_model.fit_transform(returns_scaled)
            
            # Calculate factor exposures for each asset
            factor_exposures = {}
            
            for i, asset in enumerate(clean_data.columns):
                # Get loadings for this asset
                if self.method == 'pca':
                    loadings = self.factor_model.components_[:, i]
                    explained_var = self.factor_model.explained_variance_ratio_.sum()
                else:
                    loadings = self.factor_model.components_[:, i]
                    explained_var = 1 - np.mean(self.factor_model.noise_variance_) / np.var(returns_scaled[:, i])
                
                # Create factor loading dictionary
                factor_loadings = {
                    f'factor_{j+1}': float(loadings[j]) for j in range(len(loadings))
                }
                
                # Calculate residual variance
                asset_returns = clean_data.iloc[:, i]
                factor_contribution = np.dot(factor_scores, loadings)
                residuals = asset_returns - factor_contribution
                residual_var = np.var(residuals)
                
                # Calculate factor stability (consistency over time)
                if len(clean_data) > 126:  # Need sufficient data for rolling analysis
                    rolling_loadings = self._calculate_rolling_factor_loadings(
                        clean_data.iloc[:, i], factor_scores, window=63
                    )
                    factor_stability = 1 - np.std(rolling_loadings) / (np.abs(np.mean(rolling_loadings)) + 1e-6)
                else:
                    factor_stability = 0.5  # Default moderate stability
                
                # Create factor returns (first factor as proxy)
                factor_returns = pd.Series(factor_scores[:, 0], index=clean_data.index)
                
                factor_exposures[asset] = FactorExposure(
                    factor_loadings=factor_loadings,
                    explained_variance=float(explained_var),
                    factor_returns=factor_returns,
                    residual_variance=float(residual_var),
                    factor_stability=max(0.0, min(1.0, float(factor_stability)))
                )
            
            return factor_exposures
            
        except Exception as e:
            logger.error(f"Factor analysis failed: {e}")
            return {}
    
    def _calculate_rolling_factor_loadings(self, asset_returns: pd.Series, 
                                         factor_scores: np.ndarray, 
                                         window: int = 63) -> np.ndarray:
        """Calculate rolling factor loadings for stability analysis."""
        rolling_loadings = []
        
        for i in range(window, len(asset_returns)):
            window_returns = asset_returns.iloc[i-window:i]
            window_factors = factor_scores[i-window:i, 0]  # Use first factor
            
            # Simple linear regression
            if np.std(window_factors) > 1e-6:
                loading = np.corrcoef(window_returns, window_factors)[0, 1]
                rolling_loadings.append(loading if not pd.isna(loading) else 0.0)
        
        return np.array(rolling_loadings)
    
    def get_factor_returns(self) -> Optional[pd.DataFrame]:
        """Get the extracted factor returns."""
        if self.factor_model is None:
            return None
        
        # This would need to be called after extract_factors
        # and would return the factor time series
        return None


class AdaptiveWeightingSystem:
    """Adaptive cross-asset feature weighting based on correlation changes."""
    
    def __init__(self, lookback_window: int = 63, adaptation_rate: float = 0.1):
        """
        Initialize adaptive weighting system.
        
        Args:
            lookback_window: Window for correlation calculation
            adaptation_rate: Rate of weight adaptation (0-1)
        """
        self.lookback_window = lookback_window
        self.adaptation_rate = adaptation_rate
        self.current_weights = {}
        self.correlation_history = {}
        
    def calculate_adaptive_weights(self, returns_data: pd.DataFrame, 
                                 target_asset: str) -> Dict[str, float]:
        """
        Calculate adaptive weights for cross-asset features.
        
        Args:
            returns_data: DataFrame with asset returns
            target_asset: Asset for which to calculate weights
            
        Returns:
            Dictionary of adaptive weights for each asset
        """
        if target_asset not in returns_data.columns:
            return {}
        
        target_returns = returns_data[target_asset]
        other_assets = [col for col in returns_data.columns if col != target_asset]
        
        if len(other_assets) == 0:
            return {}
        
        weights = {}
        
        for asset in other_assets:
            # Calculate rolling correlation
            rolling_corr = target_returns.rolling(
                window=self.lookback_window
            ).corr(returns_data[asset]).dropna()
            
            if len(rolling_corr) == 0:
                weights[asset] = 0.0
                continue
            
            # Current correlation strength
            current_corr = abs(rolling_corr.iloc[-1]) if not pd.isna(rolling_corr.iloc[-1]) else 0.0
            
            # Correlation stability (inverse of volatility)
            corr_stability = 1 - (rolling_corr.std() / (abs(rolling_corr.mean()) + 1e-6))
            corr_stability = max(0.0, min(1.0, corr_stability))
            
            # Combine correlation strength and stability
            base_weight = current_corr * (0.7 + 0.3 * corr_stability)
            
            # Adaptive adjustment
            if asset in self.current_weights:
                # Exponential moving average of weights
                adapted_weight = (
                    (1 - self.adaptation_rate) * self.current_weights[asset] +
                    self.adaptation_rate * base_weight
                )
            else:
                adapted_weight = base_weight
            
            weights[asset] = max(0.0, min(1.0, adapted_weight))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {asset: weight / total_weight for asset, weight in weights.items()}
        
        # Update current weights
        self.current_weights.update(weights)
        
        return weights
    
    def update_correlation_history(self, returns_data: pd.DataFrame):
        """Update correlation history for trend analysis."""
        corr_matrix = returns_data.corr()
        timestamp = datetime.now()
        
        self.correlation_history[timestamp] = corr_matrix
        
        # Keep only recent history (last 100 updates)
        if len(self.correlation_history) > 100:
            oldest_key = min(self.correlation_history.keys())
            del self.correlation_history[oldest_key]
    
    def detect_correlation_regime_changes(self, threshold: float = 0.3) -> Dict[Tuple[str, str], bool]:
        """
        Detect significant changes in correlation regimes.
        
        Args:
            threshold: Threshold for detecting significant correlation changes
            
        Returns:
            Dictionary indicating regime changes for asset pairs
        """
        if len(self.correlation_history) < 2:
            return {}
        
        # Get recent and older correlation matrices
        timestamps = sorted(self.correlation_history.keys())
        recent_corr = self.correlation_history[timestamps[-1]]
        older_corr = self.correlation_history[timestamps[-min(10, len(timestamps))]]
        
        regime_changes = {}
        
        for i, asset1 in enumerate(recent_corr.columns):
            for j, asset2 in enumerate(recent_corr.columns):
                if i < j:  # Avoid duplicates
                    recent_val = recent_corr.iloc[i, j]
                    older_val = older_corr.iloc[i, j]
                    
                    if not (pd.isna(recent_val) or pd.isna(older_val)):
                        change = abs(recent_val - older_val)
                        regime_changes[(asset1, asset2)] = change > threshold
        
        return regime_changes


class CrossAssetRelationshipModeler:
    """
    Main class for comprehensive cross-asset relationship modeling.
    Integrates correlation analysis, sector classification, factor analysis, and adaptive weighting.
    """
    
    def __init__(self, 
                 correlation_window: int = 63,
                 min_correlation_periods: int = 30,
                 n_factors: int = 5,
                 max_sectors: int = 10):
        """
        Initialize cross-asset relationship modeler.
        
        Args:
            correlation_window: Window for rolling correlation calculations
            min_correlation_periods: Minimum periods for correlation calculation
            n_factors: Number of factors for factor analysis
            max_sectors: Maximum number of sectors for classification
        """
        self.correlation_analyzer = CorrelationAnalyzer(correlation_window, min_correlation_periods)
        self.sector_classifier = SectorClassifier(max_clusters=max_sectors)
        self.factor_analyzer = FactorAnalyzer(n_factors=n_factors)
        self.adaptive_weighting = AdaptiveWeightingSystem(lookback_window=correlation_window)
        
        self.relationship_cache = {}
        self.last_update = None
        
    def analyze_cross_asset_relationships(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive cross-asset relationship analysis.
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            Dictionary with complete relationship analysis
        """
        if returns_data.empty:
            return {}
        
        logger.info(f"Analyzing cross-asset relationships for {len(returns_data.columns)} assets")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_assets': len(returns_data.columns),
            'analysis_period': {
                'start': returns_data.index[0].isoformat(),
                'end': returns_data.index[-1].isoformat(),
                'n_periods': len(returns_data)
            }
        }
        
        # 1. Correlation Analysis
        logger.info("Performing correlation analysis...")
        try:
            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(returns_data)
            rolling_correlations = self.correlation_analyzer.calculate_rolling_correlations(returns_data)
            correlation_regimes = self.correlation_analyzer.detect_correlation_regimes(rolling_correlations)
            
            results['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
                'avg_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()) if not correlation_matrix.empty else 0.0,
                'correlation_regimes': correlation_regimes,
                'n_correlation_pairs': len(rolling_correlations)
            }
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            results['correlation_analysis'] = {'error': str(e)}
        
        # 2. Sector Classification
        logger.info("Performing sector classification...")
        try:
            sector_assignments = self.sector_classifier.classify_sectors(returns_data)
            sector_momentum = self.sector_classifier.calculate_sector_momentum(returns_data, sector_assignments)
            
            # Summarize sector information
            sectors_summary = {}
            for asset, sector_info in sector_assignments.items():
                sector_name = sector_info.sector_name
                if sector_name not in sectors_summary:
                    sectors_summary[sector_name] = {
                        'assets': [],
                        'characteristics': sector_info.characteristics,
                        'momentum': sector_momentum.get(sector_name, 0.0)
                    }
                sectors_summary[sector_name]['assets'].append(asset)
            
            results['sector_analysis'] = {
                'sector_assignments': {asset: info.to_dict() for asset, info in sector_assignments.items()},
                'sectors_summary': sectors_summary,
                'n_sectors': len(sectors_summary)
            }
        except Exception as e:
            logger.error(f"Sector classification failed: {e}")
            results['sector_analysis'] = {'error': str(e)}
        
        # 3. Factor Analysis
        logger.info("Performing factor analysis...")
        try:
            factor_exposures = self.factor_analyzer.extract_factors(returns_data)
            
            # Summarize factor analysis
            if factor_exposures:
                avg_explained_variance = np.mean([exp.explained_variance for exp in factor_exposures.values()])
                avg_factor_stability = np.mean([exp.factor_stability for exp in factor_exposures.values()])
                
                results['factor_analysis'] = {
                    'factor_exposures': {asset: exp.to_dict() for asset, exp in factor_exposures.items()},
                    'avg_explained_variance': float(avg_explained_variance),
                    'avg_factor_stability': float(avg_factor_stability),
                    'n_factors': self.factor_analyzer.n_factors
                }
            else:
                results['factor_analysis'] = {'error': 'No factor exposures calculated'}
        except Exception as e:
            logger.error(f"Factor analysis failed: {e}")
            results['factor_analysis'] = {'error': str(e)}
        
        # 4. Adaptive Weighting
        logger.info("Calculating adaptive weights...")
        try:
            adaptive_weights = {}
            for target_asset in returns_data.columns:
                weights = self.adaptive_weighting.calculate_adaptive_weights(returns_data, target_asset)
                if weights:
                    adaptive_weights[target_asset] = weights
            
            # Update correlation history
            self.adaptive_weighting.update_correlation_history(returns_data)
            
            # Detect regime changes
            regime_changes = self.adaptive_weighting.detect_correlation_regime_changes()
            
            results['adaptive_weighting'] = {
                'adaptive_weights': adaptive_weights,
                'regime_changes': {f"{pair[0]}-{pair[1]}": change for pair, change in regime_changes.items()},
                'n_regime_changes': sum(regime_changes.values())
            }
        except Exception as e:
            logger.error(f"Adaptive weighting failed: {e}")
            results['adaptive_weighting'] = {'error': str(e)}
        
        # Cache results
        self.relationship_cache = results
        self.last_update = datetime.now()
        
        logger.info("Cross-asset relationship analysis completed")
        return results
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get a summary of cross-asset relationships."""
        if not self.relationship_cache:
            return {}
        
        summary = {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'n_assets': self.relationship_cache.get('n_assets', 0)
        }
        
        # Correlation summary
        corr_analysis = self.relationship_cache.get('correlation_analysis', {})
        if 'avg_correlation' in corr_analysis:
            summary['avg_correlation'] = corr_analysis['avg_correlation']
            summary['n_correlation_pairs'] = corr_analysis.get('n_correlation_pairs', 0)
        
        # Sector summary
        sector_analysis = self.relationship_cache.get('sector_analysis', {})
        if 'n_sectors' in sector_analysis:
            summary['n_sectors'] = sector_analysis['n_sectors']
        
        # Factor summary
        factor_analysis = self.relationship_cache.get('factor_analysis', {})
        if 'avg_explained_variance' in factor_analysis:
            summary['avg_explained_variance'] = factor_analysis['avg_explained_variance']
            summary['avg_factor_stability'] = factor_analysis.get('avg_factor_stability', 0)
        
        # Adaptive weighting summary
        adaptive_analysis = self.relationship_cache.get('adaptive_weighting', {})
        if 'n_regime_changes' in adaptive_analysis:
            summary['n_regime_changes'] = adaptive_analysis['n_regime_changes']
        
        return summary
    
    def export_relationship_analysis(self, filepath: str):
        """Export relationship analysis to file."""
        if not self.relationship_cache:
            logger.warning("No relationship analysis to export")
            return
        
        import json
        
        # Prepare data for JSON serialization
        export_data = self.relationship_cache.copy()
        
        # Convert any remaining pandas objects to serializable format
        def convert_for_json(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        # Recursively convert objects
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_for_json(data)
        
        export_data = recursive_convert(export_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Relationship analysis exported to {filepath}")