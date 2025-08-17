"""Adaptive outlier detector for ultra-precision prediction"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..core.interfaces import OutlierDetector
from ..core.exceptions import OutlierDetectionError
from ..core.logging_config import setup_logging
from .local_outlier_analyzer import LocalOutlierAnalyzer
from .market_regime_filter import MarketRegimeFilter


class AdaptiveOutlierDetector(OutlierDetector):
    """Multi-stage outlier detection with extreme strictness for ultra-precision"""
    
    def __init__(self, strictness_level: float = 0.8):
        """
        Initialize adaptive outlier detector
        
        Args:
            strictness_level: Strictness level (0.0 to 1.0, higher = more strict)
        """
        self.strictness_level = strictness_level
        self.logger = setup_logging()
        
        # Initialize sub-components
        self.local_analyzer = LocalOutlierAnalyzer()
        self.regime_filter = MarketRegimeFilter()
        
        # Outlier detection statistics
        self.removal_stats = {
            'total_samples': 0,
            'z_score_removed': 0,
            'modified_z_removed': 0,
            'iqr_removed': 0,
            'local_removed': 0,
            'regime_removed': 0,
            'total_removed': 0,
            'removal_percentage': 0.0
        }
        
        # Fitted parameters
        self.is_fitted = False
        self.z_threshold = 1.0  # Extremely strict
        self.modified_z_threshold = 1.5
        self.iqr_percentiles = (15, 85)  # Tighter than usual (25, 75)
        
    def fit_outlier_detectors(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit outlier detection models"""
        try:
            self.logger.info("Fitting outlier detection models...")
            
            # Adjust thresholds based on strictness level
            self.z_threshold = 1.0 * (2.0 - self.strictness_level)  # More strict = lower threshold
            self.modified_z_threshold = 1.5 * (2.0 - self.strictness_level)
            
            # Fit local outlier analyzer
            self.local_analyzer.fit(X, y)
            
            # Fit regime filter
            self.regime_filter.fit(X, y)
            
            self.is_fitted = True
            self.logger.info("Outlier detection models fitted successfully")
            
        except Exception as e:
            raise OutlierDetectionError(f"Failed to fit outlier detectors: {str(e)}")
    
    def _z_score_filter(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply extreme Z-score filtering"""
        try:
            z_scores = np.abs((y - np.mean(y)) / (np.std(y) + 1e-10))
            mask = z_scores < self.z_threshold
            
            removed_count = np.sum(~mask)
            self.removal_stats['z_score_removed'] = removed_count
            
            self.logger.debug(f"Z-score filter removed {removed_count} samples (threshold: {self.z_threshold})")
            return mask
            
        except Exception as e:
            raise OutlierDetectionError(f"Z-score filtering failed: {str(e)}")
    
    def _modified_z_score_filter(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply modified Z-score using median absolute deviation"""
        try:
            median_y = np.median(y)
            mad = np.median(np.abs(y - median_y))
            modified_z_scores = 0.6745 * (y - median_y) / (mad + 1e-10)
            mask = np.abs(modified_z_scores) < self.modified_z_threshold
            
            removed_count = np.sum(~mask)
            self.removal_stats['modified_z_removed'] = removed_count
            
            self.logger.debug(f"Modified Z-score filter removed {removed_count} samples")
            return mask
            
        except Exception as e:
            raise OutlierDetectionError(f"Modified Z-score filtering failed: {str(e)}")
    
    def _iqr_filter(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply tight IQR bounds filtering"""
        try:
            Q1 = np.percentile(y, self.iqr_percentiles[0])
            Q3 = np.percentile(y, self.iqr_percentiles[1])
            IQR = Q3 - Q1
            
            # Tighter bounds than usual
            multiplier = 0.5 * (2.0 - self.strictness_level)  # More strict = smaller multiplier
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            
            removed_count = np.sum(~mask)
            self.removal_stats['iqr_removed'] = removed_count
            
            self.logger.debug(f"IQR filter removed {removed_count} samples (bounds: {lower_bound:.4f} - {upper_bound:.4f})")
            return mask
            
        except Exception as e:
            raise OutlierDetectionError(f"IQR filtering failed: {str(e)}")
    
    def _percentile_filter(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply percentile-based filtering"""
        try:
            # Remove extreme percentiles based on strictness
            lower_percentile = 5 + (10 * self.strictness_level)  # 5-15%
            upper_percentile = 95 - (10 * self.strictness_level)  # 85-95%
            
            lower_bound = np.percentile(y, lower_percentile)
            upper_bound = np.percentile(y, upper_percentile)
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            
            removed_count = np.sum(~mask)
            self.logger.debug(f"Percentile filter removed {removed_count} samples ({lower_percentile}%-{upper_percentile}%)")
            
            return mask
            
        except Exception as e:
            raise OutlierDetectionError(f"Percentile filtering failed: {str(e)}")
    
    def detect_outliers(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Detect outliers without removal"""
        try:
            if not self.is_fitted:
                self.fit_outlier_detectors(X, y)
            
            # Apply all detection methods
            z_mask = self._z_score_filter(X, y)
            modified_z_mask = self._modified_z_score_filter(X, y)
            iqr_mask = self._iqr_filter(X, y)
            percentile_mask = self._percentile_filter(X, y)
            
            # Get local outliers
            local_mask = self.local_analyzer.detect_outliers(X, y)
            
            # Get regime outliers
            regime_mask = self.regime_filter.detect_outliers(X, y)
            
            # Combine all masks (intersection - all must agree it's not an outlier)
            final_mask = (z_mask & modified_z_mask & iqr_mask & 
                         percentile_mask & local_mask & regime_mask)
            
            return final_mask
            
        except Exception as e:
            raise OutlierDetectionError(f"Outlier detection failed: {str(e)}")
    
    def remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using multi-stage detection"""
        try:
            self.logger.info("Starting multi-stage outlier removal...")
            
            # Store original counts
            self.removal_stats['total_samples'] = len(y)
            
            # Detect outliers
            final_mask = self.detect_outliers(X, y)
            
            # Apply mask to remove outliers
            X_clean = X[final_mask]
            y_clean = y[final_mask]
            
            # Update statistics
            total_removed = len(y) - len(y_clean)
            self.removal_stats['total_removed'] = total_removed
            self.removal_stats['removal_percentage'] = (total_removed / len(y)) * 100
            
            self.logger.info(f"Removed {total_removed} outliers ({self.removal_stats['removal_percentage']:.2f}%)")
            self.logger.info(f"Remaining samples: {len(y_clean)}")
            
            # Validate minimum removal percentage
            if self.removal_stats['removal_percentage'] < 15.0:
                self.logger.warning(f"Removal percentage ({self.removal_stats['removal_percentage']:.2f}%) below target 20%")
            
            return X_clean, y_clean
            
        except Exception as e:
            raise OutlierDetectionError(f"Outlier removal failed: {str(e)}")
    
    def get_removal_statistics(self) -> Dict[str, float]:
        """Get outlier removal statistics"""
        return self.removal_stats.copy()
    
    def set_strictness_level(self, strictness: float) -> None:
        """Set strictness level and update thresholds"""
        if not 0.0 <= strictness <= 1.0:
            raise ValueError("Strictness level must be between 0.0 and 1.0")
        
        self.strictness_level = strictness
        
        # Update thresholds
        self.z_threshold = 1.0 * (2.0 - strictness)
        self.modified_z_threshold = 1.5 * (2.0 - strictness)
        
        # Update percentiles for IQR
        percentile_adjustment = 10 * strictness
        self.iqr_percentiles = (15 + percentile_adjustment, 85 - percentile_adjustment)
        
        self.logger.info(f"Updated strictness to {strictness}, Z-threshold: {self.z_threshold}")
    
    def analyze_outlier_patterns(self, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """Analyze patterns in detected outliers"""
        try:
            outlier_mask = ~self.detect_outliers(X, y)
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(outlier_indices) == 0:
                return {'outlier_count': 0, 'patterns': {}}
            
            outlier_values = y[outlier_indices]
            
            analysis = {
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(y)) * 100,
                'outlier_value_stats': {
                    'mean': float(np.mean(outlier_values)),
                    'std': float(np.std(outlier_values)),
                    'min': float(np.min(outlier_values)),
                    'max': float(np.max(outlier_values))
                },
                'outlier_positions': {
                    'first_10_percent': np.sum(outlier_indices < len(y) * 0.1),
                    'middle_80_percent': np.sum((outlier_indices >= len(y) * 0.1) & 
                                              (outlier_indices < len(y) * 0.9)),
                    'last_10_percent': np.sum(outlier_indices >= len(y) * 0.9)
                },
                'consecutive_outliers': self._find_consecutive_outliers(outlier_indices)
            }
            
            return analysis
            
        except Exception as e:
            raise OutlierDetectionError(f"Outlier pattern analysis failed: {str(e)}")
    
    def _find_consecutive_outliers(self, outlier_indices: np.ndarray) -> Dict[str, int]:
        """Find consecutive outlier patterns"""
        if len(outlier_indices) == 0:
            return {'max_consecutive': 0, 'total_groups': 0}
        
        consecutive_groups = []
        current_group = [outlier_indices[0]]
        
        for i in range(1, len(outlier_indices)):
            if outlier_indices[i] == outlier_indices[i-1] + 1:
                current_group.append(outlier_indices[i])
            else:
                consecutive_groups.append(len(current_group))
                current_group = [outlier_indices[i]]
        
        consecutive_groups.append(len(current_group))
        
        return {
            'max_consecutive': max(consecutive_groups),
            'total_groups': len(consecutive_groups),
            'avg_group_size': np.mean(consecutive_groups)
        }
    
    def get_outlier_summary(self) -> str:
        """Get human-readable outlier removal summary"""
        stats = self.removal_stats
        
        summary = f"""
Outlier Removal Summary:
========================
Total samples: {stats['total_samples']}
Total removed: {stats['total_removed']} ({stats['removal_percentage']:.2f}%)

Breakdown by method:
- Z-score filter: {stats['z_score_removed']} samples
- Modified Z-score: {stats['modified_z_removed']} samples  
- IQR filter: {stats['iqr_removed']} samples
- Local outliers: {stats['local_removed']} samples
- Regime filter: {stats['regime_removed']} samples

Strictness level: {self.strictness_level}
Z-score threshold: {self.z_threshold}
IQR percentiles: {self.iqr_percentiles}
"""
        return summary