"""Local outlier analyzer for time series anomaly detection"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import OutlierDetectionError
from ..core.logging_config import setup_logging


class LocalOutlierAnalyzer:
    """Analyze local outliers in time series data"""
    
    def __init__(self, window_size: int = 20, sensitivity: float = 2.0):
        """
        Initialize local outlier analyzer
        
        Args:
            window_size: Size of sliding window for local analysis
            sensitivity: Sensitivity multiplier for outlier detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.logger = setup_logging()
        self.is_fitted = False
        
        # Local statistics
        self.local_stats = {
            'windows_analyzed': 0,
            'outliers_detected': 0,
            'avg_local_std': 0.0
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit local outlier analyzer"""
        try:
            self.logger.debug("Fitting local outlier analyzer...")
            
            # Calculate average local standard deviation
            local_stds = []
            for i in range(self.window_size, len(y)):
                window = y[i-self.window_size:i]
                local_stds.append(np.std(window))
            
            self.local_stats['avg_local_std'] = np.mean(local_stds) if local_stds else 0.0
            self.is_fitted = True
            
            self.logger.debug(f"Local analyzer fitted with avg std: {self.local_stats['avg_local_std']:.6f}")
            
        except Exception as e:
            raise OutlierDetectionError(f"Failed to fit local analyzer: {str(e)}")
    
    def detect_outliers(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Detect local outliers using sliding window analysis"""
        try:
            if not self.is_fitted:
                self.fit(X, y)
            
            mask = np.ones(len(y), dtype=bool)
            outliers_detected = 0
            
            # Sliding window outlier detection
            for i in range(self.window_size, len(y) - self.window_size):
                # Get local window
                window_start = max(0, i - self.window_size // 2)
                window_end = min(len(y), i + self.window_size // 2)
                local_window = y[window_start:window_end]
                
                # Calculate local statistics
                local_median = np.median(local_window)
                local_mad = np.median(np.abs(local_window - local_median))
                
                # Check if current point is outlier
                if local_mad > 0:
                    deviation = abs(y[i] - local_median) / local_mad
                    if deviation > self.sensitivity:
                        mask[i] = False
                        outliers_detected += 1
            
            # Pattern-based outlier detection
            mask = self._detect_pattern_outliers(y, mask)
            
            # Temporal consistency check
            mask = self._temporal_consistency_check(y, mask)
            
            self.local_stats['outliers_detected'] = np.sum(~mask)
            self.local_stats['windows_analyzed'] = len(y) - 2 * self.window_size
            
            self.logger.debug(f"Local analyzer detected {self.local_stats['outliers_detected']} outliers")
            
            return mask
            
        except Exception as e:
            raise OutlierDetectionError(f"Local outlier detection failed: {str(e)}")
    
    def _detect_pattern_outliers(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect outliers based on local patterns"""
        try:
            # Detect sudden spikes
            for i in range(2, len(y) - 2):
                if not mask[i]:  # Already marked as outlier
                    continue
                
                # Check for sudden spike pattern
                prev_avg = np.mean(y[i-2:i])
                next_avg = np.mean(y[i+1:i+3])
                current = y[i]
                
                # If current value is very different from neighbors
                if abs(current - prev_avg) > 3 * np.std(y[i-2:i]) and \
                   abs(current - next_avg) > 3 * np.std(y[i+1:i+3]):
                    mask[i] = False
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"Pattern outlier detection failed: {str(e)}")
            return mask
    
    def _temporal_consistency_check(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Check temporal consistency of values"""
        try:
            # Check for values that break temporal trends
            for i in range(5, len(y) - 5):
                if not mask[i]:  # Already marked as outlier
                    continue
                
                # Get trend before and after
                before_trend = np.polyfit(range(5), y[i-5:i], 1)[0]
                after_trend = np.polyfit(range(5), y[i+1:i+6], 1)[0]
                
                # Expected value based on trend
                expected = y[i-1] + before_trend
                
                # If actual value deviates significantly from expected
                if abs(y[i] - expected) > 2 * np.std(y[i-5:i+5]):
                    # Check if it's consistent with overall pattern
                    local_std = np.std(y[i-10:i+10]) if i >= 10 and i < len(y) - 10 else np.std(y)
                    if abs(y[i] - expected) > 3 * local_std:
                        mask[i] = False
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"Temporal consistency check failed: {str(e)}")
            return mask
    
    def analyze_local_patterns(self, y: np.ndarray) -> Dict[str, any]:
        """Analyze local patterns in the data"""
        try:
            analysis = {
                'local_volatility': [],
                'trend_changes': 0,
                'spike_count': 0,
                'local_correlations': []
            }
            
            # Analyze local volatility
            for i in range(self.window_size, len(y), self.window_size):
                window = y[i-self.window_size:i]
                volatility = np.std(window) / np.mean(window) if np.mean(window) != 0 else 0
                analysis['local_volatility'].append(volatility)
            
            # Count trend changes
            trends = []
            for i in range(10, len(y) - 10, 10):
                window = y[i-10:i+10]
                trend = np.polyfit(range(len(window)), window, 1)[0]
                trends.append(trend)
            
            # Count significant trend changes
            for i in range(1, len(trends)):
                if np.sign(trends[i]) != np.sign(trends[i-1]):
                    analysis['trend_changes'] += 1
            
            # Count spikes
            rolling_mean = pd.Series(y).rolling(5).mean()
            rolling_std = pd.Series(y).rolling(5).std()
            spikes = np.abs(y - rolling_mean) > 3 * rolling_std
            analysis['spike_count'] = np.sum(spikes)
            
            # Local correlations
            for i in range(self.window_size, len(y) - self.window_size, self.window_size):
                window = y[i-self.window_size:i+self.window_size]
                if len(window) > 1:
                    correlation = np.corrcoef(range(len(window)), window)[0, 1]
                    if not np.isnan(correlation):
                        analysis['local_correlations'].append(correlation)
            
            return analysis
            
        except Exception as e:
            raise OutlierDetectionError(f"Local pattern analysis failed: {str(e)}")
    
    def get_local_statistics(self) -> Dict[str, any]:
        """Get local outlier analysis statistics"""
        return self.local_stats.copy()
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """Set sensitivity level for outlier detection"""
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
        
        self.sensitivity = sensitivity
        self.logger.debug(f"Updated sensitivity to {sensitivity}")
    
    def set_window_size(self, window_size: int) -> None:
        """Set window size for local analysis"""
        if window_size < 5:
            raise ValueError("Window size must be at least 5")
        
        self.window_size = window_size
        self.is_fitted = False  # Need to refit with new window size
        self.logger.debug(f"Updated window size to {window_size}")