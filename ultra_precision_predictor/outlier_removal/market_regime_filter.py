"""Market regime filter for removing unpredictable periods"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import OutlierDetectionError
from ..core.logging_config import setup_logging


class MarketRegimeFilter:
    """Filter out data from unpredictable market regimes"""
    
    def __init__(self, volatility_threshold: float = 0.7, trend_consistency_threshold: float = 0.3):
        """
        Initialize market regime filter
        
        Args:
            volatility_threshold: Percentile threshold for high volatility filtering
            trend_consistency_threshold: Threshold for trend consistency
        """
        self.volatility_threshold = volatility_threshold
        self.trend_consistency_threshold = trend_consistency_threshold
        self.logger = setup_logging()
        self.is_fitted = False
        
        # Regime statistics
        self.regime_stats = {
            'high_volatility_periods': 0,
            'inconsistent_trend_periods': 0,
            'market_stress_periods': 0,
            'total_filtered': 0
        }
        
        # Fitted parameters
        self.volatility_cutoff = None
        self.trend_consistency_cutoff = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit market regime filter"""
        try:
            self.logger.debug("Fitting market regime filter...")
            
            # Calculate volatility measures
            returns = np.diff(y) / y[:-1]
            rolling_volatility = pd.Series(returns).rolling(20).std()
            
            # Set volatility cutoff
            self.volatility_cutoff = np.percentile(rolling_volatility.dropna(), 
                                                 self.volatility_threshold * 100)
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(y)
            self.trend_consistency_cutoff = np.percentile(trend_consistency.dropna(),
                                                        self.trend_consistency_threshold * 100)
            
            self.is_fitted = True
            self.logger.debug(f"Regime filter fitted - Vol cutoff: {self.volatility_cutoff:.6f}, "
                            f"Trend cutoff: {self.trend_consistency_cutoff:.6f}")
            
        except Exception as e:
            raise OutlierDetectionError(f"Failed to fit regime filter: {str(e)}")
    
    def detect_outliers(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Detect outliers based on market regime analysis"""
        try:
            if not self.is_fitted:
                self.fit(X, y)
            
            mask = np.ones(len(y), dtype=bool)
            
            # High volatility regime filtering
            mask = self._filter_high_volatility_periods(y, mask)
            
            # Trend inconsistency filtering
            mask = self._filter_inconsistent_trends(y, mask)
            
            # Market stress period filtering
            mask = self._filter_market_stress_periods(y, mask)
            
            # Update statistics
            self.regime_stats['total_filtered'] = np.sum(~mask)
            
            self.logger.debug(f"Regime filter removed {self.regime_stats['total_filtered']} samples")
            
            return mask
            
        except Exception as e:
            raise OutlierDetectionError(f"Regime outlier detection failed: {str(e)}")
    
    def _filter_high_volatility_periods(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Filter periods of extremely high volatility"""
        try:
            returns = np.diff(y) / y[:-1]
            rolling_volatility = pd.Series(returns).rolling(20).std()
            
            # Mark high volatility periods
            high_vol_mask = rolling_volatility > self.volatility_cutoff
            
            # Extend the mask to include surrounding periods
            extended_mask = np.zeros(len(y), dtype=bool)
            for i in range(len(high_vol_mask)):
                if high_vol_mask.iloc[i]:
                    # Mark current and surrounding periods
                    start = max(0, i - 5)
                    end = min(len(y), i + 15)  # Extend further forward
                    extended_mask[start:end] = True
            
            # Apply to main mask
            mask = mask & ~extended_mask
            
            self.regime_stats['high_volatility_periods'] = np.sum(extended_mask)
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"High volatility filtering failed: {str(e)}")
            return mask
    
    def _filter_inconsistent_trends(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Filter periods with inconsistent trends"""
        try:
            trend_consistency = self._calculate_trend_consistency(y)
            
            # Mark inconsistent trend periods
            inconsistent_mask = trend_consistency < self.trend_consistency_cutoff
            
            # Convert to numpy array and handle NaN values
            inconsistent_array = inconsistent_mask.values if hasattr(inconsistent_mask, 'values') else inconsistent_mask
            inconsistent_array = np.nan_to_num(inconsistent_array, nan=False)
            
            # Ensure same length as mask
            if len(inconsistent_array) < len(mask):
                # Pad with False values
                padding = np.zeros(len(mask) - len(inconsistent_array), dtype=bool)
                inconsistent_array = np.concatenate([padding, inconsistent_array])
            elif len(inconsistent_array) > len(mask):
                # Truncate
                inconsistent_array = inconsistent_array[:len(mask)]
            
            # Apply to main mask
            mask = mask & ~inconsistent_array
            
            self.regime_stats['inconsistent_trend_periods'] = np.sum(inconsistent_array)
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"Trend consistency filtering failed: {str(e)}")
            return mask
    
    def _filter_market_stress_periods(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Filter periods of market stress (gaps, extreme moves)"""
        try:
            stress_mask = np.zeros(len(y), dtype=bool)
            
            # Detect large gaps (assuming daily data)
            returns = np.diff(y) / y[:-1]
            large_moves = np.abs(returns) > 3 * np.std(returns)
            
            # Mark stress periods around large moves
            for i in range(len(large_moves)):
                if large_moves[i]:
                    start = max(0, i - 2)
                    end = min(len(y), i + 8)  # Longer recovery period
                    stress_mask[start:end] = True
            
            # Detect consecutive extreme moves
            consecutive_extreme = 0
            for i in range(len(returns)):
                if np.abs(returns[i]) > 2 * np.std(returns):
                    consecutive_extreme += 1
                    if consecutive_extreme >= 3:  # 3+ consecutive extreme moves
                        start = max(0, i - consecutive_extreme)
                        end = min(len(y), i + 5)
                        stress_mask[start:end] = True
                else:
                    consecutive_extreme = 0
            
            # Apply to main mask
            mask = mask & ~stress_mask
            
            self.regime_stats['market_stress_periods'] = np.sum(stress_mask)
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"Market stress filtering failed: {str(e)}")
            return mask
    
    def _calculate_trend_consistency(self, y: np.ndarray) -> pd.Series:
        """Calculate trend consistency measure"""
        try:
            # Calculate rolling trend slopes
            window_size = 20
            trend_slopes = []
            
            for i in range(window_size, len(y)):
                window = y[i-window_size:i]
                slope = np.polyfit(range(len(window)), window, 1)[0]
                trend_slopes.append(slope)
            
            trend_slopes = pd.Series(trend_slopes)
            
            # Calculate consistency as rolling correlation of slopes
            consistency = trend_slopes.rolling(10).apply(
                lambda x: np.corrcoef(range(len(x)), x)[0, 1] if len(x) > 1 else 0
            )
            
            # Handle NaN values
            consistency = consistency.fillna(0)
            
            return consistency
            
        except Exception as e:
            self.logger.warning(f"Trend consistency calculation failed: {str(e)}")
            return pd.Series(np.zeros(len(y) - 20))
    
    def analyze_market_regimes(self, y: np.ndarray) -> Dict[str, any]:
        """Analyze different market regimes in the data"""
        try:
            analysis = {
                'volatility_regimes': {},
                'trend_regimes': {},
                'regime_transitions': 0,
                'stable_periods': 0
            }
            
            # Volatility regime analysis
            returns = np.diff(y) / y[:-1]
            rolling_vol = pd.Series(returns).rolling(20).std()
            
            vol_percentiles = [25, 50, 75, 90]
            for p in vol_percentiles:
                threshold = np.percentile(rolling_vol.dropna(), p)
                periods = np.sum(rolling_vol > threshold)
                analysis['volatility_regimes'][f'above_{p}th_percentile'] = periods
            
            # Trend regime analysis
            trend_consistency = self._calculate_trend_consistency(y)
            
            analysis['trend_regimes'] = {
                'strong_trend': np.sum(np.abs(trend_consistency) > 0.7),
                'weak_trend': np.sum((np.abs(trend_consistency) > 0.3) & (np.abs(trend_consistency) <= 0.7)),
                'no_trend': np.sum(np.abs(trend_consistency) <= 0.3)
            }
            
            # Count regime transitions
            high_vol_periods = rolling_vol > np.percentile(rolling_vol.dropna(), 70)
            transitions = np.sum(np.diff(high_vol_periods.astype(int)) != 0)
            analysis['regime_transitions'] = transitions
            
            # Stable periods (low volatility + consistent trend)
            stable = (rolling_vol < np.percentile(rolling_vol.dropna(), 30)) & \
                    (np.abs(trend_consistency) > 0.5)
            analysis['stable_periods'] = np.sum(stable)
            
            return analysis
            
        except Exception as e:
            raise OutlierDetectionError(f"Market regime analysis failed: {str(e)}")
    
    def get_regime_statistics(self) -> Dict[str, any]:
        """Get market regime filtering statistics"""
        return self.regime_stats.copy()
    
    def set_volatility_threshold(self, threshold: float) -> None:
        """Set volatility threshold for filtering"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Volatility threshold must be between 0.0 and 1.0")
        
        self.volatility_threshold = threshold
        self.is_fitted = False  # Need to refit
        self.logger.debug(f"Updated volatility threshold to {threshold}")
    
    def set_trend_consistency_threshold(self, threshold: float) -> None:
        """Set trend consistency threshold for filtering"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Trend consistency threshold must be between 0.0 and 1.0")
        
        self.trend_consistency_threshold = threshold
        self.is_fitted = False  # Need to refit
        self.logger.debug(f"Updated trend consistency threshold to {threshold}")