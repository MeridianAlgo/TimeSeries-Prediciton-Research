"""Advanced Bollinger Bands system for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


class AdvancedBollingerBands(FeatureEngineer):
    """Advanced Bollinger Bands with multiple periods and standard deviation multipliers."""
    
    def __init__(self, periods: Optional[List[int]] = None, multipliers: Optional[List[float]] = None):
        """Initialize advanced Bollinger Bands calculator.
        
        Args:
            periods: List of periods for Bollinger Bands calculation
            multipliers: List of standard deviation multipliers including golden ratio
        """
        self.periods = periods or [10, 20, 50]
        self.multipliers = multipliers or [0.5, 1.0, 1.618, 2.0, 2.618]  # Including golden ratio
        self.feature_names = []
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized AdvancedBollingerBands with periods: {self.periods}, multipliers: {self.multipliers}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced Bollinger Bands features.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with Bollinger Bands features added
        """
        try:
            self.logger.info("Generating advanced Bollinger Bands features")
            df = data.copy()
            
            # Validate required columns
            if 'Close' not in df.columns:
                raise FeatureEngineeringError("Close column is required for Bollinger Bands")
            
            # Check for empty data
            if len(df) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Generate Bollinger Bands for all period/multiplier combinations
            df = self._generate_bollinger_bands(df)
            
            # Generate band position and dynamics
            df = self._generate_band_dynamics(df)
            
            # Generate squeeze and expansion indicators
            df = self._generate_squeeze_indicators(df)
            
            # Generate band momentum and trends
            df = self._generate_band_momentum(df)
            
            # Generate multi-timeframe band relationships
            df = self._generate_band_relationships(df)
            
            # Generate adaptive bands
            df = self._generate_adaptive_bands(df)
            
            # Clean up any NaN values
            df = df.ffill().bfill().fillna(0)
            
            # Update feature names
            self._update_feature_names(df, data.columns)
            
            self.logger.info(f"Generated {len(self.feature_names)} Bollinger Bands features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Bands features: {str(e)}")
            raise FeatureEngineeringError(f"Bollinger Bands feature generation failed: {str(e)}") from e
    
    def _generate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic Bollinger Bands for all period/multiplier combinations."""
        self.logger.debug("Generating basic Bollinger Bands")
        
        for period in self.periods:
            # Calculate SMA and standard deviation
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            
            df[f'bb_sma_{period}'] = sma
            df[f'bb_std_{period}'] = std
            
            for multiplier in self.multipliers:
                # Upper and lower bands
                df[f'bb_upper_{period}_{multiplier}'] = sma + (multiplier * std)
                df[f'bb_lower_{period}_{multiplier}'] = sma - (multiplier * std)
                
                # Band width (normalized by SMA)
                df[f'bb_width_{period}_{multiplier}'] = (
                    (df[f'bb_upper_{period}_{multiplier}'] - df[f'bb_lower_{period}_{multiplier}']) / sma
                )
                
                # Price position within bands (0 = lower band, 1 = upper band)
                band_range = df[f'bb_upper_{period}_{multiplier}'] - df[f'bb_lower_{period}_{multiplier}']
                df[f'bb_position_{period}_{multiplier}'] = (
                    (df['Close'] - df[f'bb_lower_{period}_{multiplier}']) / (band_range + 1e-10)
                )
                
                # Distance from bands (normalized)
                df[f'bb_upper_distance_{period}_{multiplier}'] = (
                    (df[f'bb_upper_{period}_{multiplier}'] - df['Close']) / df['Close']
                )
                df[f'bb_lower_distance_{period}_{multiplier}'] = (
                    (df['Close'] - df[f'bb_lower_{period}_{multiplier}']) / df['Close']
                )
        
        return df
    
    def _generate_band_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate band dynamics and breakout indicators."""
        self.logger.debug("Generating band dynamics")
        
        for period in self.periods:
            for multiplier in self.multipliers:
                upper_col = f'bb_upper_{period}_{multiplier}'
                lower_col = f'bb_lower_{period}_{multiplier}'
                position_col = f'bb_position_{period}_{multiplier}'
                
                if all(col in df.columns for col in [upper_col, lower_col, position_col]):
                    # Band touches and breakouts
                    df[f'bb_upper_touch_{period}_{multiplier}'] = (
                        df['Close'] >= df[upper_col] * 0.999
                    ).astype(int)
                    df[f'bb_lower_touch_{period}_{multiplier}'] = (
                        df['Close'] <= df[lower_col] * 1.001
                    ).astype(int)
                    
                    # Band breakouts
                    df[f'bb_upper_breakout_{period}_{multiplier}'] = (
                        df['Close'] > df[upper_col]
                    ).astype(int)
                    df[f'bb_lower_breakout_{period}_{multiplier}'] = (
                        df['Close'] < df[lower_col]
                    ).astype(int)
                    
                    # Time since last touch
                    upper_touches = df[f'bb_upper_touch_{period}_{multiplier}']
                    lower_touches = df[f'bb_lower_touch_{period}_{multiplier}']
                    
                    df[f'bb_time_since_upper_{period}_{multiplier}'] = (
                        upper_touches.eq(0).groupby(upper_touches.ne(0).cumsum()).cumsum()
                    )
                    df[f'bb_time_since_lower_{period}_{multiplier}'] = (
                        lower_touches.eq(0).groupby(lower_touches.ne(0).cumsum()).cumsum()
                    )
                    
                    # Position momentum
                    df[f'bb_position_momentum_{period}_{multiplier}'] = (
                        df[position_col].diff()
                    )
                    
                    # Position acceleration
                    df[f'bb_position_acceleration_{period}_{multiplier}'] = (
                        df[f'bb_position_momentum_{period}_{multiplier}'].diff()
                    )
                    
                    # Extreme positions
                    df[f'bb_extreme_high_{period}_{multiplier}'] = (
                        df[position_col] > 0.95
                    ).astype(int)
                    df[f'bb_extreme_low_{period}_{multiplier}'] = (
                        df[position_col] < 0.05
                    ).astype(int)
        
        return df
    
    def _generate_squeeze_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate squeeze and expansion indicators."""
        self.logger.debug("Generating squeeze indicators")
        
        for period in self.periods:
            for multiplier in self.multipliers:
                width_col = f'bb_width_{period}_{multiplier}'
                
                if width_col in df.columns:
                    # Squeeze detection (width below historical average)
                    width_ma = df[width_col].rolling(50).mean()
                    width_std = df[width_col].rolling(50).std()
                    
                    df[f'bb_squeeze_{period}_{multiplier}'] = (
                        df[width_col] < (width_ma - 0.5 * width_std)
                    ).astype(int)
                    
                    # Expansion detection
                    df[f'bb_expansion_{period}_{multiplier}'] = (
                        df[width_col] > (width_ma + 0.5 * width_std)
                    ).astype(int)
                    
                    # Squeeze intensity
                    df[f'bb_squeeze_intensity_{period}_{multiplier}'] = (
                        (width_ma - df[width_col]) / (width_std + 1e-10)
                    )
                    
                    # Width momentum
                    df[f'bb_width_momentum_{period}_{multiplier}'] = (
                        df[width_col].pct_change()
                    )
                    
                    # Width acceleration
                    df[f'bb_width_acceleration_{period}_{multiplier}'] = (
                        df[f'bb_width_momentum_{period}_{multiplier}'].diff()
                    )
                    
                    # Squeeze duration
                    squeeze_signal = df[f'bb_squeeze_{period}_{multiplier}']
                    df[f'bb_squeeze_duration_{period}_{multiplier}'] = (
                        squeeze_signal.groupby((squeeze_signal != squeeze_signal.shift()).cumsum()).cumcount() + 1
                    ) * squeeze_signal
                    
                    # Width percentile rank
                    df[f'bb_width_rank_{period}_{multiplier}'] = (
                        df[width_col].rolling(100).rank(pct=True)
                    )
        
        return df
    
    def _generate_band_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate band momentum and trend indicators."""
        self.logger.debug("Generating band momentum")
        
        for period in self.periods:
            sma_col = f'bb_sma_{period}'
            
            if sma_col in df.columns:
                # SMA momentum and trend
                df[f'bb_sma_momentum_{period}'] = df[sma_col].pct_change()
                df[f'bb_sma_trend_{period}'] = (
                    df[sma_col] > df[sma_col].shift(1)
                ).astype(int)
                
                # SMA slope
                df[f'bb_sma_slope_{period}'] = df[sma_col].diff(3)
                
                # SMA acceleration
                df[f'bb_sma_acceleration_{period}'] = df[f'bb_sma_slope_{period}'].diff()
                
                for multiplier in self.multipliers:
                    upper_col = f'bb_upper_{period}_{multiplier}'
                    lower_col = f'bb_lower_{period}_{multiplier}'
                    
                    if upper_col in df.columns and lower_col in df.columns:
                        # Band momentum
                        df[f'bb_upper_momentum_{period}_{multiplier}'] = (
                            df[upper_col].pct_change()
                        )
                        df[f'bb_lower_momentum_{period}_{multiplier}'] = (
                            df[lower_col].pct_change()
                        )
                        
                        # Band slope
                        df[f'bb_upper_slope_{period}_{multiplier}'] = (
                            df[upper_col].diff(3)
                        )
                        df[f'bb_lower_slope_{period}_{multiplier}'] = (
                            df[lower_col].diff(3)
                        )
                        
                        # Band convergence/divergence
                        band_center = (df[upper_col] + df[lower_col]) / 2
                        df[f'bb_center_{period}_{multiplier}'] = band_center
                        df[f'bb_center_momentum_{period}_{multiplier}'] = (
                            band_center.pct_change()
                        )
        
        return df
    
    def _generate_band_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate relationships between different band configurations."""
        self.logger.debug("Generating band relationships")
        
        # Compare different periods with same multiplier
        for multiplier in self.multipliers:
            for i in range(len(self.periods) - 1):
                fast_period = self.periods[i]
                slow_period = self.periods[i + 1]
                
                fast_width = f'bb_width_{fast_period}_{multiplier}'
                slow_width = f'bb_width_{slow_period}_{multiplier}'
                fast_pos = f'bb_position_{fast_period}_{multiplier}'
                slow_pos = f'bb_position_{slow_period}_{multiplier}'
                
                if all(col in df.columns for col in [fast_width, slow_width, fast_pos, slow_pos]):
                    # Width ratio
                    df[f'bb_width_ratio_{fast_period}_{slow_period}_{multiplier}'] = (
                        df[fast_width] / (df[slow_width] + 1e-10)
                    )
                    
                    # Position difference
                    df[f'bb_position_diff_{fast_period}_{slow_period}_{multiplier}'] = (
                        df[fast_pos] - df[slow_pos]
                    )
                    
                    # Squeeze alignment
                    fast_squeeze = f'bb_squeeze_{fast_period}_{multiplier}'
                    slow_squeeze = f'bb_squeeze_{slow_period}_{multiplier}'
                    
                    if fast_squeeze in df.columns and slow_squeeze in df.columns:
                        df[f'bb_squeeze_alignment_{fast_period}_{slow_period}_{multiplier}'] = (
                            df[fast_squeeze] & df[slow_squeeze]
                        ).astype(int)
        
        # Compare different multipliers with same period
        for period in self.periods:
            for i in range(len(self.multipliers) - 1):
                tight_mult = self.multipliers[i]
                wide_mult = self.multipliers[i + 1]
                
                tight_pos = f'bb_position_{period}_{tight_mult}'
                wide_pos = f'bb_position_{period}_{wide_mult}'
                
                if tight_pos in df.columns and wide_pos in df.columns:
                    # Position consistency across multipliers
                    df[f'bb_position_consistency_{period}_{tight_mult}_{wide_mult}'] = (
                        np.abs(df[tight_pos] - df[wide_pos])
                    )
                    
                    # Multi-band breakout
                    tight_breakout_up = f'bb_upper_breakout_{period}_{tight_mult}'
                    wide_breakout_up = f'bb_upper_breakout_{period}_{wide_mult}'
                    
                    if tight_breakout_up in df.columns and wide_breakout_up in df.columns:
                        df[f'bb_multi_breakout_up_{period}_{tight_mult}_{wide_mult}'] = (
                            df[tight_breakout_up] & ~df[wide_breakout_up]
                        ).astype(int)
        
        return df
    
    def _generate_adaptive_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive Bollinger Bands based on volatility."""
        self.logger.debug("Generating adaptive bands")
        
        # Calculate volatility for adaptation
        volatility = df['Close'].pct_change().rolling(20).std()
        vol_ma = volatility.rolling(50).mean()
        vol_ratio = volatility / (vol_ma + 1e-10)
        
        for period in self.periods[:2]:  # Only for shorter periods to avoid too many features
            sma = df['Close'].rolling(period).mean()
            base_std = df['Close'].rolling(period).std()
            
            # Adaptive standard deviation
            adaptive_std = base_std * (0.5 + 0.5 * vol_ratio)
            
            for multiplier in [1.0, 2.0]:  # Only key multipliers
                # Adaptive bands
                df[f'bb_adaptive_upper_{period}_{multiplier}'] = sma + (multiplier * adaptive_std)
                df[f'bb_adaptive_lower_{period}_{multiplier}'] = sma - (multiplier * adaptive_std)
                
                # Adaptive position
                adaptive_range = (
                    df[f'bb_adaptive_upper_{period}_{multiplier}'] - 
                    df[f'bb_adaptive_lower_{period}_{multiplier}']
                )
                df[f'bb_adaptive_position_{period}_{multiplier}'] = (
                    (df['Close'] - df[f'bb_adaptive_lower_{period}_{multiplier}']) / 
                    (adaptive_range + 1e-10)
                )
                
                # Difference from regular bands
                regular_pos = f'bb_position_{period}_{multiplier}'
                if regular_pos in df.columns:
                    df[f'bb_adaptive_diff_{period}_{multiplier}'] = (
                        df[f'bb_adaptive_position_{period}_{multiplier}'] - df[regular_pos]
                    )
        
        return df
    
    def _update_feature_names(self, df: pd.DataFrame, original_columns: pd.Index) -> None:
        """Update the list of generated feature names."""
        self.feature_names = [col for col in df.columns if col not in original_columns]
        
        # Initialize feature importance (will be updated during training)
        self.feature_importance = {name: 0.0 for name in self.feature_names}
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings."""
        return self.feature_importance.copy()
    
    def update_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """Update feature importance scores."""
        for feature_name, importance in importance_dict.items():
            if feature_name in self.feature_importance:
                self.feature_importance[feature_name] = importance
    
    def get_feature_statistics(self) -> Dict[str, any]:
        """Get statistics about generated features."""
        return {
            'total_features': len(self.feature_names),
            'periods': self.periods,
            'multipliers': self.multipliers,
            'feature_categories': {
                'basic_bands': len([f for f in self.feature_names if any(x in f for x in ['bb_upper', 'bb_lower', 'bb_width', 'bb_position'])]),
                'dynamics': len([f for f in self.feature_names if any(x in f for x in ['touch', 'breakout', 'momentum', 'acceleration'])]),
                'squeeze': len([f for f in self.feature_names if 'squeeze' in f or 'expansion' in f]),
                'trends': len([f for f in self.feature_names if any(x in f for x in ['slope', 'trend', 'sma'])]),
                'relationships': len([f for f in self.feature_names if any(x in f for x in ['ratio', 'diff', 'alignment', 'consistency'])]),
                'adaptive': len([f for f in self.feature_names if 'adaptive' in f])
            }
        }