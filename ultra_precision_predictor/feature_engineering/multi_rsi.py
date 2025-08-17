"""Multi-timeframe RSI system for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


class MultiRSISystem(FeatureEngineer):
    """Multi-timeframe RSI system with divergence analysis and advanced features."""
    
    def __init__(self, periods: Optional[List[int]] = None):
        """Initialize multi-RSI system.
        
        Args:
            periods: List of RSI periods for calculation
        """
        self.periods = periods or [5, 7, 11, 14, 19, 25, 31]
        self.feature_names = []
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized MultiRSISystem with periods: {self.periods}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-timeframe RSI features.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with RSI features added
        """
        try:
            self.logger.info("Generating multi-timeframe RSI features")
            df = data.copy()
            
            # Validate required columns
            if 'Close' not in df.columns:
                raise FeatureEngineeringError("Close column is required for RSI calculation")
            
            # Check for empty data
            if len(df) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Generate RSI for all periods
            df = self._generate_rsi_indicators(df)
            
            # Generate RSI derivatives and momentum
            df = self._generate_rsi_derivatives(df)
            
            # Generate RSI mean reversion signals
            df = self._generate_mean_reversion_signals(df)
            
            # Generate RSI divergence analysis
            df = self._generate_divergence_analysis(df)
            
            # Generate RSI cycle analysis
            df = self._generate_cycle_analysis(df)
            
            # Generate multi-timeframe RSI relationships
            df = self._generate_rsi_relationships(df)
            
            # Generate RSI regime detection
            df = self._generate_regime_detection(df)
            
            # Clean up any NaN values
            df = df.ffill().bfill().fillna(0)
            
            # Update feature names
            self._update_feature_names(df, data.columns)
            
            self.logger.info(f"Generated {len(self.feature_names)} RSI features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating RSI features: {str(e)}")
            raise FeatureEngineeringError(f"RSI feature generation failed: {str(e)}") from e
    
    def _generate_rsi_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI indicators for all periods."""
        self.logger.debug("Generating RSI indicators")
        
        for period in self.periods:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            df[f'rsi_{period}'] = rsi
            
            # RSI smoothed versions
            df[f'rsi_smooth_{period}'] = rsi.rolling(3).mean()
            df[f'rsi_ema_{period}'] = rsi.ewm(span=5).mean()
            
            # RSI normalized (0-1 scale)
            df[f'rsi_normalized_{period}'] = rsi / 100.0
            
            # RSI centered (-50 to +50)
            df[f'rsi_centered_{period}'] = rsi - 50
            
            # RSI strength (distance from 50)
            df[f'rsi_strength_{period}'] = np.abs(rsi - 50)
            
            # RSI zones
            df[f'rsi_overbought_{period}'] = (rsi > 70).astype(int)
            df[f'rsi_oversold_{period}'] = (rsi < 30).astype(int)
            df[f'rsi_extreme_overbought_{period}'] = (rsi > 80).astype(int)
            df[f'rsi_extreme_oversold_{period}'] = (rsi < 20).astype(int)
            df[f'rsi_neutral_{period}'] = ((rsi >= 40) & (rsi <= 60)).astype(int)
            
            # Time in zones
            overbought = df[f'rsi_overbought_{period}']
            oversold = df[f'rsi_oversold_{period}']
            
            df[f'rsi_time_overbought_{period}'] = (
                overbought.groupby((overbought != overbought.shift()).cumsum()).cumcount() + 1
            ) * overbought
            
            df[f'rsi_time_oversold_{period}'] = (
                oversold.groupby((oversold != oversold.shift()).cumsum()).cumcount() + 1
            ) * oversold
        
        return df
    
    def _generate_rsi_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI derivatives and momentum indicators."""
        self.logger.debug("Generating RSI derivatives")
        
        for period in self.periods:
            rsi_col = f'rsi_{period}'
            
            if rsi_col in df.columns:
                # RSI velocity (first derivative)
                df[f'rsi_velocity_{period}'] = df[rsi_col].diff()
                
                # RSI acceleration (second derivative)
                df[f'rsi_acceleration_{period}'] = df[f'rsi_velocity_{period}'].diff()
                
                # RSI jerk (third derivative)
                df[f'rsi_jerk_{period}'] = df[f'rsi_acceleration_{period}'].diff()
                
                # RSI momentum (rate of change)
                df[f'rsi_momentum_{period}'] = df[rsi_col].pct_change()
                
                # RSI momentum strength
                df[f'rsi_momentum_strength_{period}'] = np.abs(df[f'rsi_momentum_{period}'])
                
                # RSI direction and consistency
                rsi_direction = np.sign(df[f'rsi_velocity_{period}'])
                df[f'rsi_direction_{period}'] = rsi_direction
                
                # RSI direction consistency
                df[f'rsi_direction_consistency_{period}'] = (
                    rsi_direction.rolling(5).apply(
                        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
                    )
                )
                
                # RSI turning points
                df[f'rsi_turning_point_{period}'] = (
                    (df[f'rsi_velocity_{period}'] > 0) & 
                    (df[f'rsi_velocity_{period}'].shift(1) <= 0)
                ).astype(int) - (
                    (df[f'rsi_velocity_{period}'] < 0) & 
                    (df[f'rsi_velocity_{period}'].shift(1) >= 0)
                ).astype(int)
                
                # RSI slope over multiple periods
                for slope_period in [3, 5, 8]:
                    df[f'rsi_slope_{period}_{slope_period}'] = (
                        df[rsi_col].rolling(slope_period).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == slope_period else 0
                        )
                    )
        
        return df
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI mean reversion signals."""
        self.logger.debug("Generating RSI mean reversion signals")
        
        for period in self.periods:
            rsi_col = f'rsi_{period}'
            
            if rsi_col in df.columns:
                # RSI mean and standard deviation
                rsi_mean = df[rsi_col].rolling(20).mean()
                rsi_std = df[rsi_col].rolling(20).std()
                
                # Mean reversion z-score
                df[f'rsi_mean_reversion_{period}'] = (
                    (df[rsi_col] - rsi_mean) / (rsi_std + 1e-10)
                )
                
                # Distance from mean
                df[f'rsi_mean_distance_{period}'] = np.abs(df[rsi_col] - rsi_mean)
                
                # Mean reversion signals
                df[f'rsi_mean_revert_up_{period}'] = (
                    (df[rsi_col] < rsi_mean - rsi_std) & 
                    (df[f'rsi_velocity_{period}'] > 0)
                ).astype(int)
                
                df[f'rsi_mean_revert_down_{period}'] = (
                    (df[rsi_col] > rsi_mean + rsi_std) & 
                    (df[f'rsi_velocity_{period}'] < 0)
                ).astype(int)
                
                # RSI percentile rank
                df[f'rsi_percentile_{period}'] = (
                    df[rsi_col].rolling(100).rank(pct=True)
                )
                
                # Extreme percentile signals
                df[f'rsi_extreme_low_percentile_{period}'] = (
                    df[f'rsi_percentile_{period}'] < 0.1
                ).astype(int)
                
                df[f'rsi_extreme_high_percentile_{period}'] = (
                    df[f'rsi_percentile_{period}'] > 0.9
                ).astype(int)
        
        return df
    
    def _generate_divergence_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI divergence analysis."""
        self.logger.debug("Generating RSI divergence analysis")
        
        # Calculate price slope for divergence comparison
        price_slope_5 = df['Close'].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        price_slope_10 = df['Close'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        
        for period in self.periods:
            rsi_col = f'rsi_{period}'
            rsi_slope_5 = f'rsi_slope_{period}_5'
            
            if rsi_col in df.columns and rsi_slope_5 in df.columns:
                # RSI-Price divergence (5-period)
                df[f'rsi_divergence_5_{period}'] = (
                    (price_slope_5 > 0) & (df[rsi_slope_5] < 0)
                ).astype(int) - (
                    (price_slope_5 < 0) & (df[rsi_slope_5] > 0)
                ).astype(int)
                
                # RSI-Price divergence strength
                df[f'rsi_divergence_strength_5_{period}'] = (
                    np.abs(price_slope_5) * np.abs(df[rsi_slope_5]) * 
                    np.abs(df[f'rsi_divergence_5_{period}'])
                )
                
                # Hidden divergence (trend continuation)
                df[f'rsi_hidden_divergence_5_{period}'] = (
                    (price_slope_5 > 0) & (df[rsi_slope_5] > 0) & 
                    (df[rsi_col] < 50)
                ).astype(int) - (
                    (price_slope_5 < 0) & (df[rsi_slope_5] < 0) & 
                    (df[rsi_col] > 50)
                ).astype(int)
                
                # Divergence confirmation
                df[f'rsi_divergence_confirmed_5_{period}'] = (
                    (df[f'rsi_divergence_5_{period}'].shift(1) != 0) & 
                    (np.sign(df[f'rsi_velocity_{period}']) == np.sign(df[f'rsi_divergence_5_{period}'].shift(1)))
                ).astype(int)
        
        return df
    
    def _generate_cycle_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI cycle analysis."""
        self.logger.debug("Generating RSI cycle analysis")
        
        for period in self.periods:
            rsi_col = f'rsi_{period}'
            
            if rsi_col in df.columns:
                # RSI cycle using sine transformation
                df[f'rsi_cycle_{period}'] = np.sin(2 * np.pi * df[rsi_col] / 100)
                df[f'rsi_cycle_cos_{period}'] = np.cos(2 * np.pi * df[rsi_col] / 100)
                
                # RSI cycle momentum
                df[f'rsi_cycle_momentum_{period}'] = df[f'rsi_cycle_{period}'].diff()
                
                # RSI cycle extremes
                df[f'rsi_cycle_peak_{period}'] = (
                    (df[f'rsi_cycle_{period}'] > df[f'rsi_cycle_{period}'].shift(1)) & 
                    (df[f'rsi_cycle_{period}'] > df[f'rsi_cycle_{period}'].shift(-1))
                ).astype(int)
                
                df[f'rsi_cycle_trough_{period}'] = (
                    (df[f'rsi_cycle_{period}'] < df[f'rsi_cycle_{period}'].shift(1)) & 
                    (df[f'rsi_cycle_{period}'] < df[f'rsi_cycle_{period}'].shift(-1))
                ).astype(int)
                
                # RSI harmonic analysis
                df[f'rsi_harmonic_2_{period}'] = np.sin(4 * np.pi * df[rsi_col] / 100)
                df[f'rsi_harmonic_3_{period}'] = np.sin(6 * np.pi * df[rsi_col] / 100)
                
                # RSI phase analysis
                df[f'rsi_phase_{period}'] = np.arctan2(
                    df[f'rsi_cycle_{period}'], df[f'rsi_cycle_cos_{period}']
                )
                df[f'rsi_phase_velocity_{period}'] = df[f'rsi_phase_{period}'].diff()
        
        return df
    
    def _generate_rsi_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate relationships between different RSI timeframes."""
        self.logger.debug("Generating RSI relationships")
        
        # Compare different periods
        for i in range(len(self.periods) - 1):
            fast_period = self.periods[i]
            slow_period = self.periods[i + 1]
            
            fast_rsi = f'rsi_{fast_period}'
            slow_rsi = f'rsi_{slow_period}'
            
            if fast_rsi in df.columns and slow_rsi in df.columns:
                # RSI spread
                df[f'rsi_spread_{fast_period}_{slow_period}'] = (
                    df[fast_rsi] - df[slow_rsi]
                )
                
                # RSI ratio
                df[f'rsi_ratio_{fast_period}_{slow_period}'] = (
                    df[fast_rsi] / (df[slow_rsi] + 1e-10)
                )
                
                # RSI crossover
                df[f'rsi_crossover_{fast_period}_{slow_period}'] = (
                    df[fast_rsi] > df[slow_rsi]
                ).astype(int)
                
                # RSI crossover momentum
                df[f'rsi_crossover_momentum_{fast_period}_{slow_period}'] = (
                    df[f'rsi_crossover_{fast_period}_{slow_period}'].diff()
                )
                
                # RSI convergence/divergence
                spread = df[f'rsi_spread_{fast_period}_{slow_period}']
                df[f'rsi_convergence_{fast_period}_{slow_period}'] = (
                    np.abs(spread) < np.abs(spread.shift(1))
                ).astype(int)
                
                # RSI alignment (both in same zone)
                fast_overbought = f'rsi_overbought_{fast_period}'
                slow_overbought = f'rsi_overbought_{slow_period}'
                fast_oversold = f'rsi_oversold_{fast_period}'
                slow_oversold = f'rsi_oversold_{slow_period}'
                
                if all(col in df.columns for col in [fast_overbought, slow_overbought, fast_oversold, slow_oversold]):
                    df[f'rsi_aligned_overbought_{fast_period}_{slow_period}'] = (
                        df[fast_overbought] & df[slow_overbought]
                    ).astype(int)
                    
                    df[f'rsi_aligned_oversold_{fast_period}_{slow_period}'] = (
                        df[fast_oversold] & df[slow_oversold]
                    ).astype(int)
        
        return df
    
    def _generate_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI regime detection features."""
        self.logger.debug("Generating RSI regime detection")
        
        for period in self.periods:
            rsi_col = f'rsi_{period}'
            
            if rsi_col in df.columns:
                # RSI regime based on average level
                rsi_ma = df[rsi_col].rolling(50).mean()
                df[f'rsi_regime_{period}'] = (df[rsi_col] > rsi_ma).astype(int)
                
                # RSI volatility regime
                rsi_vol = df[rsi_col].rolling(20).std()
                rsi_vol_ma = rsi_vol.rolling(50).mean()
                df[f'rsi_vol_regime_{period}'] = (rsi_vol > rsi_vol_ma).astype(int)
                
                # RSI trend regime
                rsi_trend = df[rsi_col].rolling(10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
                )
                df[f'rsi_trend_regime_{period}'] = (rsi_trend > 0).astype(int)
                
                # RSI momentum regime
                rsi_momentum = f'rsi_momentum_{period}'
                if rsi_momentum in df.columns:
                    momentum_ma = df[rsi_momentum].rolling(20).mean()
                    df[f'rsi_momentum_regime_{period}'] = (
                        df[rsi_momentum] > momentum_ma
                    ).astype(int)
                
                # RSI extreme regime (frequent extreme readings)
                extreme_count = (
                    df[f'rsi_overbought_{period}'].rolling(20).sum() + 
                    df[f'rsi_oversold_{period}'].rolling(20).sum()
                )
                df[f'rsi_extreme_regime_{period}'] = (extreme_count > 5).astype(int)
        
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
            'feature_categories': {
                'basic_rsi': len([f for f in self.feature_names if f.startswith('rsi_') and not any(x in f for x in ['velocity', 'momentum', 'divergence', 'cycle', 'spread'])]),
                'derivatives': len([f for f in self.feature_names if any(x in f for x in ['velocity', 'acceleration', 'jerk', 'momentum', 'slope'])]),
                'mean_reversion': len([f for f in self.feature_names if any(x in f for x in ['mean_reversion', 'percentile', 'revert'])]),
                'divergence': len([f for f in self.feature_names if 'divergence' in f]),
                'cycles': len([f for f in self.feature_names if any(x in f for x in ['cycle', 'harmonic', 'phase'])]),
                'relationships': len([f for f in self.feature_names if any(x in f for x in ['spread', 'ratio', 'crossover', 'alignment'])]),
                'regimes': len([f for f in self.feature_names if 'regime' in f])
            }
        }