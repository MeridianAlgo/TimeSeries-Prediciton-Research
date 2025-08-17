"""Fractional indicator calculator for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


class FractionalIndicatorCalculator(FeatureEngineer):
    """Calculates indicators with non-integer periods for ultra-precision."""
    
    def __init__(self, fractional_periods: Optional[List[float]] = None):
        """Initialize fractional indicator calculator.
        
        Args:
            fractional_periods: List of fractional periods for moving averages
        """
        self.fractional_periods = fractional_periods or [2.5, 3.7, 5.2, 7.8, 10.3, 13.6, 18.4, 25.1, 34.2]
        self.feature_names = []
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized FractionalIndicatorCalculator with periods: {self.fractional_periods}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fractional indicator features.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with fractional indicator features added
        """
        try:
            self.logger.info("Generating fractional indicator features")
            df = data.copy()
            
            # Validate required columns
            if 'Close' not in df.columns:
                raise FeatureEngineeringError("Close column is required for fractional indicators")
            
            # Check for empty data
            if len(df) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Generate fractional exponential moving averages
            df = self._generate_fractional_emas(df)
            
            # Generate EMA slopes and curvatures
            df = self._generate_ema_derivatives(df)
            
            # Generate price-to-EMA ratios
            df = self._generate_price_ema_ratios(df)
            
            # Generate EMA crossovers and relationships
            df = self._generate_ema_relationships(df)
            
            # Generate adaptive EMAs
            df = self._generate_adaptive_emas(df)
            
            # Generate EMA momentum and acceleration
            df = self._generate_ema_momentum(df)
            
            # Clean up any NaN values
            df = df.ffill().bfill().fillna(0)
            
            # Update feature names
            self._update_feature_names(df, data.columns)
            
            self.logger.info(f"Generated {len(self.feature_names)} fractional indicator features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating fractional indicator features: {str(e)}")
            raise FeatureEngineeringError(f"Fractional indicator feature generation failed: {str(e)}") from e
    
    def _generate_fractional_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate exponential moving averages with fractional periods."""
        self.logger.debug("Generating fractional EMAs")
        
        for period in self.fractional_periods:
            # Calculate alpha for fractional period
            alpha = 2.0 / (period + 1)
            
            # Calculate EMA using exponential smoothing
            ema_values = []
            ema = df['Close'].iloc[0]  # Initialize with first value
            
            for price in df['Close']:
                ema = alpha * price + (1 - alpha) * ema
                ema_values.append(ema)
            
            df[f'ema_precise_{period}'] = ema_values
            
            # Calculate EMA using pandas ewm for comparison/validation
            df[f'ema_pandas_{period}'] = df['Close'].ewm(alpha=alpha).mean()
            
            # Use the more precise manual calculation
            df[f'ema_{period}'] = df[f'ema_precise_{period}']
            
            # Clean up intermediate columns
            df = df.drop([f'ema_precise_{period}', f'ema_pandas_{period}'], axis=1)
        
        return df
    
    def _generate_ema_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA slopes and curvatures."""
        self.logger.debug("Generating EMA derivatives")
        
        for period in self.fractional_periods:
            ema_col = f'ema_{period}'
            if ema_col in df.columns:
                # First derivative (slope)
                df[f'ema_slope_{period}'] = df[ema_col].diff(2)
                
                # Second derivative (curvature)
                df[f'ema_curvature_{period}'] = df[f'ema_slope_{period}'].diff()
                
                # Slope momentum (rate of slope change)
                df[f'ema_slope_momentum_{period}'] = df[f'ema_slope_{period}'].diff()
                
                # Slope strength (absolute slope)
                df[f'ema_slope_strength_{period}'] = np.abs(df[f'ema_slope_{period}'])
                
                # Slope direction consistency
                slope_sign = np.sign(df[f'ema_slope_{period}'])
                df[f'ema_slope_consistency_{period}'] = (
                    slope_sign.rolling(5).apply(
                        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
                    )
                )
                
                # Curvature strength
                df[f'ema_curvature_strength_{period}'] = np.abs(df[f'ema_curvature_{period}'])
                
                # Inflection points (where curvature changes sign)
                curvature_sign = np.sign(df[f'ema_curvature_{period}'])
                df[f'ema_inflection_{period}'] = (curvature_sign != curvature_sign.shift()).astype(int)
        
        return df
    
    def _generate_price_ema_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price-to-EMA ratios and deviations."""
        self.logger.debug("Generating price-EMA ratios")
        
        for period in self.fractional_periods:
            ema_col = f'ema_{period}'
            if ema_col in df.columns:
                # Price to EMA ratio
                df[f'price_ema_ratio_{period}'] = df['Close'] / df[ema_col]
                
                # Price deviation from EMA (percentage)
                df[f'price_ema_deviation_{period}'] = (
                    (df['Close'] - df[ema_col]) / df[ema_col]
                )
                
                # Normalized deviation (z-score)
                deviation = df[f'price_ema_deviation_{period}']
                rolling_std = deviation.rolling(20).std()
                df[f'price_ema_zscore_{period}'] = deviation / (rolling_std + 1e-10)
                
                # Distance from EMA in standard deviations
                price_std = df['Close'].rolling(int(period)).std()
                df[f'price_ema_distance_{period}'] = (
                    (df['Close'] - df[ema_col]) / (price_std + 1e-10)
                )
                
                # EMA support/resistance levels
                df[f'above_ema_{period}'] = (df['Close'] > df[ema_col]).astype(int)
                df[f'below_ema_{period}'] = (df['Close'] < df[ema_col]).astype(int)
                
                # Time above/below EMA
                above_ema = df[f'above_ema_{period}']
                df[f'time_above_ema_{period}'] = (
                    above_ema.groupby((above_ema != above_ema.shift()).cumsum()).cumcount() + 1
                ) * above_ema
                
                below_ema = df[f'below_ema_{period}']
                df[f'time_below_ema_{period}'] = (
                    below_ema.groupby((below_ema != below_ema.shift()).cumsum()).cumcount() + 1
                ) * below_ema
        
        return df
    
    def _generate_ema_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate relationships between different EMA periods."""
        self.logger.debug("Generating EMA relationships")
        
        sorted_periods = sorted(self.fractional_periods)
        
        for i in range(len(sorted_periods) - 1):
            fast_period = sorted_periods[i]
            slow_period = sorted_periods[i + 1]
            
            fast_ema = f'ema_{fast_period}'
            slow_ema = f'ema_{slow_period}'
            
            if fast_ema in df.columns and slow_ema in df.columns:
                # EMA crossover signals
                df[f'ema_cross_{fast_period}_{slow_period}'] = (
                    df[fast_ema] > df[slow_ema]
                ).astype(int)
                
                # EMA distance (spread)
                df[f'ema_distance_{fast_period}_{slow_period}'] = (
                    df[fast_ema] - df[slow_ema]
                ) / df['Close']
                
                # EMA distance momentum
                df[f'ema_distance_momentum_{fast_period}_{slow_period}'] = (
                    df[f'ema_distance_{fast_period}_{slow_period}'].diff()
                )
                
                # EMA convergence/divergence
                distance = df[f'ema_distance_{fast_period}_{slow_period}']
                df[f'ema_convergence_{fast_period}_{slow_period}'] = (
                    np.abs(distance) < np.abs(distance.shift(1))
                ).astype(int)
                
                # EMA crossover momentum
                crossover = df[f'ema_cross_{fast_period}_{slow_period}']
                df[f'ema_crossover_momentum_{fast_period}_{slow_period}'] = (
                    crossover.diff()
                )
        
        return df
    
    def _generate_adaptive_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive EMAs that adjust based on volatility."""
        self.logger.debug("Generating adaptive EMAs")
        
        # Calculate volatility for adaptation
        volatility = df['Close'].pct_change().rolling(10).std()
        
        for period in self.fractional_periods[:5]:  # Only for shorter periods to avoid too many features
            # Adaptive alpha based on volatility
            base_alpha = 2.0 / (period + 1)
            volatility_factor = volatility / volatility.rolling(50).mean()
            adaptive_alpha = base_alpha * (1 + volatility_factor.fillna(1))
            adaptive_alpha = np.clip(adaptive_alpha, 0.01, 0.99)  # Keep within reasonable bounds
            
            # Calculate adaptive EMA
            adaptive_ema_values = []
            ema = df['Close'].iloc[0]
            
            for i, (price, alpha) in enumerate(zip(df['Close'], adaptive_alpha)):
                if pd.isna(alpha):
                    alpha = base_alpha
                ema = alpha * price + (1 - alpha) * ema
                adaptive_ema_values.append(ema)
            
            df[f'adaptive_ema_{period}'] = adaptive_ema_values
            
            # Adaptive EMA vs regular EMA
            regular_ema = f'ema_{period}'
            if regular_ema in df.columns:
                df[f'adaptive_ema_diff_{period}'] = (
                    df[f'adaptive_ema_{period}'] - df[regular_ema]
                ) / df['Close']
        
        return df
    
    def _generate_ema_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA momentum and acceleration features."""
        self.logger.debug("Generating EMA momentum")
        
        for period in self.fractional_periods:
            ema_col = f'ema_{period}'
            if ema_col in df.columns:
                # EMA momentum (rate of change)
                df[f'ema_momentum_{period}'] = df[ema_col].pct_change()
                
                # EMA acceleration
                df[f'ema_acceleration_{period}'] = df[f'ema_momentum_{period}'].diff()
                
                # EMA momentum strength
                df[f'ema_momentum_strength_{period}'] = np.abs(df[f'ema_momentum_{period}'])
                
                # EMA momentum consistency
                momentum = df[f'ema_momentum_{period}']
                momentum_sign = np.sign(momentum)
                df[f'ema_momentum_consistency_{period}'] = (
                    momentum_sign.rolling(5).apply(
                        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
                    )
                )
                
                # EMA momentum vs price momentum
                price_momentum = df['Close'].pct_change()
                df[f'ema_price_momentum_diff_{period}'] = (
                    momentum - price_momentum
                )
                
                # EMA momentum relative to volatility
                volatility = df['Close'].pct_change().rolling(int(period)).std()
                df[f'ema_momentum_vol_ratio_{period}'] = (
                    momentum / (volatility + 1e-10)
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
        """Update feature importance scores.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance scores
        """
        for feature_name, importance in importance_dict.items():
            if feature_name in self.feature_importance:
                self.feature_importance[feature_name] = importance
    
    def get_feature_statistics(self) -> Dict[str, any]:
        """Get statistics about generated features."""
        return {
            'total_features': len(self.feature_names),
            'fractional_periods': self.fractional_periods,
            'feature_categories': {
                'emas': len([f for f in self.feature_names if f.startswith('ema_') and not any(x in f for x in ['slope', 'ratio', 'cross', 'momentum'])]),
                'slopes': len([f for f in self.feature_names if 'slope' in f]),
                'curvatures': len([f for f in self.feature_names if 'curvature' in f]),
                'ratios': len([f for f in self.feature_names if 'ratio' in f or 'deviation' in f]),
                'crossovers': len([f for f in self.feature_names if 'cross' in f or 'distance' in f]),
                'adaptive': len([f for f in self.feature_names if 'adaptive' in f]),
                'momentum': len([f for f in self.feature_names if 'momentum' in f])
            }
        }