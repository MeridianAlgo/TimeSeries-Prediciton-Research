"""Micro-pattern extraction system for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


class MicroPatternExtractor(FeatureEngineer):
    """Extracts micro-patterns from 1-3 bar price movements with sub-tick precision."""
    
    def __init__(self, lookback_periods: Optional[List[int]] = None):
        """Initialize micro-pattern extractor.
        
        Args:
            lookback_periods: List of lookback periods for momentum analysis
        """
        self.lookback_periods = lookback_periods or [1, 2, 3, 5, 8, 13]
        self.feature_names = []
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized MicroPatternExtractor with lookbacks: {self.lookback_periods}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate micro-pattern features from price data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with micro-pattern features added
        """
        try:
            self.logger.info("Generating micro-pattern features")
            df = data.copy()
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise FeatureEngineeringError(f"Missing required columns: {missing_cols}")
            
            # Check for empty data
            if len(df) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Generate micro-return features
            df = self._generate_micro_returns(df)
            
            # Generate price momentum features
            df = self._generate_price_momentum(df)
            
            # Generate momentum acceleration features
            df = self._generate_momentum_acceleration(df)
            
            # Generate momentum consistency features
            df = self._generate_momentum_consistency(df)
            
            # Generate intraday pattern features
            df = self._generate_intraday_patterns(df)
            
            # Generate price velocity and acceleration
            df = self._generate_price_dynamics(df)
            
            # Generate micro-volatility features
            df = self._generate_micro_volatility(df)
            
            # Clean up any NaN values
            df = df.ffill().bfill().fillna(0)
            
            # Update feature names
            self._update_feature_names(df, data.columns)
            
            self.logger.info(f"Generated {len(self.feature_names)} micro-pattern features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating micro-pattern features: {str(e)}")
            raise FeatureEngineeringError(f"Micro-pattern feature generation failed: {str(e)}") from e
    
    def _generate_micro_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate micro-return features with sub-tick precision."""
        self.logger.debug("Generating micro-return features")
        
        # Basic micro-returns
        df['micro_return_1'] = df['Close'].pct_change(1)
        df['micro_return_2'] = df['Close'].pct_change(2) 
        df['micro_return_3'] = df['Close'].pct_change(3)
        
        # Log returns for better statistical properties
        df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_2'] = np.log(df['Close'] / df['Close'].shift(2))
        df['log_return_3'] = np.log(df['Close'] / df['Close'].shift(3))
        
        # Normalized returns (z-score)
        for period in [1, 2, 3]:
            returns = df[f'micro_return_{period}']
            rolling_mean = returns.rolling(20).mean()
            rolling_std = returns.rolling(20).std()
            df[f'normalized_return_{period}'] = (returns - rolling_mean) / (rolling_std + 1e-10)
        
        # Return ratios
        df['return_ratio_1_2'] = df['micro_return_1'] / (df['micro_return_2'] + 1e-10)
        df['return_ratio_2_3'] = df['micro_return_2'] / (df['micro_return_3'] + 1e-10)
        df['return_ratio_1_3'] = df['micro_return_1'] / (df['micro_return_3'] + 1e-10)
        
        return df
    
    def _generate_price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price momentum features at multiple scales."""
        self.logger.debug("Generating price momentum features")
        
        for lookback in self.lookback_periods:
            # Basic momentum
            df[f'price_momentum_{lookback}'] = (
                df['Close'] - df['Close'].shift(lookback)
            ) / df['Close'].shift(lookback)
            
            # Momentum strength (absolute value)
            df[f'momentum_strength_{lookback}'] = np.abs(df[f'price_momentum_{lookback}'])
            
            # Momentum direction (sign)
            df[f'momentum_direction_{lookback}'] = np.sign(df[f'price_momentum_{lookback}'])
            
            # Momentum persistence (how long momentum has been in same direction)
            momentum_sign = df[f'momentum_direction_{lookback}']
            df[f'momentum_persistence_{lookback}'] = (
                momentum_sign.groupby((momentum_sign != momentum_sign.shift()).cumsum()).cumcount() + 1
            )
            
            # Momentum relative to volatility
            volatility = df['Close'].rolling(lookback).std()
            df[f'momentum_vol_ratio_{lookback}'] = (
                df[f'price_momentum_{lookback}'] / (volatility + 1e-10)
            )
        
        return df
    
    def _generate_momentum_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum acceleration features."""
        self.logger.debug("Generating momentum acceleration features")
        
        for lookback in self.lookback_periods:
            momentum_col = f'price_momentum_{lookback}'
            if momentum_col in df.columns:
                # First derivative (acceleration)
                df[f'momentum_accel_{lookback}'] = df[momentum_col].diff()
                
                # Second derivative (jerk)
                df[f'momentum_jerk_{lookback}'] = df[f'momentum_accel_{lookback}'].diff()
                
                # Acceleration strength
                df[f'accel_strength_{lookback}'] = np.abs(df[f'momentum_accel_{lookback}'])
                
                # Acceleration direction change
                accel_sign = np.sign(df[f'momentum_accel_{lookback}'])
                df[f'accel_direction_change_{lookback}'] = (accel_sign != accel_sign.shift()).astype(int)
                
                # Momentum turning points (where acceleration changes sign)
                df[f'momentum_turning_point_{lookback}'] = (
                    (df[f'momentum_accel_{lookback}'] > 0) & 
                    (df[f'momentum_accel_{lookback}'].shift(1) <= 0)
                ).astype(int) - (
                    (df[f'momentum_accel_{lookback}'] < 0) & 
                    (df[f'momentum_accel_{lookback}'].shift(1) >= 0)
                ).astype(int)
        
        return df
    
    def _generate_momentum_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum consistency features."""
        self.logger.debug("Generating momentum consistency features")
        
        for lookback in self.lookback_periods:
            momentum_col = f'price_momentum_{lookback}'
            if momentum_col in df.columns:
                # Rolling standard deviation of momentum
                df[f'momentum_consistency_{lookback}'] = (
                    df[momentum_col].rolling(5).std()
                )
                
                # Momentum coefficient of variation
                momentum_mean = df[momentum_col].rolling(5).mean()
                momentum_std = df[momentum_col].rolling(5).std()
                df[f'momentum_cv_{lookback}'] = (
                    momentum_std / (np.abs(momentum_mean) + 1e-10)
                )
                
                # Momentum trend consistency (how often momentum is in same direction)
                momentum_sign = np.sign(df[momentum_col])
                df[f'momentum_trend_consistency_{lookback}'] = (
                    momentum_sign.rolling(5).apply(
                        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
                    )
                )
                
                # Momentum reversal frequency
                df[f'momentum_reversal_freq_{lookback}'] = (
                    (momentum_sign != momentum_sign.shift()).rolling(10).sum()
                )
        
        return df
    
    def _generate_intraday_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate intraday pattern features."""
        self.logger.debug("Generating intraday pattern features")
        
        # OHLC relationships
        df['body_size'] = np.abs(df['Close'] - df['Open']) / df['Close']
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        df['total_range'] = (df['High'] - df['Low']) / df['Close']
        
        # Body position within range
        df['body_position'] = (
            (np.minimum(df['Open'], df['Close']) - df['Low']) / 
            (df['High'] - df['Low'] + 1e-10)
        )
        
        # Candle patterns
        df['doji'] = (df['body_size'] < 0.001).astype(int)
        df['hammer'] = (
            (df['lower_shadow'] > 2 * df['body_size']) & 
            (df['upper_shadow'] < df['body_size'])
        ).astype(int)
        df['shooting_star'] = (
            (df['upper_shadow'] > 2 * df['body_size']) & 
            (df['lower_shadow'] < df['body_size'])
        ).astype(int)
        
        # Gap analysis
        df['gap'] = df['Open'] - df['Close'].shift(1)
        df['gap_pct'] = df['gap'] / df['Close'].shift(1)
        df['gap_size_normalized'] = df['gap'] / df['Close'].rolling(20).std()
        
        # Gap types
        df['gap_up'] = (df['gap'] > 0).astype(int)
        df['gap_down'] = (df['gap'] < 0).astype(int)
        df['gap_filled'] = (
            ((df['gap'] > 0) & (df['Low'] <= df['Close'].shift(1))) |
            ((df['gap'] < 0) & (df['High'] >= df['Close'].shift(1)))
        ).astype(int)
        
        return df
    
    def _generate_price_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price velocity and acceleration features."""
        self.logger.debug("Generating price dynamics features")
        
        # Price change (velocity)
        df['price_change'] = df['Close'].diff()
        df['price_change_pct'] = df['price_change'] / df['Close'].shift(1)
        
        # Price velocity (rate of change)
        df['price_velocity'] = df['price_change'].diff()
        df['price_velocity_pct'] = df['price_velocity'] / df['Close'].shift(2)
        
        # Price acceleration (rate of velocity change)
        df['price_acceleration'] = df['price_velocity'].diff()
        df['price_acceleration_pct'] = df['price_acceleration'] / df['Close'].shift(3)
        
        # Velocity and acceleration strength
        df['velocity_strength'] = np.abs(df['price_velocity'])
        df['acceleration_strength'] = np.abs(df['price_acceleration'])
        
        # Velocity consistency
        df['velocity_consistency'] = df['price_velocity'].rolling(5).std()
        
        # Speed (magnitude of velocity)
        df['price_speed'] = np.sqrt(df['price_velocity']**2)
        
        return df
    
    def _generate_micro_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate micro-volatility features."""
        self.logger.debug("Generating micro-volatility features")
        
        # Micro-volatility measures
        for period in [2, 3, 5]:
            returns = df['micro_return_1']
            df[f'micro_volatility_{period}'] = returns.rolling(period).std()
            
            # Volatility of volatility
            df[f'vol_of_vol_{period}'] = df[f'micro_volatility_{period}'].rolling(5).std()
            
            # Volatility momentum
            df[f'vol_momentum_{period}'] = df[f'micro_volatility_{period}'].pct_change()
            
            # Volatility relative to long-term average
            long_term_vol = returns.rolling(50).std()
            df[f'vol_ratio_{period}'] = (
                df[f'micro_volatility_{period}'] / (long_term_vol + 1e-10)
            )
        
        # Intraday volatility
        df['intraday_volatility'] = (df['High'] - df['Low']) / df['Close']
        df['overnight_volatility'] = np.abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Volatility clustering
        high_vol_threshold = df['micro_volatility_5'].rolling(50).quantile(0.8)
        df['vol_clustering'] = (df['micro_volatility_5'] > high_vol_threshold).astype(int)
        
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
            'lookback_periods': self.lookback_periods,
            'feature_categories': {
                'micro_returns': len([f for f in self.feature_names if 'return' in f]),
                'momentum': len([f for f in self.feature_names if 'momentum' in f]),
                'acceleration': len([f for f in self.feature_names if 'accel' in f]),
                'patterns': len([f for f in self.feature_names if any(p in f for p in ['doji', 'hammer', 'gap'])]),
                'dynamics': len([f for f in self.feature_names if any(d in f for d in ['velocity', 'speed'])]),
                'volatility': len([f for f in self.feature_names if 'vol' in f])
            }
        }