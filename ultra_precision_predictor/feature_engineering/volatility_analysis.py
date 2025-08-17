"""Volatility analysis system for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from scipy import stats

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


class VolatilityAnalysisSystem(FeatureEngineer):
    """Advanced volatility analysis system for market prediction."""
    
    def __init__(self, 
                 lookback_periods: Optional[List[int]] = None,
                 volatility_windows: Optional[List[int]] = None):
        """Initialize volatility analysis system.
        
        Args:
            lookback_periods: Periods for volatility calculations
            volatility_windows: Windows for different volatility measures
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]
        self.volatility_windows = volatility_windows or [5, 10, 20, 30]
        self.feature_names = []
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized VolatilityAnalysisSystem")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility analysis features.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with volatility features added
        """
        try:
            self.logger.info("Generating volatility analysis features")
            df = data.copy()
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise FeatureEngineeringError(f"Missing required columns: {missing_cols}")
            
            # Check for empty data
            if len(df) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Generate basic volatility measures
            df = self._generate_basic_volatility(df)
            
            # Generate realized volatility components
            df = self._generate_realized_volatility(df)
            
            # Generate volatility clustering analysis
            df = self._generate_volatility_clustering(df)
            
            # Generate volatility regimes
            df = self._generate_volatility_regimes(df)
            
            # Generate volatility risk measures
            df = self._generate_volatility_risk_measures(df)
            
            # Generate volatility forecasting features
            df = self._generate_volatility_forecasting(df)
            
            # Clean up any NaN values
            df = df.ffill().bfill().fillna(0)
            
            # Update feature names
            self._update_feature_names(df, data.columns)
            
            self.logger.info(f"Generated {len(self.feature_names)} volatility features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating volatility features: {str(e)}")
            raise FeatureEngineeringError(f"Volatility feature generation failed: {str(e)}") from e
    
    def _generate_basic_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic volatility measures."""
        self.logger.debug("Generating basic volatility measures")
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Basic volatility measures
        for window in self.volatility_windows:
            # Standard deviation volatility
            df[f'volatility_std_{window}'] = df['returns'].rolling(window).std()
            df[f'log_volatility_std_{window}'] = df['log_returns'].rolling(window).std()
            
            # Mean absolute deviation
            df[f'volatility_mad_{window}'] = (
                df['returns'].rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
            )
            
            # Volatility of volatility
            df[f'vol_of_vol_{window}'] = df[f'volatility_std_{window}'].rolling(window).std()
            
            # Skewness and kurtosis of returns
            df[f'returns_skewness_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurtosis_{window}'] = df['returns'].rolling(window).kurt()
            
            # Volatility momentum
            df[f'volatility_momentum_{window}'] = df[f'volatility_std_{window}'].pct_change()
            
            # Volatility mean reversion
            vol_ma = df[f'volatility_std_{window}'].rolling(window * 2).mean()
            df[f'volatility_mean_reversion_{window}'] = (
                (df[f'volatility_std_{window}'] - vol_ma) / (vol_ma + 1e-10)
            )
        
        # Parkinson volatility (uses high-low range)
        df['parkinson_volatility'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['High'] / df['Low']) ** 2
        )
        
        # Garman-Klass volatility
        df['garman_klass_volatility'] = np.sqrt(
            0.5 * np.log(df['High'] / df['Low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2
        )
        
        # Rogers-Satchell volatility
        df['rogers_satchell_volatility'] = np.sqrt(
            np.log(df['High'] / df['Close']) * np.log(df['High'] / df['Open']) +
            np.log(df['Low'] / df['Close']) * np.log(df['Low'] / df['Open'])
        )
        
        return df
    
    def _generate_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate realized volatility components."""
        self.logger.debug("Generating realized volatility components")
        
        for window in self.volatility_windows:
            # Realized volatility (sum of squared returns)
            df[f'realized_vol_{window}'] = np.sqrt(
                df['returns'].rolling(window).apply(lambda x: np.sum(x ** 2))
            )
            
            # Bipower variation (robust to jumps)
            abs_returns = np.abs(df['returns'])
            df[f'bipower_variation_{window}'] = (
                (np.pi / 2) * abs_returns.rolling(window).apply(
                    lambda x: np.sum(x[:-1] * x[1:]) if len(x) > 1 else 0
                )
            )
            
            # Jump component (difference between realized vol and bipower)
            df[f'jump_component_{window}'] = np.maximum(
                0, df[f'realized_vol_{window}'] - df[f'bipower_variation_{window}']
            )
            
            # Continuous component
            df[f'continuous_component_{window}'] = (
                df[f'realized_vol_{window}'] - df[f'jump_component_{window}']
            )
            
            # Jump ratio
            df[f'jump_ratio_{window}'] = (
                df[f'jump_component_{window}'] / (df[f'realized_vol_{window}'] + 1e-10)
            )
        
        return df
    
    def _generate_volatility_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility clustering analysis."""
        self.logger.debug("Generating volatility clustering analysis")
        
        # GARCH-like features
        for window in self.volatility_windows:
            vol_col = f'volatility_std_{window}'
            
            if vol_col in df.columns:
                # Volatility persistence (autocorrelation)
                for lag in [1, 2, 3, 5]:
                    df[f'vol_persistence_{window}_{lag}'] = (
                        df[vol_col].rolling(window).apply(
                            lambda x: x.autocorr(lag=lag) if len(x) >= lag + 10 else 0
                        )
                    )
                
                # Volatility clustering strength
                vol_squared = df[vol_col] ** 2
                vol_sq_ma = vol_squared.rolling(window).mean()
                df[f'vol_clustering_{window}'] = (
                    vol_squared / (vol_sq_ma + 1e-10)
                )
                
                # High volatility periods
                vol_threshold = df[vol_col].rolling(window * 2).quantile(0.8)
                high_vol = (df[vol_col] > vol_threshold).astype(int)
                
                # Volatility regime persistence
                df[f'high_vol_persistence_{window}'] = (
                    high_vol.groupby((high_vol != high_vol.shift()).cumsum()).cumcount() + 1
                ) * high_vol
                
                # Volatility shock detection
                vol_ma = df[vol_col].rolling(window).mean()
                vol_std = df[vol_col].rolling(window).std()
                df[f'vol_shock_{window}'] = (
                    df[vol_col] > vol_ma + 2 * vol_std
                ).astype(int)
        
        return df
    
    def _generate_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime analysis."""
        self.logger.debug("Generating volatility regime analysis")
        
        # Use primary volatility measure
        primary_vol = df['volatility_std_20'] if 'volatility_std_20' in df.columns else df['returns'].rolling(20).std()
        
        # Volatility quantiles for regime identification
        vol_20th = primary_vol.rolling(100).quantile(0.2)
        vol_40th = primary_vol.rolling(100).quantile(0.4)
        vol_60th = primary_vol.rolling(100).quantile(0.6)
        vol_80th = primary_vol.rolling(100).quantile(0.8)
        
        # Regime classification
        df['vol_regime_low'] = (primary_vol <= vol_20th).astype(int)
        df['vol_regime_medium_low'] = (
            (primary_vol > vol_20th) & (primary_vol <= vol_40th)
        ).astype(int)
        df['vol_regime_medium'] = (
            (primary_vol > vol_40th) & (primary_vol <= vol_60th)
        ).astype(int)
        df['vol_regime_medium_high'] = (
            (primary_vol > vol_60th) & (primary_vol <= vol_80th)
        ).astype(int)
        df['vol_regime_high'] = (primary_vol > vol_80th).astype(int)
        
        # Regime transitions
        df['vol_regime_transition'] = (
            df['vol_regime_low'].diff().abs() +
            df['vol_regime_medium_low'].diff().abs() +
            df['vol_regime_medium'].diff().abs() +
            df['vol_regime_medium_high'].diff().abs() +
            df['vol_regime_high'].diff().abs()
        )
        
        # Time in current regime
        current_regime = (
            df['vol_regime_low'] * 1 +
            df['vol_regime_medium_low'] * 2 +
            df['vol_regime_medium'] * 3 +
            df['vol_regime_medium_high'] * 4 +
            df['vol_regime_high'] * 5
        )
        
        df['time_in_regime'] = (
            current_regime.groupby((current_regime != current_regime.shift()).cumsum()).cumcount() + 1
        )
        
        return df
    
    def _generate_volatility_risk_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based risk measures."""
        self.logger.debug("Generating volatility risk measures")
        
        returns = df['returns']
        
        # Value at Risk (VaR) estimates
        for confidence in [0.95, 0.99]:
            for window in [20, 50]:
                # Historical VaR
                df[f'var_{int(confidence*100)}_{window}'] = (
                    returns.rolling(window).quantile(1 - confidence)
                )
                
                # Parametric VaR (assuming normal distribution)
                vol = returns.rolling(window).std()
                mean_return = returns.rolling(window).mean()
                z_score = stats.norm.ppf(1 - confidence)
                df[f'parametric_var_{int(confidence*100)}_{window}'] = (
                    mean_return + z_score * vol
                )
        
        # Expected Shortfall (Conditional VaR)
        for window in [20, 50]:
            # Expected shortfall as mean of returns below VaR
            df[f'expected_shortfall_95_{window}'] = (
                returns.rolling(window).apply(
                    lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.quantile(0.05)
                )
            )
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        df['current_drawdown'] = drawdown
        df['max_drawdown_20'] = drawdown.rolling(20).min()
        df['max_drawdown_50'] = drawdown.rolling(50).min()
        
        return df
    
    def _generate_volatility_forecasting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility forecasting features."""
        self.logger.debug("Generating volatility forecasting features")
        
        # GARCH-like volatility prediction features
        returns_squared = df['returns'] ** 2
        
        # Simple GARCH(1,1) approximation
        for window in [20, 50]:
            # Long-term variance
            long_term_var = returns_squared.rolling(window * 2).mean()
            
            # GARCH variance prediction components
            df[f'garch_lt_var_{window}'] = long_term_var
            df[f'garch_lagged_var_{window}'] = returns_squared.shift(1)
            
            # GARCH prediction (simplified)
            alpha, beta, omega = 0.1, 0.8, 0.1  # Typical GARCH parameters
            df[f'garch_forecast_{window}'] = (
                omega * long_term_var +
                alpha * returns_squared.shift(1) +
                beta * df['returns'].rolling(window).std().shift(1) ** 2
            )
        
        # Volatility momentum for forecasting
        for window in self.volatility_windows:
            vol_col = f'volatility_std_{window}'
            if vol_col in df.columns:
                # Volatility trend
                df[f'vol_trend_{window}'] = (
                    df[vol_col].rolling(5).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
                    )
                )
                
                # Volatility cycle position
                vol_detrended = df[vol_col] - df[vol_col].rolling(window).mean()
                df[f'vol_cycle_position_{window}'] = (
                    vol_detrended / (df[vol_col].rolling(window).std() + 1e-10)
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
            'lookback_periods': self.lookback_periods,
            'volatility_windows': self.volatility_windows,
            'feature_categories': {
                'basic_volatility': len([f for f in self.feature_names if any(x in f for x in ['volatility_std', 'parkinson', 'garman', 'rogers'])]),
                'realized_volatility': len([f for f in self.feature_names if any(x in f for x in ['realized_vol', 'bipower', 'jump', 'continuous'])]),
                'clustering': len([f for f in self.feature_names if any(x in f for x in ['clustering', 'persistence', 'shock'])]),
                'regimes': len([f for f in self.feature_names if any(x in f for x in ['regime', 'transition'])]),
                'risk_measures': len([f for f in self.feature_names if any(x in f for x in ['var_', 'shortfall', 'drawdown'])]),
                'forecasting': len([f for f in self.feature_names if any(x in f for x in ['garch', 'forecast', 'trend', 'cycle'])])
            }
        }