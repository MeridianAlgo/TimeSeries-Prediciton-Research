"""Market microstructure analyzer for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


class MarketMicrostructureAnalyzer(FeatureEngineer):
    """Analyzes market microstructure for bid-ask spreads, price impact, and market efficiency."""
    
    def __init__(self, vwap_periods: Optional[List[int]] = None):
        """Initialize market microstructure analyzer.
        
        Args:
            vwap_periods: List of periods for VWAP calculations
        """
        self.vwap_periods = vwap_periods or [10, 20, 50]
        self.feature_names = []
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized MarketMicrostructureAnalyzer with VWAP periods: {self.vwap_periods}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with microstructure features added
        """
        try:
            self.logger.info("Generating market microstructure features")
            df = data.copy()
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise FeatureEngineeringError(f"Missing required columns: {missing_cols}")
            
            # Check for empty data
            if len(df) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Generate bid-ask spread proxies
            df = self._generate_spread_proxies(df)
            
            # Generate price impact measures
            df = self._generate_price_impact(df)
            
            # Generate market efficiency indicators
            df = self._generate_market_efficiency(df)
            
            # Generate VWAP analysis
            df = self._generate_vwap_analysis(df)
            
            # Generate volume-price relationships
            df = self._generate_volume_price_relationships(df)
            
            # Generate order flow proxies
            df = self._generate_order_flow_proxies(df)
            
            # Generate liquidity indicators
            df = self._generate_liquidity_indicators(df)
            
            # Clean up any NaN values
            df = df.ffill().bfill().fillna(0)
            
            # Update feature names
            self._update_feature_names(df, data.columns)
            
            self.logger.info(f"Generated {len(self.feature_names)} microstructure features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating microstructure features: {str(e)}")
            raise FeatureEngineeringError(f"Microstructure feature generation failed: {str(e)}") from e
    
    def _generate_spread_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate bid-ask spread proxy indicators."""
        self.logger.debug("Generating spread proxies")
        
        # Basic spread proxy (High-Low range)
        df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Normalized spread by volume
        df['spread_volume_normalized'] = df['spread_proxy'] / np.log(df['Volume'] + 1)
        
        # Spread relative to volatility
        volatility = df['Close'].pct_change().rolling(20).std()
        df['spread_volatility_ratio'] = df['spread_proxy'] / (volatility + 1e-10)
        
        # Intraday spread dynamics
        df['spread_open_close'] = np.abs(df['Open'] - df['Close']) / df['Close']
        df['spread_high_close'] = (df['High'] - df['Close']) / df['Close']
        df['spread_close_low'] = (df['Close'] - df['Low']) / df['Close']
        
        # Spread momentum
        df['spread_momentum'] = df['spread_proxy'].pct_change()
        df['spread_acceleration'] = df['spread_momentum'].diff()
        
        # Spread percentile ranking
        df['spread_percentile'] = df['spread_proxy'].rolling(100).rank(pct=True)
        
        # Spread regime (high/low spread periods)
        spread_ma = df['spread_proxy'].rolling(50).mean()
        df['spread_regime'] = (df['spread_proxy'] > spread_ma).astype(int)
        
        # Spread clustering
        spread_std = df['spread_proxy'].rolling(50).std()
        df['spread_clustering'] = (
            df['spread_proxy'] > spread_ma + spread_std
        ).astype(int)
        
        return df
    
    def _generate_price_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price impact measures."""
        self.logger.debug("Generating price impact measures")
        
        # Basic price impact (price change per unit volume)
        price_change = np.abs(df['Close'] - df['Open'])
        df['price_impact'] = price_change / (df['Volume'] + 1e-10)
        
        # Normalized price impact
        df['price_impact_normalized'] = df['price_impact'] / df['Close']
        
        # Price impact relative to spread
        df['price_impact_spread_ratio'] = (
            df['price_impact_normalized'] / (df['spread_proxy'] + 1e-10)
        )
        
        # Temporary vs permanent impact proxy
        next_open = df['Open'].shift(-1)
        df['temporary_impact'] = np.abs(df['Close'] - next_open) / df['Close']
        df['permanent_impact'] = df['price_impact_normalized'] - df['temporary_impact']
        
        # Volume-weighted price impact
        volume_ma = df['Volume'].rolling(20).mean()
        df['volume_weighted_impact'] = (
            df['price_impact_normalized'] * df['Volume'] / (volume_ma + 1e-10)
        )
        
        # Price impact momentum
        df['price_impact_momentum'] = df['price_impact_normalized'].pct_change()
        
        # Price impact volatility
        df['price_impact_volatility'] = (
            df['price_impact_normalized'].rolling(20).std()
        )
        
        # Price impact efficiency (lower is more efficient)
        df['price_impact_efficiency'] = (
            df['price_impact_normalized'] / (np.abs(df['Close'].pct_change()) + 1e-10)
        )
        
        return df
    
    def _generate_market_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate market efficiency indicators."""
        self.logger.debug("Generating market efficiency indicators")
        
        # Price efficiency (how well price reflects information)
        returns = df['Close'].pct_change()
        
        # Autocorrelation of returns (efficient markets should have low autocorr)
        for lag in [1, 2, 3, 5]:
            df[f'return_autocorr_{lag}'] = (
                returns.rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) >= lag + 10 else 0
                )
            )
        
        # Variance ratio test proxy
        for period in [2, 4, 8]:
            returns_period = df['Close'].pct_change(period)
            variance_1 = returns.rolling(50).var()
            variance_period = returns_period.rolling(50).var()
            df[f'variance_ratio_{period}'] = (
                variance_period / (period * variance_1 + 1e-10)
            )
        
        # Price discovery efficiency
        df['price_discovery_efficiency'] = (
            np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        )
        
        # Information incorporation speed
        df['info_incorporation_speed'] = (
            np.abs(df['Close'] - df['Close'].shift(1)) / 
            (np.abs(df['High'] - df['Low']) + 1e-10)
        )
        
        # Market depth proxy (resistance to price movement)
        df['market_depth_proxy'] = (
            df['Volume'] / (np.abs(df['Close'].pct_change()) + 1e-10)
        )
        
        # Noise-to-signal ratio
        price_noise = df['High'] - df['Low'] - np.abs(df['Close'] - df['Open'])
        price_signal = np.abs(df['Close'] - df['Open'])
        df['noise_signal_ratio'] = price_noise / (price_signal + 1e-10)
        
        # Market synchronization (how well price moves with volume)
        volume_change = df['Volume'].pct_change()
        df['price_volume_sync'] = (
            returns.rolling(20).corr(volume_change)
        )
        
        return df
    
    def _generate_vwap_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP (Volume Weighted Average Price) analysis."""
        self.logger.debug("Generating VWAP analysis")
        
        for period in self.vwap_periods:
            # Calculate VWAP
            price_volume = df['Close'] * df['Volume']
            vwap = price_volume.rolling(period).sum() / df['Volume'].rolling(period).sum()
            df[f'vwap_{period}'] = vwap
            
            # VWAP deviation
            df[f'vwap_deviation_{period}'] = (df['Close'] - vwap) / vwap
            
            # VWAP distance (normalized)
            df[f'vwap_distance_{period}'] = (
                np.abs(df['Close'] - vwap) / df['Close']
            )
            
            # VWAP momentum
            df[f'vwap_momentum_{period}'] = vwap.pct_change()
            
            # VWAP slope
            df[f'vwap_slope_{period}'] = vwap.diff(3)
            
            # Price position relative to VWAP
            df[f'above_vwap_{period}'] = (df['Close'] > vwap).astype(int)
            
            # VWAP support/resistance strength
            vwap_touches = (np.abs(df['Close'] - vwap) / vwap < 0.01).astype(int)
            df[f'vwap_support_strength_{period}'] = (
                vwap_touches.rolling(20).sum()
            )
            
            # VWAP reversion tendency
            above_vwap = df[f'above_vwap_{period}']
            df[f'vwap_reversion_signal_{period}'] = (
                (above_vwap == 1) & (df['Close'].pct_change() < 0)
            ).astype(int) - (
                (above_vwap == 0) & (df['Close'].pct_change() > 0)
            ).astype(int)
            
            # VWAP efficiency (how close price stays to VWAP)
            df[f'vwap_efficiency_{period}'] = (
                1 / (df[f'vwap_distance_{period}'].rolling(20).mean() + 1e-10)
            )
        
        return df
    
    def _generate_volume_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-price relationship indicators."""
        self.logger.debug("Generating volume-price relationships")
        
        # Volume momentum
        df['volume_momentum'] = df['Volume'].pct_change()
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # Volume relative to moving average
        for period in [10, 20, 50]:
            volume_ma = df['Volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / (volume_ma + 1e-10)
            
            # Volume breakout
            volume_std = df['Volume'].rolling(period).std()
            df[f'volume_breakout_{period}'] = (
                df['Volume'] > volume_ma + 2 * volume_std
            ).astype(int)
        
        # Price-volume correlation
        returns = df['Close'].pct_change()
        for window in [10, 20, 50]:
            df[f'price_volume_corr_{window}'] = (
                returns.rolling(window).corr(df['volume_momentum'])
            )
        
        # Volume-weighted returns
        df['volume_weighted_return'] = (
            returns * df['Volume'] / df['Volume'].rolling(20).mean()
        )
        
        # On-Balance Volume (OBV)
        obv_changes = np.where(
            df['Close'] > df['Close'].shift(1), df['Volume'],
            np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)
        )
        df['obv'] = pd.Series(obv_changes).cumsum()
        df['obv_momentum'] = df['obv'].pct_change()
        
        # Volume distribution
        df['volume_percentile'] = df['Volume'].rolling(100).rank(pct=True)
        
        # Volume clustering
        volume_ma = df['Volume'].rolling(50).mean()
        volume_std = df['Volume'].rolling(50).std()
        df['volume_clustering'] = (
            df['Volume'] > volume_ma + volume_std
        ).astype(int)
        
        return df
    
    def _generate_order_flow_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate order flow proxy indicators."""
        self.logger.debug("Generating order flow proxies")
        
        # Buying/selling pressure proxies
        df['buying_pressure'] = (
            (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        )
        df['selling_pressure'] = (
            (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)
        )
        
        # Net buying pressure
        df['net_buying_pressure'] = df['buying_pressure'] - df['selling_pressure']
        
        # Volume-weighted buying pressure
        df['volume_weighted_buying'] = df['buying_pressure'] * df['Volume']
        df['volume_weighted_selling'] = df['selling_pressure'] * df['Volume']
        
        # Cumulative buying/selling pressure
        df['cumulative_buying'] = df['volume_weighted_buying'].rolling(20).sum()
        df['cumulative_selling'] = df['volume_weighted_selling'].rolling(20).sum()
        df['net_cumulative_pressure'] = df['cumulative_buying'] - df['cumulative_selling']
        
        # Order flow momentum
        df['order_flow_momentum'] = df['net_buying_pressure'].diff()
        
        # Order flow persistence
        pressure_sign = np.sign(df['net_buying_pressure'])
        df['order_flow_persistence'] = (
            pressure_sign.groupby((pressure_sign != pressure_sign.shift()).cumsum()).cumcount() + 1
        )
        
        # Order flow intensity
        df['order_flow_intensity'] = (
            np.abs(df['net_buying_pressure']) * df['Volume']
        )
        
        # Order flow efficiency
        price_change = df['Close'].pct_change()
        df['order_flow_efficiency'] = (
            price_change / (df['net_buying_pressure'] + 1e-10)
        )
        
        return df
    
    def _generate_liquidity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate liquidity indicators."""
        self.logger.debug("Generating liquidity indicators")
        
        # Amihud illiquidity measure proxy
        returns = np.abs(df['Close'].pct_change())
        df['amihud_illiquidity'] = returns / (df['Volume'] + 1e-10)
        
        # Roll spread estimator
        price_changes = df['Close'].diff()
        df['roll_spread'] = 2 * np.sqrt(np.abs(
            price_changes.rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) >= 11 else 0
            )
        ))
        
        # Liquidity ratio (volume to volatility)
        volatility = df['Close'].pct_change().rolling(20).std()
        df['liquidity_ratio'] = df['Volume'] / (volatility + 1e-10)
        
        # Market depth indicator
        df['market_depth'] = df['Volume'] / (df['spread_proxy'] + 1e-10)
        
        # Liquidity momentum
        df['liquidity_momentum'] = df['liquidity_ratio'].pct_change()
        
        # Liquidity regime
        liquidity_ma = df['liquidity_ratio'].rolling(50).mean()
        df['liquidity_regime'] = (df['liquidity_ratio'] > liquidity_ma).astype(int)
        
        # Liquidity stress indicator
        liquidity_std = df['liquidity_ratio'].rolling(50).std()
        df['liquidity_stress'] = (
            df['liquidity_ratio'] < liquidity_ma - 2 * liquidity_std
        ).astype(int)
        
        # Turnover rate proxy
        df['turnover_proxy'] = df['Volume'] / df['Close']  # Simplified turnover
        df['turnover_momentum'] = df['turnover_proxy'].pct_change()
        
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
            'vwap_periods': self.vwap_periods,
            'feature_categories': {
                'spreads': len([f for f in self.feature_names if 'spread' in f]),
                'price_impact': len([f for f in self.feature_names if 'impact' in f]),
                'efficiency': len([f for f in self.feature_names if any(x in f for x in ['efficiency', 'autocorr', 'variance_ratio'])]),
                'vwap': len([f for f in self.feature_names if 'vwap' in f]),
                'volume_price': len([f for f in self.feature_names if any(x in f for x in ['volume', 'obv'])]),
                'order_flow': len([f for f in self.feature_names if any(x in f for x in ['buying', 'selling', 'pressure', 'flow'])]),
                'liquidity': len([f for f in self.feature_names if any(x in f for x in ['liquidity', 'amihud', 'roll', 'turnover'])])
            }
        }