"""
Market microstructure feature extraction for enhanced time series prediction.
Implements advanced microstructure indicators including bid-ask spread proxies,
order flow analysis, and liquidity measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MicrostructureFeatures:
    """Market microstructure feature extraction."""
    
    def __init__(self):
        self.feature_cache = {}
        
    def calculate_all_microstructure_features(self, data: pd.DataFrame, 
                                            lookback_periods: List[int] = [5, 10, 20, 50]) -> Dict[str, np.ndarray]:
        """Calculate all microstructure features."""
        features = {}
        
        # Spread and liquidity features
        features.update(self._spread_features(data, lookback_periods))
        
        # Order flow features
        features.update(self._order_flow_features(data, lookback_periods))
        
        # Price impact features
        features.update(self._price_impact_features(data, lookback_periods))
        
        # Volatility microstructure features
        features.update(self._volatility_microstructure_features(data, lookback_periods))
        
        # Market depth proxies
        features.update(self._market_depth_features(data, lookback_periods))
        
        # Intraday patterns
        features.update(self._intraday_pattern_features(data, lookback_periods))
        
        return features
    
    def _spread_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate bid-ask spread proxy features."""
        features = {}
        
        # High-Low spread as bid-ask proxy
        hl_spread = (data['High'] - data['Low']) / data['Close']
        features['hl_spread'] = hl_spread.fillna(0).values
        
        # Relative spread
        features['relative_spread'] = hl_spread.values
        
        # Rolling spread statistics
        for period in periods:
            spread_mean = hl_spread.rolling(period).mean().fillna(0)
            spread_std = hl_spread.rolling(period).std().fillna(0)
            
            features[f'spread_mean_{period}'] = spread_mean.values
            features[f'spread_std_{period}'] = spread_std.values
            features[f'spread_cv_{period}'] = (spread_std / spread_mean).fillna(0).values
            
            # Spread percentiles
            features[f'spread_percentile_{period}'] = (
                hl_spread.rolling(period).rank(pct=True).fillna(0.5).values
            )
        
        # Effective spread (using close-to-close volatility)
        returns = data['Close'].pct_change().fillna(0)
        effective_spread = 2 * np.abs(returns)
        features['effective_spread'] = effective_spread.values
        
        # Quoted spread proxy (using open-close difference)
        quoted_spread = np.abs(data['Open'] - data['Close']) / data['Close']
        features['quoted_spread'] = quoted_spread.fillna(0).values
        
        # Roll's spread estimator
        features['roll_spread'] = self._calculate_roll_spread(data)
        
        return features
    
    def _order_flow_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate order flow features."""
        features = {}
        
        # Price-volume relationship
        returns = data['Close'].pct_change().fillna(0)
        volume_normalized = data['Volume'] / data['Volume'].rolling(20).mean().fillna(1)
        
        # Volume-weighted returns
        features['volume_weighted_returns'] = (returns * volume_normalized).values
        
        # Order flow imbalance proxy
        # Positive when price moves up with high volume, negative when down with high volume
        price_direction = np.sign(returns)
        volume_intensity = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std()
        volume_intensity = volume_intensity.fillna(0)
        
        features['order_flow_imbalance'] = (price_direction * volume_intensity).values
        
        # Tick rule (Lee-Ready algorithm proxy)
        # Classify trades as buyer or seller initiated
        mid_price = (data['High'] + data['Low']) / 2
        trade_direction = np.where(data['Close'] > mid_price, 1, 
                                 np.where(data['Close'] < mid_price, -1, 0))
        features['trade_direction'] = trade_direction.astype(float)
        
        # Signed volume
        signed_volume = trade_direction * data['Volume']
        features['signed_volume'] = signed_volume.astype(float)
        
        # Rolling order flow statistics
        for period in periods:
            # Net order flow
            net_flow = pd.Series(signed_volume).rolling(period).sum().fillna(0)
            features[f'net_order_flow_{period}'] = net_flow.values
            
            # Order flow ratio
            total_volume = data['Volume'].rolling(period).sum()
            flow_ratio = net_flow / total_volume
            features[f'order_flow_ratio_{period}'] = flow_ratio.fillna(0).values
            
            # Buy/sell pressure
            buy_volume = pd.Series(np.maximum(0, signed_volume)).rolling(period).sum()
            sell_volume = pd.Series(np.maximum(0, -signed_volume)).rolling(period).sum()
            
            features[f'buy_pressure_{period}'] = (buy_volume / total_volume).fillna(0).values
            features[f'sell_pressure_{period}'] = (sell_volume / total_volume).fillna(0).values
            
            # Order flow momentum
            flow_momentum = net_flow.diff().fillna(0)
            features[f'order_flow_momentum_{period}'] = flow_momentum.values
        
        # Volume at price levels
        features.update(self._volume_at_price_features(data))
        
        return features
    
    def _price_impact_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate price impact features."""
        features = {}
        
        # Temporary price impact (intraday reversal)
        intraday_return = (data['Close'] - data['Open']) / data['Open']
        overnight_return = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        features['intraday_return'] = intraday_return.fillna(0).values
        features['overnight_return'] = overnight_return.fillna(0).values
        
        # Price impact per unit volume
        returns = data['Close'].pct_change().fillna(0)
        volume_normalized = data['Volume'] / data['Volume'].rolling(20).mean().fillna(1)
        
        price_impact = returns / np.sqrt(volume_normalized)
        features['price_impact_per_volume'] = price_impact.fillna(0).values
        
        # Permanent vs temporary impact
        for period in [1, 5, 10]:
            future_return = data['Close'].shift(-period).pct_change(period).fillna(0)
            current_return = returns
            
            # Correlation between current and future returns (permanent impact)
            rolling_corr = current_return.rolling(50).corr(future_return).fillna(0)
            features[f'permanent_impact_{period}'] = rolling_corr.values
            
            # Reversal measure (temporary impact)
            reversal = -current_return.rolling(period).sum() * future_return
            features[f'temporary_impact_{period}'] = reversal.fillna(0).values
        
        # Kyle's lambda (price impact coefficient)
        features['kyle_lambda'] = self._calculate_kyle_lambda(data)
        
        # Amihud illiquidity measure
        features['amihud_illiquidity'] = self._calculate_amihud_illiquidity(data, periods)
        
        return features
    
    def _volatility_microstructure_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate volatility-based microstructure features."""
        features = {}
        
        # Realized volatility components
        returns = data['Close'].pct_change().fillna(0)
        
        # Garman-Klass volatility estimator
        gk_vol = self._calculate_garman_klass_volatility(data)
        features['garman_klass_volatility'] = gk_vol
        
        # Rogers-Satchell volatility estimator
        rs_vol = self._calculate_rogers_satchell_volatility(data)
        features['rogers_satchell_volatility'] = rs_vol
        
        # Yang-Zhang volatility estimator
        yz_vol = self._calculate_yang_zhang_volatility(data)
        features['yang_zhang_volatility'] = yz_vol
        
        # Microstructure noise measures
        for period in periods:
            # Bid-ask bounce (negative autocorrelation in returns)
            return_autocorr = returns.rolling(period).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            ).fillna(0)
            features[f'return_autocorr_{period}'] = return_autocorr.values
            
            # Variance ratio test statistic
            var_ratio = self._calculate_variance_ratio(returns, period)
            features[f'variance_ratio_{period}'] = var_ratio
        
        # Noise-to-signal ratio
        features['noise_to_signal'] = self._calculate_noise_to_signal_ratio(data)
        
        return features
    
    def _market_depth_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate market depth proxy features."""
        features = {}
        
        # Volume-weighted price measures
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        features['vwap_deviation'] = ((data['Close'] - vwap) / vwap).fillna(0).values
        
        # Price clustering (round number effects)
        features['price_clustering'] = self._calculate_price_clustering(data)
        
        # Market depth proxies
        for period in periods:
            # Average trade size proxy
            avg_trade_size = data['Volume'].rolling(period).mean()
            features[f'avg_trade_size_{period}'] = avg_trade_size.fillna(0).values
            
            # Trade intensity
            trade_count_proxy = data['Volume'] / avg_trade_size
            features[f'trade_intensity_{period}'] = trade_count_proxy.fillna(0).values
            
            # Depth imbalance (using volume and price range)
            price_range = data['High'] - data['Low']
            depth_proxy = data['Volume'] / price_range
            features[f'market_depth_{period}'] = depth_proxy.rolling(period).mean().fillna(0).values
        
        # Resilience measures
        features.update(self._calculate_resilience_measures(data, periods))
        
        return features
    
    def _intraday_pattern_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate intraday pattern features."""
        features = {}
        
        # Opening and closing effects
        open_close_ratio = (data['Close'] - data['Open']) / data['Open']
        features['open_close_ratio'] = open_close_ratio.fillna(0).values
        
        # High-low position
        hl_position = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        features['hl_position'] = hl_position.fillna(0.5).values
        
        # Opening gap
        opening_gap = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        features['opening_gap'] = opening_gap.fillna(0).values
        
        # Intraday momentum
        intraday_momentum = (data['High'] - data['Low']) / data['Open']
        features['intraday_momentum'] = intraday_momentum.fillna(0).values
        
        # Volume patterns
        for period in periods:
            # Volume trend within period
            volume_trend = data['Volume'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            ).fillna(0)
            features[f'volume_trend_{period}'] = volume_trend.values
            
            # Volume acceleration
            volume_accel = data['Volume'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 2)[0] if len(x) > 2 else 0
            ).fillna(0)
            features[f'volume_acceleration_{period}'] = volume_accel.values
        
        return features
    
    def _volume_at_price_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volume at price level features."""
        features = {}
        
        # Volume-weighted average price deviation
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Volume concentration at different price levels
        price_levels = np.array([data['Open'], data['High'], data['Low'], data['Close']])
        
        # Volume at high/low vs close
        volume_at_high = np.where(data['Close'] == data['High'], data['Volume'], 0)
        volume_at_low = np.where(data['Close'] == data['Low'], data['Volume'], 0)
        
        features['volume_at_high_ratio'] = (volume_at_high / data['Volume']).astype(float)
        features['volume_at_low_ratio'] = (volume_at_low / data['Volume']).astype(float)
        
        # Price-volume distribution
        pv_correlation = data['Close'].rolling(20).corr(data['Volume']).fillna(0)
        features['price_volume_correlation'] = pv_correlation.values
        
        return features
    
    # Helper methods for complex calculations
    
    def _calculate_roll_spread(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Roll's spread estimator."""
        returns = data['Close'].pct_change().fillna(0)
        
        # Roll's estimator: 2 * sqrt(-Cov(r_t, r_{t-1}))
        roll_spread = np.zeros(len(data))
        
        for i in range(50, len(data)):
            window_returns = returns.iloc[i-50:i]
            if len(window_returns) > 1:
                covariance = np.cov(window_returns[:-1], window_returns[1:])[0, 1]
                if covariance < 0:
                    roll_spread[i] = 2 * np.sqrt(-covariance)
                else:
                    roll_spread[i] = 0
        
        return roll_spread
    
    def _calculate_kyle_lambda(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Kyle's lambda (price impact coefficient)."""
        returns = data['Close'].pct_change().fillna(0)
        volume = data['Volume']
        
        kyle_lambda = np.zeros(len(data))
        
        for i in range(50, len(data)):
            window_returns = returns.iloc[i-50:i]
            window_volume = volume.iloc[i-50:i]
            
            if len(window_returns) > 1 and window_volume.std() > 0:
                # Regression of |returns| on volume
                abs_returns = np.abs(window_returns)
                coeff = np.cov(abs_returns, window_volume)[0, 1] / np.var(window_volume)
                kyle_lambda[i] = coeff
        
        return kyle_lambda
    
    def _calculate_amihud_illiquidity(self, data: pd.DataFrame, periods: List[int]) -> np.ndarray:
        """Calculate Amihud illiquidity measure."""
        returns = data['Close'].pct_change().fillna(0)
        dollar_volume = data['Close'] * data['Volume']
        
        # Amihud measure: |return| / dollar_volume
        illiquidity = np.abs(returns) / dollar_volume
        illiquidity = illiquidity.replace([np.inf, -np.inf], 0).fillna(0)
        
        return illiquidity.values
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Garman-Klass volatility estimator."""
        log_hl = np.log(data['High'] / data['Low'])
        log_co = np.log(data['Close'] / data['Open'])
        
        gk_vol = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return gk_vol.fillna(0).values
    
    def _calculate_rogers_satchell_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Rogers-Satchell volatility estimator."""
        log_ho = np.log(data['High'] / data['Open'])
        log_hc = np.log(data['High'] / data['Close'])
        log_lo = np.log(data['Low'] / data['Open'])
        log_lc = np.log(data['Low'] / data['Close'])
        
        rs_vol = log_ho * log_hc + log_lo * log_lc
        return rs_vol.fillna(0).values
    
    def _calculate_yang_zhang_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Yang-Zhang volatility estimator."""
        log_co = np.log(data['Close'] / data['Open'])
        log_oc = np.log(data['Open'] / data['Close'].shift(1))
        
        # Rogers-Satchell component
        rs_vol = self._calculate_rogers_satchell_volatility(data)
        
        # Overnight component
        k = 0.34 / (1.34 + (len(data) + 1) / (len(data) - 1))
        
        yz_vol = log_oc**2 + k * log_co**2 + (1 - k) * rs_vol
        return yz_vol.fillna(0).values
    
    def _calculate_variance_ratio(self, returns: pd.Series, period: int) -> np.ndarray:
        """Calculate variance ratio test statistic."""
        var_ratio = np.zeros(len(returns))
        
        for i in range(period * 2, len(returns)):
            window_returns = returns.iloc[i-period*2:i]
            
            if len(window_returns) >= period * 2:
                # Variance of period-returns
                period_returns = window_returns.rolling(period).sum().dropna()
                var_period = np.var(period_returns)
                
                # Variance of 1-period returns
                var_1 = np.var(window_returns)
                
                if var_1 > 0:
                    var_ratio[i] = var_period / (period * var_1)
        
        return var_ratio
    
    def _calculate_noise_to_signal_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate noise-to-signal ratio."""
        returns = data['Close'].pct_change().fillna(0)
        
        # Signal: trend component (using moving average)
        signal = data['Close'].rolling(20).mean().pct_change().fillna(0)
        
        # Noise: deviation from trend
        noise = returns - signal
        
        # Noise-to-signal ratio
        signal_var = signal.rolling(50).var().fillna(1)
        noise_var = noise.rolling(50).var().fillna(0)
        
        ns_ratio = noise_var / signal_var
        return ns_ratio.fillna(0).values
    
    def _calculate_price_clustering(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate price clustering measure."""
        # Measure tendency of prices to cluster at round numbers
        prices = data['Close']
        
        # Check for clustering at different decimal places
        clustering = np.zeros(len(data))
        
        for i in range(len(data)):
            price = prices.iloc[i]
            
            # Check clustering at different levels
            if price % 1 == 0:  # Whole numbers
                clustering[i] = 1.0
            elif price % 0.5 == 0:  # Half numbers
                clustering[i] = 0.5
            elif price % 0.25 == 0:  # Quarter numbers
                clustering[i] = 0.25
            else:
                clustering[i] = 0.0
        
        return clustering
    
    def _calculate_resilience_measures(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate market resilience measures."""
        features = {}
        
        returns = data['Close'].pct_change().fillna(0)
        
        for period in periods:
            # Price reversal after large moves
            large_moves = np.abs(returns) > returns.rolling(period).std() * 2
            
            reversal_strength = np.zeros(len(data))
            for i in range(1, len(data)):
                if large_moves.iloc[i-1]:
                    # Measure reversal in next period
                    if returns.iloc[i-1] * returns.iloc[i] < 0:  # Opposite direction
                        reversal_strength[i] = np.abs(returns.iloc[i]) / np.abs(returns.iloc[i-1])
            
            features[f'price_resilience_{period}'] = reversal_strength
            
            # Volume resilience (volume after large price moves)
            volume_after_moves = np.where(large_moves.shift(1), data['Volume'], 0)
            avg_volume = data['Volume'].rolling(period).mean()
            volume_resilience = volume_after_moves / avg_volume
            
            features[f'volume_resilience_{period}'] = volume_resilience.fillna(0).values
        
        return features