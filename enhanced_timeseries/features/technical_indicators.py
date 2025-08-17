"""
Comprehensive technical indicator calculations for time series analysis.
Implements 25+ technical indicators including momentum, trend, volatility, and volume indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
from scipy.signal import argrelextrema
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.
    """
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Exponential Moving Average."""
        if alpha is None:
            alpha = 2.0 / (window + 1)
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def wma(data: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            data: Price series
            window: RSI period
            
        Returns:
            RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence.
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            data: Price series
            window: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': (upper_band - lower_band) / sma,
            'percent_b': (data - lower_band) / (upper_band - lower_band)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_window: %K period
            d_window: %D period
            
        Returns:
            Dictionary with %K and %D values
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Lookback period
            
        Returns:
            Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ATR period
            
        Returns:
            ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Commodity Channel Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: CCI period
            
        Returns:
            CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """
        Average Directional Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ADX period
            
        Returns:
            Dictionary with ADX, +DI, and -DI
        """
        # Calculate True Range
        atr_values = TechnicalIndicators.atr(high, low, close, window)
        
        # Calculate Directional Movement
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=low.index)
        
        # Smooth the directional movements
        plus_dm_smooth = plus_dm.rolling(window=window).mean()
        minus_dm_smooth = minus_dm.rolling(window=window).mean()
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr_values)
        minus_di = 100 * (minus_dm_smooth / atr_values)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume data
            
        Returns:
            OBV values
        """
        price_change = close.diff()
        
        obv_values = []
        obv = 0
        
        for i in range(len(close)):
            if i == 0:
                obv_values.append(0)
            elif price_change.iloc[i] > 0:
                obv += volume.iloc[i]
                obv_values.append(obv)
            elif price_change.iloc[i] < 0:
                obv -= volume.iloc[i]
                obv_values.append(obv)
            else:
                obv_values.append(obv)
        
        return pd.Series(obv_values, index=close.index)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            
        Returns:
            VWAP values
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """
        Money Flow Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            window: MFI period
            
        Returns:
            MFI values (0-100)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        price_change = typical_price.diff()
        
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, 
                     af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """
        Parabolic SAR.
        
        Args:
            high: High prices
            low: Low prices
            af_start: Starting acceleration factor
            af_increment: AF increment
            af_max: Maximum AF
            
        Returns:
            Parabolic SAR values
        """
        sar = np.zeros(len(high))
        trend = np.zeros(len(high))
        af = np.zeros(len(high))
        ep = np.zeros(len(high))
        
        # Initialize
        sar[0] = low.iloc[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        af[0] = af_start
        ep[0] = high.iloc[0]
        
        for i in range(1, len(high)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low.iloc[i] <= sar[i]:
                    # Trend reversal
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = low.iloc[i]
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high.iloc[i] >= sar[i]:
                    # Trend reversal
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = high.iloc[i]
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(sar, index=high.index)
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                tenkan_period: int = 9, kijun_period: int = 26, 
                senkou_b_period: int = 52) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            tenkan_period: Tenkan-sen period
            kijun_period: Kijun-sen period
            senkou_b_period: Senkou Span B period
            
        Returns:
            Dictionary with Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                     low.rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun_period).max() + 
                    low.rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou_b_period).max() + 
                         low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """
        Aroon Indicator.
        
        Args:
            high: High prices
            low: Low prices
            window: Aroon period
            
        Returns:
            Dictionary with Aroon Up, Aroon Down, and Aroon Oscillator
        """
        aroon_up = []
        aroon_down = []
        
        for i in range(len(high)):
            if i < window - 1:
                aroon_up.append(np.nan)
                aroon_down.append(np.nan)
            else:
                high_window = high.iloc[i-window+1:i+1]
                low_window = low.iloc[i-window+1:i+1]
                
                high_idx = high_window.idxmax()
                low_idx = low_window.idxmin()
                
                periods_since_high = i - high_window.index.get_loc(high_idx)
                periods_since_low = i - low_window.index.get_loc(low_idx)
                
                aroon_up.append(((window - periods_since_high) / window) * 100)
                aroon_down.append(((window - periods_since_low) / window) * 100)
        
        aroon_up = pd.Series(aroon_up, index=high.index)
        aroon_down = pd.Series(aroon_down, index=low.index)
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        window: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Keltner Channels.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: EMA period
            multiplier: ATR multiplier
            
        Returns:
            Dictionary with upper, middle, and lower channels
        """
        middle = TechnicalIndicators.ema(close, window)
        atr = TechnicalIndicators.atr(high, low, close, window)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """
        Donchian Channels.
        
        Args:
            high: High prices
            low: Low prices
            window: Lookback period
            
        Returns:
            Dictionary with upper, middle, and lower channels
        """
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()
        middle = (upper + lower) / 2
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }


class MomentumIndicators:
    """Momentum-based technical indicators."""
    
    @staticmethod
    def momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """Price momentum."""
        return data / data.shift(window) - 1
    
    @staticmethod
    def rate_of_change(data: pd.Series, window: int = 10) -> pd.Series:
        """Rate of Change."""
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    @staticmethod
    def trix(data: pd.Series, window: int = 14) -> pd.Series:
        """TRIX indicator."""
        ema1 = TechnicalIndicators.ema(data, window)
        ema2 = TechnicalIndicators.ema(ema1, window)
        ema3 = TechnicalIndicators.ema(ema2, window)
        
        trix = ema3.pct_change() * 10000
        return trix
    
    @staticmethod
    def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                          period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Ultimate Oscillator."""
        prev_close = close.shift(1)
        
        # True Low and Buying Pressure
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        buying_pressure = close - true_low
        
        # True Range
        true_range = TechnicalIndicators.atr(high, low, close, 1) * len(close)  # Unnormalized TR
        
        # Calculate averages for each period
        bp_sum1 = buying_pressure.rolling(window=period1).sum()
        tr_sum1 = true_range.rolling(window=period1).sum()
        
        bp_sum2 = buying_pressure.rolling(window=period2).sum()
        tr_sum2 = true_range.rolling(window=period2).sum()
        
        bp_sum3 = buying_pressure.rolling(window=period3).sum()
        tr_sum3 = true_range.rolling(window=period3).sum()
        
        # Ultimate Oscillator calculation
        uo = 100 * ((4 * (bp_sum1 / tr_sum1)) + (2 * (bp_sum2 / tr_sum2)) + (bp_sum3 / tr_sum3)) / 7
        
        return uo


class VolatilityIndicators:
    """Volatility-based technical indicators."""
    
    @staticmethod
    def historical_volatility(data: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
        """Historical volatility."""
        returns = data.pct_change()
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            volatility = volatility * np.sqrt(252)  # Assuming daily data
        
        return volatility
    
    @staticmethod
    def garch_volatility(data: pd.Series, window: int = 20) -> pd.Series:
        """GARCH-like volatility estimate."""
        returns = data.pct_change()
        
        # Simple GARCH(1,1) approximation
        alpha = 0.1
        beta = 0.85
        
        volatility = []
        long_run_var = returns.var()
        
        for i in range(len(returns)):
            if i == 0:
                vol = np.sqrt(long_run_var)
            else:
                prev_return = returns.iloc[i-1]
                prev_vol = volatility[i-1]
                
                variance = (1 - alpha - beta) * long_run_var + alpha * (prev_return ** 2) + beta * (prev_vol ** 2)
                vol = np.sqrt(variance)
            
            volatility.append(vol)
        
        return pd.Series(volatility, index=data.index)
    
    @staticmethod
    def volatility_clustering(data: pd.Series, window: int = 20) -> pd.Series:
        """Volatility clustering indicator."""
        returns = data.pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Volatility of volatility
        vol_of_vol = volatility.rolling(window=window).std()
        
        return vol_of_vol


class VolumeIndicators:
    """Volume-based technical indicators."""
    
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series, 
                                close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line."""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        
        ad_line = money_flow_volume.cumsum()
        return ad_line
    
    @staticmethod
    def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, fast: int = 3, slow: int = 10) -> pd.Series:
        """Chaikin Oscillator."""
        ad_line = VolumeIndicators.accumulation_distribution(high, low, close, volume)
        
        fast_ema = TechnicalIndicators.ema(ad_line, fast)
        slow_ema = TechnicalIndicators.ema(ad_line, slow)
        
        chaikin_osc = fast_ema - slow_ema
        return chaikin_osc
    
    @staticmethod
    def volume_rate_of_change(volume: pd.Series, window: int = 14) -> pd.Series:
        """Volume Rate of Change."""
        return ((volume - volume.shift(window)) / volume.shift(window)) * 100
    
    @staticmethod
    def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, 
                        window: int = 14) -> pd.Series:
        """Ease of Movement."""
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        box_height = volume / (high - low)
        
        emv = distance_moved / box_height
        emv_ma = emv.rolling(window=window).mean()
        
        return emv_ma


class MultiTimeframeIndicators:
    """Multi-timeframe technical indicators."""
    
    @staticmethod
    def multi_timeframe_rsi(data: pd.Series, windows: List[int] = [14, 21, 28]) -> Dict[str, pd.Series]:
        """RSI across multiple timeframes."""
        rsi_dict = {}
        
        for window in windows:
            rsi_dict[f'rsi_{window}'] = TechnicalIndicators.rsi(data, window)
        
        return rsi_dict
    
    @staticmethod
    def multi_timeframe_sma(data: pd.Series, windows: List[int] = [10, 20, 50, 200]) -> Dict[str, pd.Series]:
        """SMA across multiple timeframes."""
        sma_dict = {}
        
        for window in windows:
            sma_dict[f'sma_{window}'] = TechnicalIndicators.sma(data, window)
        
        return sma_dict
    
    @staticmethod
    def sma_crossover_signals(data: pd.Series, fast_window: int = 10, slow_window: int = 20) -> pd.Series:
        """SMA crossover signals."""
        fast_sma = TechnicalIndicators.sma(data, fast_window)
        slow_sma = TechnicalIndicators.sma(data, slow_window)
        
        # 1 for bullish crossover, -1 for bearish crossover, 0 for no signal
        signals = pd.Series(0, index=data.index)
        
        crossover = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        crossunder = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        
        signals[crossover] = 1
        signals[crossunder] = -1
        
        return signals


class TechnicalIndicatorEngine:
    """
    Main engine for calculating comprehensive technical indicators.
    """
    
    def __init__(self):
        """Initialize the technical indicator engine."""
        self.indicators = TechnicalIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()
        self.multi_timeframe = MultiTimeframeIndicators()
    
    def calculate_all_indicators(self, data: pd.DataFrame, 
                               price_columns: Dict[str, str] = None,
                               volume_column: str = 'volume') -> pd.DataFrame:
        """
        Calculate all available technical indicators.
        
        Args:
            data: OHLCV data
            price_columns: Column name mapping {'high': 'High', 'low': 'Low', 'close': 'Close', 'open': 'Open'}
            volume_column: Volume column name
            
        Returns:
            DataFrame with all calculated indicators
        """
        if price_columns is None:
            price_columns = {'high': 'high', 'low': 'low', 'close': 'close', 'open': 'open'}
        
        result_df = data.copy()
        
        # Extract price series
        high = data[price_columns['high']]
        low = data[price_columns['low']]
        close = data[price_columns['close']]
        open_price = data[price_columns['open']]
        
        # Volume (if available)
        volume = data[volume_column] if volume_column in data.columns else None
        
        try:
            # Basic indicators
            result_df['sma_10'] = self.indicators.sma(close, 10)
            result_df['sma_20'] = self.indicators.sma(close, 20)
            result_df['sma_50'] = self.indicators.sma(close, 50)
            result_df['ema_12'] = self.indicators.ema(close, 12)
            result_df['ema_26'] = self.indicators.ema(close, 26)
            
            # Momentum indicators
            result_df['rsi_14'] = self.indicators.rsi(close, 14)
            result_df['momentum_10'] = self.momentum.momentum(close, 10)
            result_df['roc_10'] = self.momentum.rate_of_change(close, 10)
            
            # MACD
            macd_data = self.indicators.macd(close)
            result_df['macd'] = macd_data['macd']
            result_df['macd_signal'] = macd_data['signal']
            result_df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.indicators.bollinger_bands(close)
            result_df['bb_upper'] = bb_data['upper']
            result_df['bb_middle'] = bb_data['middle']
            result_df['bb_lower'] = bb_data['lower']
            result_df['bb_bandwidth'] = bb_data['bandwidth']
            result_df['bb_percent_b'] = bb_data['percent_b']
            
            # Stochastic
            stoch_data = self.indicators.stochastic(high, low, close)
            result_df['stoch_k'] = stoch_data['k_percent']
            result_df['stoch_d'] = stoch_data['d_percent']
            
            # Other oscillators
            result_df['williams_r'] = self.indicators.williams_r(high, low, close)
            result_df['cci'] = self.indicators.cci(high, low, close)
            
            # Volatility indicators
            result_df['atr'] = self.indicators.atr(high, low, close)
            result_df['historical_vol'] = self.volatility.historical_volatility(close)
            result_df['garch_vol'] = self.volatility.garch_volatility(close)
            
            # ADX
            adx_data = self.indicators.adx(high, low, close)
            result_df['adx'] = adx_data['adx']
            result_df['plus_di'] = adx_data['plus_di']
            result_df['minus_di'] = adx_data['minus_di']
            
            # Aroon
            aroon_data = self.indicators.aroon(high, low)
            result_df['aroon_up'] = aroon_data['aroon_up']
            result_df['aroon_down'] = aroon_data['aroon_down']
            result_df['aroon_oscillator'] = aroon_data['aroon_oscillator']
            
            # Parabolic SAR
            result_df['parabolic_sar'] = self.indicators.parabolic_sar(high, low)
            
            # Ichimoku
            ichimoku_data = self.indicators.ichimoku(high, low, close)
            result_df['tenkan_sen'] = ichimoku_data['tenkan_sen']
            result_df['kijun_sen'] = ichimoku_data['kijun_sen']
            result_df['senkou_span_a'] = ichimoku_data['senkou_span_a']
            result_df['senkou_span_b'] = ichimoku_data['senkou_span_b']
            
            # Keltner Channels
            keltner_data = self.indicators.keltner_channels(high, low, close)
            result_df['keltner_upper'] = keltner_data['upper']
            result_df['keltner_middle'] = keltner_data['middle']
            result_df['keltner_lower'] = keltner_data['lower']
            
            # Donchian Channels
            donchian_data = self.indicators.donchian_channels(high, low)
            result_df['donchian_upper'] = donchian_data['upper']
            result_df['donchian_middle'] = donchian_data['middle']
            result_df['donchian_lower'] = donchian_data['lower']
            
            # Volume indicators (if volume data available)
            if volume is not None:
                result_df['obv'] = self.indicators.obv(close, volume)
                result_df['vwap'] = self.indicators.vwap(high, low, close, volume)
                result_df['mfi'] = self.indicators.mfi(high, low, close, volume)
                result_df['ad_line'] = self.volume.accumulation_distribution(high, low, close, volume)
                result_df['chaikin_osc'] = self.volume.chaikin_oscillator(high, low, close, volume)
                result_df['volume_roc'] = self.volume.volume_rate_of_change(volume)
                result_df['ease_of_movement'] = self.volume.ease_of_movement(high, low, volume)
            
            # Multi-timeframe RSI
            multi_rsi = self.multi_timeframe.multi_timeframe_rsi(close)
            for key, value in multi_rsi.items():
                result_df[key] = value
            
            # SMA crossover signals
            result_df['sma_crossover_10_20'] = self.multi_timeframe.sma_crossover_signals(close, 10, 20)
            result_df['sma_crossover_20_50'] = self.multi_timeframe.sma_crossover_signals(close, 20, 50)
            
            logger.info(f"Calculated {len(result_df.columns) - len(data.columns)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
        
        return result_df
    
    def get_indicator_list(self) -> List[str]:
        """Get list of all available indicators."""
        return [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'momentum_10', 'roc_10',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent_b',
            'stoch_k', 'stoch_d', 'williams_r', 'cci',
            'atr', 'historical_vol', 'garch_vol',
            'adx', 'plus_di', 'minus_di',
            'aroon_up', 'aroon_down', 'aroon_oscillator',
            'parabolic_sar',
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
            'keltner_upper', 'keltner_middle', 'keltner_lower',
            'donchian_upper', 'donchian_middle', 'donchian_lower',
            'obv', 'vwap', 'mfi', 'ad_line', 'chaikin_osc', 'volume_roc', 'ease_of_movement',
            'rsi_14', 'rsi_21', 'rsi_28',
            'sma_crossover_10_20', 'sma_crossover_20_50'
        ]