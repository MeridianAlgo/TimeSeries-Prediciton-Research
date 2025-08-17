"""
Unit tests for comprehensive technical indicators.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from enhanced_timeseries.features.technical_indicators import (
    TechnicalIndicators, MomentumIndicators, VolatilityIndicators,
    VolumeIndicators, MultiTimeframeIndicators, TechnicalIndicatorEngine
)


class TestTechnicalIndicators(unittest.TestCase):
    """Test TechnicalIndicators class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        self.data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        self.data.set_index('date', inplace=True)
        
        # Extract series for testing
        self.close = self.data['close']
        self.high = self.data['high']
        self.low = self.data['low']
        self.volume = self.data['volume']
    
    def test_sma(self):
        """Test Simple Moving Average."""
        sma_10 = TechnicalIndicators.sma(self.close, 10)
        
        self.assertEqual(len(sma_10), len(self.close))
        self.assertTrue(pd.isna(sma_10.iloc[:9]).all())  # First 9 values should be NaN
        self.assertFalse(pd.isna(sma_10.iloc[9:]).any())  # Rest should not be NaN
        
        # Test manual calculation for a specific point
        manual_sma = self.close.iloc[0:10].mean()
        self.assertAlmostEqual(sma_10.iloc[9], manual_sma, places=6)
    
    def test_ema(self):
        """Test Exponential Moving Average."""
        ema_10 = TechnicalIndicators.ema(self.close, 10)
        
        self.assertEqual(len(ema_10), len(self.close))
        self.assertFalse(pd.isna(ema_10).any())  # EMA should not have NaN values
        
        # EMA should be different from SMA
        sma_10 = TechnicalIndicators.sma(self.close, 10)
        self.assertFalse(ema_10.iloc[20:].equals(sma_10.iloc[20:]))
    
    def test_rsi(self):
        """Test Relative Strength Index."""
        rsi = TechnicalIndicators.rsi(self.close, 14)
        
        self.assertEqual(len(rsi), len(self.close))
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_macd(self):
        """Test MACD indicator."""
        macd_data = TechnicalIndicators.macd(self.close)
        
        self.assertIn('macd', macd_data)
        self.assertIn('signal', macd_data)
        self.assertIn('histogram', macd_data)
        
        # All series should have same length
        self.assertEqual(len(macd_data['macd']), len(self.close))
        self.assertEqual(len(macd_data['signal']), len(self.close))
        self.assertEqual(len(macd_data['histogram']), len(self.close))
        
        # Histogram should equal MACD - Signal
        histogram_calc = macd_data['macd'] - macd_data['signal']
        pd.testing.assert_series_equal(
            macd_data['histogram'].dropna(), 
            histogram_calc.dropna(), 
            check_names=False
        )
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands."""
        bb_data = TechnicalIndicators.bollinger_bands(self.close, 20, 2.0)
        
        self.assertIn('upper', bb_data)
        self.assertIn('middle', bb_data)
        self.assertIn('lower', bb_data)
        self.assertIn('bandwidth', bb_data)
        self.assertIn('percent_b', bb_data)
        
        # Upper band should be above middle, middle above lower
        valid_data = bb_data['upper'].dropna()
        valid_indices = valid_data.index
        
        self.assertTrue((bb_data['upper'][valid_indices] >= bb_data['middle'][valid_indices]).all())
        self.assertTrue((bb_data['middle'][valid_indices] >= bb_data['lower'][valid_indices]).all())
    
    def test_stochastic(self):
        """Test Stochastic Oscillator."""
        stoch_data = TechnicalIndicators.stochastic(self.high, self.low, self.close)
        
        self.assertIn('k_percent', stoch_data)
        self.assertIn('d_percent', stoch_data)
        
        # Stochastic values should be between 0 and 100
        k_valid = stoch_data['k_percent'].dropna()
        d_valid = stoch_data['d_percent'].dropna()
        
        self.assertTrue((k_valid >= 0).all())
        self.assertTrue((k_valid <= 100).all())
        self.assertTrue((d_valid >= 0).all())
        self.assertTrue((d_valid <= 100).all())
    
    def test_williams_r(self):
        """Test Williams %R."""
        williams_r = TechnicalIndicators.williams_r(self.high, self.low, self.close)
        
        # Williams %R should be between -100 and 0
        valid_williams = williams_r.dropna()
        self.assertTrue((valid_williams >= -100).all())
        self.assertTrue((valid_williams <= 0).all())
    
    def test_atr(self):
        """Test Average True Range."""
        atr = TechnicalIndicators.atr(self.high, self.low, self.close)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())
    
    def test_cci(self):
        """Test Commodity Channel Index."""
        cci = TechnicalIndicators.cci(self.high, self.low, self.close)
        
        self.assertEqual(len(cci), len(self.close))
        # CCI can have any value, so just check it's not all NaN
        self.assertFalse(cci.dropna().empty)
    
    def test_adx(self):
        """Test Average Directional Index."""
        adx_data = TechnicalIndicators.adx(self.high, self.low, self.close)
        
        self.assertIn('adx', adx_data)
        self.assertIn('plus_di', adx_data)
        self.assertIn('minus_di', adx_data)
        
        # ADX and DI values should be positive
        adx_valid = adx_data['adx'].dropna()
        plus_di_valid = adx_data['plus_di'].dropna()
        minus_di_valid = adx_data['minus_di'].dropna()
        
        self.assertTrue((adx_valid >= 0).all())
        self.assertTrue((plus_di_valid >= 0).all())
        self.assertTrue((minus_di_valid >= 0).all())
    
    def test_obv(self):
        """Test On-Balance Volume."""
        obv = TechnicalIndicators.obv(self.close, self.volume)
        
        self.assertEqual(len(obv), len(self.close))
        # OBV should start at 0
        self.assertEqual(obv.iloc[0], 0)
    
    def test_vwap(self):
        """Test Volume Weighted Average Price."""
        vwap = TechnicalIndicators.vwap(self.high, self.low, self.close, self.volume)
        
        self.assertEqual(len(vwap), len(self.close))
        # VWAP should be positive
        self.assertTrue((vwap > 0).all())
    
    def test_mfi(self):
        """Test Money Flow Index."""
        mfi = TechnicalIndicators.mfi(self.high, self.low, self.close, self.volume)
        
        # MFI should be between 0 and 100
        valid_mfi = mfi.dropna()
        self.assertTrue((valid_mfi >= 0).all())
        self.assertTrue((valid_mfi <= 100).all())
    
    def test_parabolic_sar(self):
        """Test Parabolic SAR."""
        sar = TechnicalIndicators.parabolic_sar(self.high, self.low)
        
        self.assertEqual(len(sar), len(self.close))
        # SAR should be positive (assuming positive prices)
        self.assertTrue((sar > 0).all())
    
    def test_ichimoku(self):
        """Test Ichimoku Cloud."""
        ichimoku_data = TechnicalIndicators.ichimoku(self.high, self.low, self.close)
        
        expected_keys = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        for key in expected_keys:
            self.assertIn(key, ichimoku_data)
            self.assertEqual(len(ichimoku_data[key]), len(self.close))
    
    def test_aroon(self):
        """Test Aroon Indicator."""
        aroon_data = TechnicalIndicators.aroon(self.high, self.low)
        
        self.assertIn('aroon_up', aroon_data)
        self.assertIn('aroon_down', aroon_data)
        self.assertIn('aroon_oscillator', aroon_data)
        
        # Aroon values should be between 0 and 100
        aroon_up_valid = aroon_data['aroon_up'].dropna()
        aroon_down_valid = aroon_data['aroon_down'].dropna()
        
        self.assertTrue((aroon_up_valid >= 0).all())
        self.assertTrue((aroon_up_valid <= 100).all())
        self.assertTrue((aroon_down_valid >= 0).all())
        self.assertTrue((aroon_down_valid <= 100).all())
    
    def test_keltner_channels(self):
        """Test Keltner Channels."""
        keltner_data = TechnicalIndicators.keltner_channels(self.high, self.low, self.close)
        
        self.assertIn('upper', keltner_data)
        self.assertIn('middle', keltner_data)
        self.assertIn('lower', keltner_data)
        
        # Upper should be above middle, middle above lower
        valid_indices = keltner_data['upper'].dropna().index
        
        self.assertTrue((keltner_data['upper'][valid_indices] >= keltner_data['middle'][valid_indices]).all())
        self.assertTrue((keltner_data['middle'][valid_indices] >= keltner_data['lower'][valid_indices]).all())
    
    def test_donchian_channels(self):
        """Test Donchian Channels."""
        donchian_data = TechnicalIndicators.donchian_channels(self.high, self.low)
        
        self.assertIn('upper', donchian_data)
        self.assertIn('middle', donchian_data)
        self.assertIn('lower', donchian_data)
        
        # Upper should be above middle, middle above lower
        valid_indices = donchian_data['upper'].dropna().index
        
        self.assertTrue((donchian_data['upper'][valid_indices] >= donchian_data['middle'][valid_indices]).all())
        self.assertTrue((donchian_data['middle'][valid_indices] >= donchian_data['lower'][valid_indices]).all())


class TestMomentumIndicators(unittest.TestCase):
    """Test MomentumIndicators class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        
        self.data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98
        }, index=dates)
        
        self.close = self.data['close']
        self.high = self.data['high']
        self.low = self.data['low']
    
    def test_momentum(self):
        """Test momentum indicator."""
        momentum = MomentumIndicators.momentum(self.close, 10)
        
        self.assertEqual(len(momentum), len(self.close))
        # First 10 values should be NaN
        self.assertTrue(pd.isna(momentum.iloc[:10]).all())
    
    def test_rate_of_change(self):
        """Test Rate of Change."""
        roc = MomentumIndicators.rate_of_change(self.close, 10)
        
        self.assertEqual(len(roc), len(self.close))
        # ROC is percentage, so should be reasonable values
        valid_roc = roc.dropna()
        self.assertTrue((valid_roc > -100).all())  # Not more than 100% loss
    
    def test_trix(self):
        """Test TRIX indicator."""
        trix = MomentumIndicators.trix(self.close, 14)
        
        self.assertEqual(len(trix), len(self.close))
        # TRIX should have some valid values
        self.assertFalse(trix.dropna().empty)
    
    def test_ultimate_oscillator(self):
        """Test Ultimate Oscillator."""
        uo = MomentumIndicators.ultimate_oscillator(self.high, self.low, self.close)
        
        # UO should be between 0 and 100
        valid_uo = uo.dropna()
        if not valid_uo.empty:
            self.assertTrue((valid_uo >= 0).all())
            self.assertTrue((valid_uo <= 100).all())


class TestVolatilityIndicators(unittest.TestCase):
    """Test VolatilityIndicators class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        self.data = pd.DataFrame({'close': prices}, index=dates)
        self.close = self.data['close']
    
    def test_historical_volatility(self):
        """Test historical volatility."""
        vol = VolatilityIndicators.historical_volatility(self.close, 20)
        
        self.assertEqual(len(vol), len(self.close))
        # Volatility should be positive
        valid_vol = vol.dropna()
        self.assertTrue((valid_vol >= 0).all())
    
    def test_garch_volatility(self):
        """Test GARCH-like volatility."""
        garch_vol = VolatilityIndicators.garch_volatility(self.close, 20)
        
        self.assertEqual(len(garch_vol), len(self.close))
        # GARCH volatility should be positive
        self.assertTrue((garch_vol > 0).all())
    
    def test_volatility_clustering(self):
        """Test volatility clustering."""
        vol_clustering = VolatilityIndicators.volatility_clustering(self.close, 20)
        
        self.assertEqual(len(vol_clustering), len(self.close))
        # Should have some valid values
        self.assertFalse(vol_clustering.dropna().empty)


class TestVolumeIndicators(unittest.TestCase):
    """Test VolumeIndicators class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        
        self.data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        self.close = self.data['close']
        self.high = self.data['high']
        self.low = self.data['low']
        self.volume = self.data['volume']
    
    def test_accumulation_distribution(self):
        """Test Accumulation/Distribution Line."""
        ad_line = VolumeIndicators.accumulation_distribution(
            self.high, self.low, self.close, self.volume
        )
        
        self.assertEqual(len(ad_line), len(self.close))
        # A/D line is cumulative, so should be monotonic in some sense
        self.assertFalse(ad_line.dropna().empty)
    
    def test_chaikin_oscillator(self):
        """Test Chaikin Oscillator."""
        chaikin_osc = VolumeIndicators.chaikin_oscillator(
            self.high, self.low, self.close, self.volume
        )
        
        self.assertEqual(len(chaikin_osc), len(self.close))
        # Should have some valid values
        self.assertFalse(chaikin_osc.dropna().empty)
    
    def test_volume_rate_of_change(self):
        """Test Volume Rate of Change."""
        volume_roc = VolumeIndicators.volume_rate_of_change(self.volume, 14)
        
        self.assertEqual(len(volume_roc), len(self.volume))
        # Should have some valid values
        self.assertFalse(volume_roc.dropna().empty)
    
    def test_ease_of_movement(self):
        """Test Ease of Movement."""
        eom = VolumeIndicators.ease_of_movement(self.high, self.low, self.volume)
        
        self.assertEqual(len(eom), len(self.close))
        # Should have some valid values
        self.assertFalse(eom.dropna().empty)


class TestMultiTimeframeIndicators(unittest.TestCase):
    """Test MultiTimeframeIndicators class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        self.data = pd.DataFrame({'close': prices}, index=dates)
        self.close = self.data['close']
    
    def test_multi_timeframe_rsi(self):
        """Test multi-timeframe RSI."""
        multi_rsi = MultiTimeframeIndicators.multi_timeframe_rsi(self.close)
        
        expected_keys = ['rsi_14', 'rsi_21', 'rsi_28']
        for key in expected_keys:
            self.assertIn(key, multi_rsi)
            self.assertEqual(len(multi_rsi[key]), len(self.close))
    
    def test_multi_timeframe_sma(self):
        """Test multi-timeframe SMA."""
        multi_sma = MultiTimeframeIndicators.multi_timeframe_sma(self.close)
        
        expected_keys = ['sma_10', 'sma_20', 'sma_50', 'sma_200']
        for key in expected_keys:
            self.assertIn(key, multi_sma)
            self.assertEqual(len(multi_sma[key]), len(self.close))
    
    def test_sma_crossover_signals(self):
        """Test SMA crossover signals."""
        signals = MultiTimeframeIndicators.sma_crossover_signals(self.close, 10, 20)
        
        self.assertEqual(len(signals), len(self.close))
        # Signals should be -1, 0, or 1
        unique_signals = signals.dropna().unique()
        self.assertTrue(all(signal in [-1, 0, 1] for signal in unique_signals))


class TestTechnicalIndicatorEngine(unittest.TestCase):
    """Test TechnicalIndicatorEngine class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create comprehensive OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.data = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        self.engine = TechnicalIndicatorEngine()
    
    def test_engine_creation(self):
        """Test engine creation."""
        self.assertIsNotNone(self.engine.indicators)
        self.assertIsNotNone(self.engine.momentum)
        self.assertIsNotNone(self.engine.volatility)
        self.assertIsNotNone(self.engine.volume)
        self.assertIsNotNone(self.engine.multi_timeframe)
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators."""
        result = self.engine.calculate_all_indicators(self.data)
        
        # Should have more columns than original data
        self.assertGreater(len(result.columns), len(self.data.columns))
        
        # Should have same number of rows
        self.assertEqual(len(result), len(self.data))
        
        # Check some key indicators exist
        expected_indicators = [
            'sma_10', 'sma_20', 'ema_12', 'rsi_14', 'macd', 'bb_upper', 
            'stoch_k', 'atr', 'adx', 'obv', 'vwap'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
    
    def test_calculate_indicators_without_volume(self):
        """Test calculating indicators without volume data."""
        data_no_volume = self.data.drop('volume', axis=1)
        
        result = self.engine.calculate_all_indicators(data_no_volume)
        
        # Should still work without volume
        self.assertGreater(len(result.columns), len(data_no_volume.columns))
        
        # Volume-based indicators should not be present
        volume_indicators = ['obv', 'vwap', 'mfi', 'ad_line', 'chaikin_osc']
        for indicator in volume_indicators:
            self.assertNotIn(indicator, result.columns)
    
    def test_custom_column_mapping(self):
        """Test with custom column mapping."""
        # Rename columns to test custom mapping
        custom_data = self.data.rename(columns={
            'high': 'High',
            'low': 'Low', 
            'close': 'Close',
            'open': 'Open'
        })
        
        custom_mapping = {
            'high': 'High',
            'low': 'Low',
            'close': 'Close', 
            'open': 'Open'
        }
        
        result = self.engine.calculate_all_indicators(
            custom_data, 
            price_columns=custom_mapping
        )
        
        # Should work with custom column names
        self.assertGreater(len(result.columns), len(custom_data.columns))
        self.assertIn('sma_10', result.columns)
    
    def test_get_indicator_list(self):
        """Test getting indicator list."""
        indicator_list = self.engine.get_indicator_list()
        
        self.assertIsInstance(indicator_list, list)
        self.assertGreater(len(indicator_list), 20)  # Should have many indicators
        
        # Check some expected indicators
        expected_indicators = ['sma_10', 'rsi_14', 'macd', 'bb_upper', 'atr']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicator_list)
    
    def test_indicator_values_reasonable(self):
        """Test that calculated indicators have reasonable values."""
        result = self.engine.calculate_all_indicators(self.data)
        
        # RSI should be between 0 and 100
        rsi_valid = result['rsi_14'].dropna()
        if not rsi_valid.empty:
            self.assertTrue((rsi_valid >= 0).all())
            self.assertTrue((rsi_valid <= 100).all())
        
        # ATR should be positive
        atr_valid = result['atr'].dropna()
        if not atr_valid.empty:
            self.assertTrue((atr_valid > 0).all())
        
        # Bollinger Bands should be ordered correctly
        bb_valid_idx = result['bb_upper'].dropna().index
        if not bb_valid_idx.empty:
            self.assertTrue((result.loc[bb_valid_idx, 'bb_upper'] >= 
                           result.loc[bb_valid_idx, 'bb_middle']).all())
            self.assertTrue((result.loc[bb_valid_idx, 'bb_middle'] >= 
                           result.loc[bb_valid_idx, 'bb_lower']).all())


if __name__ == '__main__':
    unittest.main()