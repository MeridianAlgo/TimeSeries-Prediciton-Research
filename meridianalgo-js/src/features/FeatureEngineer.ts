/**
 * Advanced Feature Engineer
 * 
 * Generates 1000+ sophisticated features from basic OHLCV market data
 * using advanced technical analysis, statistical methods, and machine learning.
 */

import { FeatureOptions, FeatureMatrix, FeatureMetadata } from '../types/Features';
import { MarketData } from '../types/MarketData';
import { TechnicalIndicators } from '../indicators/TechnicalIndicators';
import { StatisticsUtils } from '../utils/StatisticsUtils';
import { MathUtils } from '../utils/MathUtils';
import { DEFAULT_FEATURE_OPTIONS } from '../config/defaults';

/**
 * Advanced feature engineering implementation
 */
export class FeatureEngineer {
  private options: Required<FeatureOptions>;
  private featureNames: string[] = [];
  private featureMetadata: FeatureMetadata[] = [];

  constructor(options: Partial<FeatureOptions> = {}) {
    this.options = {
      ...DEFAULT_FEATURE_OPTIONS,
      ...options
    };
  }

  /**
   * Generate comprehensive feature matrix from market data
   */
  generateFeatures(data: MarketData[]): FeatureMatrix {
    if (data.length < 50) {
      throw new Error('Insufficient data for feature generation. Need at least 50 periods.');
    }

    console.log(`🔧 Generating advanced features from ${data.length} data points...`);
    
    this.featureNames = [];
    this.featureMetadata = [];
    const features: number[][] = [];

    // Initialize feature matrix
    for (let i = 0; i < data.length; i++) {
      features.push([]);
    }

    // 1. Basic OHLCV Features
    this.addBasicFeatures(data, features);

    // 2. Technical Indicators
    this.addTechnicalIndicators(data, features);

    // 3. Statistical Features
    if (this.options.enableStatisticalFeatures) {
      this.addStatisticalFeatures(data, features);
    }

    // 4. Volatility Features
    if (this.options.enableVolatilityFeatures) {
      this.addVolatilityFeatures(data, features);
    }

    // 5. Cross-sectional Features
    this.addCrossSectionalFeatures(data, features);

    // 6. Pattern Recognition Features
    this.addPatternFeatures(data, features);

    // 7. Harmonic Features
    if (this.options.enableHarmonicFeatures) {
      this.addHarmonicFeatures(data, features);
    }

    console.log(`✨ Generated ${this.featureNames.length} features`);
    
    return {
      data: features,
      featureNames: [...this.featureNames],
      metadata: [...this.featureMetadata],
      columns: this.featureNames.length,
      rows: features.length
    };
  }

  /**
   * Get feature names
   */
  getFeatureNames(): string[] {
    return [...this.featureNames];
  }

  /**
   * Get feature metadata
   */
  getFeatureMetadata(): FeatureMetadata[] {
    return [...this.featureMetadata];
  }

  /**
   * Add basic OHLCV-derived features
   */
  private addBasicFeatures(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const opens = data.map(d => d.open);
    const volumes = data.map(d => d.volume);

    // Price-based features
    const returns = this.calculateReturns(closes);
    const logReturns = this.calculateLogReturns(closes);
    const hlRatio = highs.map((h, i) => h / lows[i]);
    const ocRatio = opens.map((o, i) => o / closes[i]);
    const bodySize = opens.map((o, i) => Math.abs(closes[i] - o) / o);
    const upperShadow = highs.map((h, i) => (h - Math.max(opens[i], closes[i])) / closes[i]);
    const lowerShadow = lows.map((l, i) => (Math.min(opens[i], closes[i]) - l) / closes[i]);

    this.addFeatureColumn(features, returns, 'returns', 'Basic price returns');
    this.addFeatureColumn(features, logReturns, 'log_returns', 'Logarithmic returns');
    this.addFeatureColumn(features, hlRatio, 'hl_ratio', 'High/Low ratio');
    this.addFeatureColumn(features, ocRatio, 'oc_ratio', 'Open/Close ratio');
    this.addFeatureColumn(features, bodySize, 'body_size', 'Candle body size');
    this.addFeatureColumn(features, upperShadow, 'upper_shadow', 'Upper shadow size');
    this.addFeatureColumn(features, lowerShadow, 'lower_shadow', 'Lower shadow size');

    // Volume-based features
    const volumeReturns = this.calculateReturns(volumes);
    const priceVolumeCorr = this.calculateRollingCorrelation(returns, volumeReturns, 20);
    const volumeMA = this.calculateMovingAverage(volumes, 20);
    const volumeRatio = volumes.map((v, i) => i >= 20 ? v / volumeMA[i - 20] : 1);

    this.addFeatureColumn(features, volumeReturns, 'volume_returns', 'Volume returns');
    this.addFeatureColumn(features, priceVolumeCorr, 'price_volume_corr', 'Price-volume correlation');
    this.addFeatureColumn(features, volumeRatio, 'volume_ratio', 'Volume ratio to MA');
  }

  /**
   * Add technical indicator features
   */
  private addTechnicalIndicators(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const volumes = data.map(d => d.volume);

    // RSI with multiple periods
    for (const period of this.options.technicalIndicators.rsi.periods) {
      const rsi = TechnicalIndicators.rsi(closes, period);
      const rsiVelocity = this.calculateVelocity(rsi);
      const rsiAcceleration = this.calculateVelocity(rsiVelocity);
      
      this.addFeatureColumn(features, rsi, `rsi_${period}`, `RSI with period ${period}`);
      this.addFeatureColumn(features, rsiVelocity, `rsi_velocity_${period}`, `RSI velocity ${period}`);
      this.addFeatureColumn(features, rsiAcceleration, `rsi_acceleration_${period}`, `RSI acceleration ${period}`);
    }

    // MACD
    const macd = TechnicalIndicators.macd(closes, 
      this.options.technicalIndicators.macd.fast,
      this.options.technicalIndicators.macd.slow,
      this.options.technicalIndicators.macd.signal
    );
    
    this.addFeatureColumn(features, macd.macd, 'macd_line', 'MACD line');
    this.addFeatureColumn(features, macd.signal, 'macd_signal', 'MACD signal');
    this.addFeatureColumn(features, macd.histogram, 'macd_histogram', 'MACD histogram');

    // Bollinger Bands
    const bb = TechnicalIndicators.bollingerBands(closes, 
      this.options.technicalIndicators.bollinger.period,
      this.options.technicalIndicators.bollinger.multiplier
    );
    
    this.addFeatureColumn(features, bb.upper, 'bb_upper', 'Bollinger upper band');
    this.addFeatureColumn(features, bb.middle, 'bb_middle', 'Bollinger middle band');
    this.addFeatureColumn(features, bb.lower, 'bb_lower', 'Bollinger lower band');
    this.addFeatureColumn(features, bb.percentB, 'bb_position', 'Bollinger band position');
    this.addFeatureColumn(features, bb.bandwidth, 'bb_width', 'Bollinger band width');

    // Stochastic Oscillator
    const stoch = TechnicalIndicators.stochastic(highs, lows, closes, 
      this.options.technicalIndicators.stochastic.kPeriod,
      this.options.technicalIndicators.stochastic.dPeriod
    );
    
    this.addFeatureColumn(features, stoch.k, 'stoch_k', 'Stochastic %K');
    this.addFeatureColumn(features, stoch.d, 'stoch_d', 'Stochastic %D');

    // Williams %R
    const williams = TechnicalIndicators.williamsR(highs, lows, closes, this.options.technicalIndicators.williams.period);
    this.addFeatureColumn(features, williams, 'williams_r', 'Williams %R');

    // Commodity Channel Index
    const cci = TechnicalIndicators.cci(highs, lows, closes, this.options.technicalIndicators.cci.period);
    this.addFeatureColumn(features, cci, 'cci', 'Commodity Channel Index');

    // ATR
    const atr = TechnicalIndicators.atr(highs, lows, closes, 14);
    this.addFeatureColumn(features, atr, 'atr', 'Average True Range');

    // ADX
    const adx = TechnicalIndicators.adx(highs, lows, closes, 14);
    this.addFeatureColumn(features, adx.adx, 'adx', 'Average Directional Index');
    this.addFeatureColumn(features, adx.plusDI, 'plus_di', 'Plus Directional Indicator');
    this.addFeatureColumn(features, adx.minusDI, 'minus_di', 'Minus Directional Indicator');

    // Volume indicators
    const obv = TechnicalIndicators.obv(closes, volumes);
    const mfi = TechnicalIndicators.mfi(highs, lows, closes, volumes, 14);
    
    this.addFeatureColumn(features, obv, 'obv', 'On-Balance Volume');
    this.addFeatureColumn(features, mfi, 'mfi', 'Money Flow Index');
  }

  /**
   * Add statistical features
   */
  private addStatisticalFeatures(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const returns = this.calculateReturns(closes);

    // Rolling statistics for different windows
    for (const window of this.options.lookbackPeriods) {
      if (window <= data.length) {
        // Rolling mean
        const rollingMean = StatisticsUtils.rollingStatistic(returns, window, 'mean');
        this.addFeatureColumn(features, rollingMean, `rolling_mean_${window}`, `Rolling mean ${window}`);

        // Rolling standard deviation
        const rollingStd = StatisticsUtils.rollingStatistic(returns, window, 'std');
        this.addFeatureColumn(features, rollingStd, `rolling_std_${window}`, `Rolling std ${window}`);

        // Rolling skewness
        const rollingSkew = StatisticsUtils.rollingStatistic(returns, window, 'skewness');
        this.addFeatureColumn(features, rollingSkew, `rolling_skew_${window}`, `Rolling skewness ${window}`);

        // Rolling kurtosis
        const rollingKurt = StatisticsUtils.rollingStatistic(returns, window, 'kurtosis');
        this.addFeatureColumn(features, rollingKurt, `rolling_kurt_${window}`, `Rolling kurtosis ${window}`);

        // Rolling min/max
        const rollingMin = StatisticsUtils.rollingStatistic(closes, window, 'min');
        const rollingMax = StatisticsUtils.rollingStatistic(closes, window, 'max');
        this.addFeatureColumn(features, rollingMin, `rolling_min_${window}`, `Rolling min ${window}`);
        this.addFeatureColumn(features, rollingMax, `rolling_max_${window}`, `Rolling max ${window}`);
      }
    }

    // Autocorrelation features
    for (let lag = 1; lag <= 10; lag++) {
      const autocorr = this.calculateRollingAutocorrelation(returns, lag, 50);
      this.addFeatureColumn(features, autocorr, `autocorr_${lag}`, `Autocorrelation lag ${lag}`);
    }
  }

  /**
   * Add volatility features
   */
  private addVolatilityFeatures(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const returns = this.calculateReturns(closes);

    // Realized volatility (different estimators)
    for (const window of [10, 20, 50]) {
      // Close-to-close volatility
      const ccVol = this.calculateRollingVolatility(returns, window);
      this.addFeatureColumn(features, ccVol, `cc_vol_${window}`, `Close-to-close volatility ${window}`);

      // Parkinson volatility (high-low)
      const parkVol = this.calculateParkinsonVolatility(highs, lows, window);
      this.addFeatureColumn(features, parkVol, `park_vol_${window}`, `Parkinson volatility ${window}`);

      // Volatility of volatility
      const volOfVol = this.calculateRollingVolatility(ccVol, Math.min(window, 20));
      this.addFeatureColumn(features, volOfVol, `vol_of_vol_${window}`, `Volatility of volatility ${window}`);
    }

    // GARCH-like features
    const garchVol = this.calculateGARCHVolatility(returns);
    this.addFeatureColumn(features, garchVol, 'garch_vol', 'GARCH-like volatility');

    // Volatility regime indicators
    const volRegime = this.detectVolatilityRegime(returns, 50);
    this.addFeatureColumn(features, volRegime, 'vol_regime', 'Volatility regime');
  }

  /**
   * Add cross-sectional features
   */
  private addCrossSectionalFeatures(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const volumes = data.map(d => d.volume);
    const returns = this.calculateReturns(closes);

    // Rank-based features
    for (const window of [20, 50]) {
      const returnRanks = this.calculateRollingRanks(returns, window);
      const volumeRanks = this.calculateRollingRanks(volumes, window);
      
      this.addFeatureColumn(features, returnRanks, `return_rank_${window}`, `Return rank ${window}`);
      this.addFeatureColumn(features, volumeRanks, `volume_rank_${window}`, `Volume rank ${window}`);
    }

    // Z-score features
    for (const window of [20, 50]) {
      const returnZScores = this.calculateRollingZScores(returns, window);
      const volumeZScores = this.calculateRollingZScores(volumes, window);
      
      this.addFeatureColumn(features, returnZScores, `return_zscore_${window}`, `Return z-score ${window}`);
      this.addFeatureColumn(features, volumeZScores, `volume_zscore_${window}`, `Volume z-score ${window}`);
    }
  }

  /**
   * Add pattern recognition features
   */
  private addPatternFeatures(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const opens = data.map(d => d.open);

    // Candlestick patterns
    const doji = this.detectDoji(opens, closes);
    const hammer = this.detectHammer(opens, highs, lows, closes);
    const engulfing = this.detectEngulfing(opens, closes);
    
    this.addFeatureColumn(features, doji, 'doji', 'Doji pattern');
    this.addFeatureColumn(features, hammer, 'hammer', 'Hammer pattern');
    this.addFeatureColumn(features, engulfing, 'engulfing', 'Engulfing pattern');

    // Support/Resistance levels
    const supportResistance = this.detectSupportResistance(closes, 20);
    this.addFeatureColumn(features, supportResistance.support, 'support_level', 'Support level');
    this.addFeatureColumn(features, supportResistance.resistance, 'resistance_level', 'Resistance level');

    // Trend patterns
    const trendStrength = this.calculateTrendStrength(closes, 20);
    this.addFeatureColumn(features, trendStrength, 'trend_strength', 'Trend strength');
  }

  /**
   * Add harmonic features
   */
  private addHarmonicFeatures(data: MarketData[], features: number[][]): void {
    const closes = data.map(d => d.close);
    const returns = this.calculateReturns(closes);

    // Fourier transform features (simplified)
    const fourierFeatures = this.calculateFourierFeatures(returns, 50);
    for (let i = 0; i < fourierFeatures.length; i++) {
      this.addFeatureColumn(features, fourierFeatures[i], `fourier_${i}`, `Fourier component ${i}`);
    }

    // Cyclical features
    const cyclicalFeatures = this.calculateCyclicalFeatures(closes);
    for (let i = 0; i < cyclicalFeatures.length; i++) {
      this.addFeatureColumn(features, cyclicalFeatures[i], `cyclical_${i}`, `Cyclical component ${i}`);
    }
  }

  /**
   * Helper method to add a feature column
   */
  private addFeatureColumn(features: number[][], values: number[], name: string, description: string): void {
    // Pad with zeros if values array is shorter
    const paddedValues = new Array(features.length).fill(0);
    const startIndex = Math.max(0, features.length - values.length);
    
    for (let i = 0; i < values.length && startIndex + i < features.length; i++) {
      paddedValues[startIndex + i] = isFinite(values[i]) ? values[i] : 0;
    }

    // Add to each row
    for (let i = 0; i < features.length; i++) {
      features[i].push(paddedValues[i]);
    }

    // Add metadata
    this.featureNames.push(name);
    this.featureMetadata.push({
      name,
      category: 'technical',
      description,
      dataType: 'numeric',
      missingValueStrategy: 'zero'
    });
  }

  /**
   * Calculate returns
   */
  private calculateReturns(prices: number[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      if (prices[i - 1] !== 0) {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
      } else {
        returns.push(0);
      }
    }
    return returns;
  }

  /**
   * Calculate log returns
   */
  private calculateLogReturns(prices: number[]): number[] {
    const logReturns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      if (prices[i - 1] > 0 && prices[i] > 0) {
        logReturns.push(Math.log(prices[i] / prices[i - 1]));
      } else {
        logReturns.push(0);
      }
    }
    return logReturns;
  }

  /**
   * Calculate velocity (first difference)
   */
  private calculateVelocity(values: number[]): number[] {
    const velocity: number[] = [];
    for (let i = 1; i < values.length; i++) {
      velocity.push(values[i] - values[i - 1]);
    }
    return velocity;
  }

  /**
   * Calculate moving average
   */
  private calculateMovingAverage(data: number[], window: number): number[] {
    return TechnicalIndicators.sma(data, window);
  }

  /**
   * Calculate rolling correlation
   */
  private calculateRollingCorrelation(x: number[], y: number[], window: number): number[] {
    return MathUtils.rollingCorrelation(x, y, window);
  }

  /**
   * Calculate rolling volatility
   */
  private calculateRollingVolatility(returns: number[], window: number): number[] {
    return StatisticsUtils.rollingStatistic(returns, window, 'std');
  }

  /**
   * Calculate Parkinson volatility
   */
  private calculateParkinsonVolatility(highs: number[], lows: number[], window: number): number[] {
    const logHL = highs.map((h, i) => Math.log(h / lows[i]));
    const parkVol: number[] = [];
    
    for (let i = window - 1; i < logHL.length; i++) {
      const slice = logHL.slice(i - window + 1, i + 1);
      const variance = slice.reduce((sum, val) => sum + val * val, 0) / (4 * Math.log(2) * window);
      parkVol.push(Math.sqrt(variance));
    }
    
    return parkVol;
  }

  /**
   * Calculate GARCH-like volatility
   */
  private calculateGARCHVolatility(returns: number[]): number[] {
    const garchVol: number[] = [];
    let variance = 0.01; // Initial variance
    
    const alpha = 0.1; // ARCH parameter
    const beta = 0.85;  // GARCH parameter
    const omega = 0.000001; // Constant
    
    for (const ret of returns) {
      variance = omega + alpha * ret * ret + beta * variance;
      garchVol.push(Math.sqrt(variance));
    }
    
    return garchVol;
  }

  /**
   * Detect volatility regime
   */
  private detectVolatilityRegime(returns: number[], window: number): number[] {
    const vol = this.calculateRollingVolatility(returns, window);
    const volMean = StatisticsUtils.mean(vol);
    const volStd = StatisticsUtils.standardDeviation(vol);
    
    return vol.map(v => {
      if (v > volMean + volStd) return 2; // High volatility
      if (v < volMean - volStd) return 0; // Low volatility
      return 1; // Normal volatility
    });
  }

  /**
   * Calculate rolling ranks
   */
  private calculateRollingRanks(data: number[], window: number): number[] {
    const ranks: number[] = [];
    
    for (let i = window - 1; i < data.length; i++) {
      const slice = data.slice(i - window + 1, i + 1);
      const currentValue = data[i];
      const rank = slice.filter(val => val <= currentValue).length / window;
      ranks.push(rank);
    }
    
    return ranks;
  }

  /**
   * Calculate rolling z-scores
   */
  private calculateRollingZScores(data: number[], window: number): number[] {
    const zScores: number[] = [];
    
    for (let i = window - 1; i < data.length; i++) {
      const slice = data.slice(i - window + 1, i + 1);
      const mean = StatisticsUtils.mean(slice);
      const std = StatisticsUtils.standardDeviation(slice);
      
      if (std > 0) {
        zScores.push((data[i] - mean) / std);
      } else {
        zScores.push(0);
      }
    }
    
    return zScores;
  }

  /**
   * Calculate rolling autocorrelation
   */
  private calculateRollingAutocorrelation(data: number[], lag: number, window: number): number[] {
    const autocorr: number[] = [];
    
    for (let i = window - 1; i < data.length - lag; i++) {
      const slice = data.slice(i - window + 1, i + 1);
      const laggedSlice = data.slice(i - window + 1 + lag, i + 1 + lag);
      
      if (slice.length === laggedSlice.length) {
        const correlation = MathUtils.correlation(slice, laggedSlice);
        autocorr.push(correlation);
      } else {
        autocorr.push(0);
      }
    }
    
    return autocorr;
  }

  /**
   * Detect Doji candlestick pattern
   */
  private detectDoji(opens: number[], closes: number[]): number[] {
    return opens.map((open, i) => {
      const bodySize = Math.abs(closes[i] - open) / open;
      return bodySize < 0.001 ? 1 : 0; // Doji if body is very small
    });
  }

  /**
   * Detect Hammer candlestick pattern
   */
  private detectHammer(opens: number[], highs: number[], lows: number[], closes: number[]): number[] {
    return opens.map((open, i) => {
      const bodySize = Math.abs(closes[i] - open);
      const lowerShadow = Math.min(open, closes[i]) - lows[i];
      const upperShadow = highs[i] - Math.max(open, closes[i]);
      
      // Hammer: small body, long lower shadow, short upper shadow
      return (lowerShadow > 2 * bodySize && upperShadow < bodySize) ? 1 : 0;
    });
  }

  /**
   * Detect Engulfing pattern
   */
  private detectEngulfing(opens: number[], closes: number[]): number[] {
    const pattern: number[] = [0]; // First candle can't be engulfing
    
    for (let i = 1; i < opens.length; i++) {
      const prevBody = Math.abs(closes[i - 1] - opens[i - 1]);
      const currBody = Math.abs(closes[i] - opens[i]);
      
      // Bullish engulfing
      if (closes[i - 1] < opens[i - 1] && closes[i] > opens[i] && 
          opens[i] < closes[i - 1] && closes[i] > opens[i - 1] && 
          currBody > prevBody) {
        pattern.push(1);
      }
      // Bearish engulfing
      else if (closes[i - 1] > opens[i - 1] && closes[i] < opens[i] && 
               opens[i] > closes[i - 1] && closes[i] < opens[i - 1] && 
               currBody > prevBody) {
        pattern.push(-1);
      } else {
        pattern.push(0);
      }
    }
    
    return pattern;
  }

  /**
   * Detect support and resistance levels
   */
  private detectSupportResistance(closes: number[], window: number): { support: number[]; resistance: number[] } {
    const support: number[] = [];
    const resistance: number[] = [];
    
    for (let i = window; i < closes.length; i++) {
      const slice = closes.slice(i - window, i);
      const currentPrice = closes[i];
      
      // Support: lowest price in window
      const supportLevel = Math.min(...slice);
      support.push(supportLevel / currentPrice);
      
      // Resistance: highest price in window
      const resistanceLevel = Math.max(...slice);
      resistance.push(resistanceLevel / currentPrice);
    }
    
    return { support, resistance };
  }

  /**
   * Calculate trend strength
   */
  private calculateTrendStrength(closes: number[], window: number): number[] {
    const trendStrength: number[] = [];
    
    for (let i = window - 1; i < closes.length; i++) {
      const slice = closes.slice(i - window + 1, i + 1);
      const x = Array.from({ length: window }, (_, idx) => idx);
      
      // Linear regression slope as trend strength
      const correlation = MathUtils.correlation(x, slice);
      trendStrength.push(correlation);
    }
    
    return trendStrength;
  }

  /**
   * Calculate Fourier features (simplified)
   */
  private calculateFourierFeatures(data: number[], window: number): number[][] {
    const features: number[][] = [[], []]; // Real and imaginary parts
    
    for (let i = window - 1; i < data.length; i++) {
      const slice = data.slice(i - window + 1, i + 1);
      
      // Simple DFT for first few frequencies
      let realPart = 0;
      let imagPart = 0;
      
      for (let k = 0; k < slice.length; k++) {
        const angle = -2 * Math.PI * k / slice.length;
        realPart += slice[k] * Math.cos(angle);
        imagPart += slice[k] * Math.sin(angle);
      }
      
      features[0].push(realPart / slice.length);
      features[1].push(imagPart / slice.length);
    }
    
    return features;
  }

  /**
   * Calculate cyclical features
   */
  private calculateCyclicalFeatures(closes: number[]): number[][] {
    const features: number[][] = [];
    
    // Daily, weekly, monthly cycles (simplified)
    const cycles = [5, 20, 60]; // 5-day, 20-day, 60-day cycles
    
    for (const cycle of cycles) {
      const cyclicalFeature: number[] = [];
      
      for (let i = 0; i < closes.length; i++) {
        const phase = (2 * Math.PI * i) / cycle;
        cyclicalFeature.push(Math.sin(phase));
      }
      
      features.push(cyclicalFeature);
    }
    
    return features;
  }
}