/**
 * Indicators Configuration
 * 
 * Configuration settings for technical indicators and their parameters.
 */

/**
 * Technical indicator configurations
 */
export const INDICATORS_CONFIG = {
  // Moving Averages
  SMA: {
    defaultPeriods: [5, 10, 20, 50, 100, 200],
    minPeriod: 2,
    maxPeriod: 500
  },
  
  EMA: {
    defaultPeriods: [5, 10, 20, 50, 100, 200],
    minPeriod: 2,
    maxPeriod: 500
  },
  
  WMA: {
    defaultPeriods: [5, 10, 20, 50, 100],
    minPeriod: 2,
    maxPeriod: 200
  },
  
  // Momentum Indicators
  RSI: {
    defaultPeriod: 14,
    alternativePeriods: [7, 9, 14, 21, 25],
    overboughtLevel: 70,
    oversoldLevel: 30,
    minPeriod: 2,
    maxPeriod: 100
  },
  
  MACD: {
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    alternativeSettings: [
      { fast: 5, slow: 35, signal: 5 },
      { fast: 8, slow: 17, signal: 9 },
      { fast: 12, slow: 26, signal: 9 }
    ]
  },
  
  STOCHASTIC: {
    kPeriod: 14,
    dPeriod: 3,
    smoothing: 3,
    overboughtLevel: 80,
    oversoldLevel: 20,
    alternativeSettings: [
      { k: 5, d: 3 },
      { k: 14, d: 3 },
      { k: 21, d: 5 }
    ]
  },
  
  WILLIAMS_R: {
    defaultPeriod: 14,
    alternativePeriods: [7, 14, 21],
    overboughtLevel: -20,
    oversoldLevel: -80
  },
  
  CCI: {
    defaultPeriod: 20,
    alternativePeriods: [14, 20, 50],
    overboughtLevel: 100,
    oversoldLevel: -100,
    constant: 0.015
  },
  
  // Volatility Indicators
  BOLLINGER_BANDS: {
    period: 20,
    multiplier: 2,
    alternativeSettings: [
      { period: 10, multiplier: 1.9 },
      { period: 20, multiplier: 2.0 },
      { period: 50, multiplier: 2.1 }
    ]
  },
  
  ATR: {
    defaultPeriod: 14,
    alternativePeriods: [7, 14, 21, 50],
    minPeriod: 2,
    maxPeriod: 100
  },
  
  // Trend Indicators
  ADX: {
    defaultPeriod: 14,
    alternativePeriods: [7, 14, 21],
    trendThreshold: 25,
    strongTrendThreshold: 40
  },
  
  PARABOLIC_SAR: {
    accelerationFactor: 0.02,
    maxAcceleration: 0.20,
    alternativeSettings: [
      { af: 0.01, max: 0.10 },
      { af: 0.02, max: 0.20 },
      { af: 0.03, max: 0.30 }
    ]
  },
  
  // Volume Indicators
  OBV: {
    // No parameters needed
  },
  
  MFI: {
    defaultPeriod: 14,
    alternativePeriods: [10, 14, 20],
    overboughtLevel: 80,
    oversoldLevel: 20
  },
  
  VWAP: {
    // Typically calculated from session start
    resetPeriod: 'session'
  },
  
  // Oscillators
  MOMENTUM: {
    defaultPeriod: 10,
    alternativePeriods: [5, 10, 20, 50]
  },
  
  ROC: {
    defaultPeriod: 10,
    alternativePeriods: [5, 10, 20, 50]
  },
  
  // Statistical Indicators
  STANDARD_DEVIATION: {
    defaultPeriod: 20,
    alternativePeriods: [10, 20, 50]
  },
  
  VARIANCE: {
    defaultPeriod: 20,
    alternativePeriods: [10, 20, 50]
  },
  
  LINEAR_REGRESSION_SLOPE: {
    defaultPeriod: 14,
    alternativePeriods: [7, 14, 21, 50]
  }
};

/**
 * Indicator combinations for different strategies
 */
export const INDICATOR_COMBINATIONS = {
  TREND_FOLLOWING: [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'ADX'
  ],
  
  MEAN_REVERSION: [
    'RSI_14', 'BOLLINGER_BANDS', 'STOCHASTIC', 'WILLIAMS_R', 'CCI'
  ],
  
  MOMENTUM: [
    'RSI_14', 'MACD', 'MOMENTUM_10', 'ROC_10', 'STOCHASTIC'
  ],
  
  VOLATILITY: [
    'ATR_14', 'BOLLINGER_BANDS', 'STANDARD_DEVIATION_20'
  ],
  
  VOLUME: [
    'OBV', 'MFI_14', 'VWAP'
  ],
  
  COMPREHENSIVE: [
    'SMA_20', 'EMA_12', 'RSI_14', 'MACD', 'BOLLINGER_BANDS',
    'STOCHASTIC', 'ATR_14', 'ADX', 'OBV', 'MFI_14'
  ]
};

/**
 * Indicator signal interpretations
 */
export const SIGNAL_INTERPRETATIONS = {
  RSI: {
    bullish: 'RSI < 30 (oversold)',
    bearish: 'RSI > 70 (overbought)',
    neutral: '30 <= RSI <= 70'
  },
  
  MACD: {
    bullish: 'MACD line crosses above signal line',
    bearish: 'MACD line crosses below signal line',
    momentum: 'Histogram increasing/decreasing'
  },
  
  BOLLINGER_BANDS: {
    bullish: 'Price touches lower band (oversold)',
    bearish: 'Price touches upper band (overbought)',
    squeeze: 'Bands contracting (low volatility)',
    expansion: 'Bands expanding (high volatility)'
  },
  
  STOCHASTIC: {
    bullish: '%K crosses above %D below 20',
    bearish: '%K crosses below %D above 80',
    overbought: '%K and %D > 80',
    oversold: '%K and %D < 20'
  },
  
  ADX: {
    trending: 'ADX > 25',
    strongTrend: 'ADX > 40',
    weakTrend: 'ADX < 25',
    bullishTrend: '+DI > -DI and ADX > 25',
    bearishTrend: '-DI > +DI and ADX > 25'
  }
};

/**
 * Indicator optimization parameters
 */
export const OPTIMIZATION_PARAMETERS = {
  RSI: {
    period: { min: 5, max: 50, step: 1 },
    overbought: { min: 60, max: 90, step: 5 },
    oversold: { min: 10, max: 40, step: 5 }
  },
  
  MACD: {
    fastPeriod: { min: 5, max: 20, step: 1 },
    slowPeriod: { min: 15, max: 40, step: 1 },
    signalPeriod: { min: 5, max: 15, step: 1 }
  },
  
  BOLLINGER_BANDS: {
    period: { min: 10, max: 50, step: 5 },
    multiplier: { min: 1.5, max: 3.0, step: 0.1 }
  },
  
  STOCHASTIC: {
    kPeriod: { min: 5, max: 30, step: 1 },
    dPeriod: { min: 2, max: 10, step: 1 }
  },
  
  ATR: {
    period: { min: 5, max: 50, step: 1 }
  }
};

/**
 * Indicator performance metrics
 */
export const PERFORMANCE_METRICS = {
  CALCULATION_COMPLEXITY: {
    SMA: 'O(n)',
    EMA: 'O(n)',
    RSI: 'O(n)',
    MACD: 'O(n)',
    BOLLINGER_BANDS: 'O(n)',
    STOCHASTIC: 'O(n)',
    ATR: 'O(n)',
    ADX: 'O(n)',
    MFI: 'O(n)'
  },
  
  MEMORY_USAGE: {
    SMA: 'Low',
    EMA: 'Very Low',
    RSI: 'Low',
    MACD: 'Medium',
    BOLLINGER_BANDS: 'Medium',
    STOCHASTIC: 'Medium',
    ATR: 'Low',
    ADX: 'High',
    MFI: 'Medium'
  },
  
  SENSITIVITY_TO_NOISE: {
    SMA: 'Low',
    EMA: 'Medium',
    RSI: 'Medium',
    MACD: 'High',
    BOLLINGER_BANDS: 'Low',
    STOCHASTIC: 'High',
    ATR: 'Low',
    ADX: 'Medium',
    MFI: 'Medium'
  }
};

/**
 * Market condition adaptations
 */
export const MARKET_ADAPTATIONS = {
  TRENDING_MARKET: {
    preferredIndicators: ['MACD', 'ADX', 'EMA', 'PARABOLIC_SAR'],
    avoidIndicators: ['RSI', 'STOCHASTIC', 'WILLIAMS_R'],
    parameterAdjustments: {
      RSI: { period: 21 }, // Longer period for trending markets
      MACD: { fast: 8, slow: 21, signal: 5 } // Faster settings
    }
  },
  
  RANGING_MARKET: {
    preferredIndicators: ['RSI', 'STOCHASTIC', 'BOLLINGER_BANDS', 'WILLIAMS_R'],
    avoidIndicators: ['MACD', 'MOMENTUM', 'ROC'],
    parameterAdjustments: {
      RSI: { period: 14, overbought: 75, oversold: 25 },
      BOLLINGER_BANDS: { multiplier: 1.8 } // Tighter bands
    }
  },
  
  HIGH_VOLATILITY: {
    preferredIndicators: ['ATR', 'BOLLINGER_BANDS', 'STANDARD_DEVIATION'],
    parameterAdjustments: {
      ATR: { period: 10 }, // Shorter period for faster adaptation
      BOLLINGER_BANDS: { multiplier: 2.5 }, // Wider bands
      RSI: { period: 21 } // Longer period to reduce noise
    }
  },
  
  LOW_VOLATILITY: {
    preferredIndicators: ['RSI', 'STOCHASTIC', 'CCI'],
    parameterAdjustments: {
      RSI: { period: 9, overbought: 65, oversold: 35 }, // More sensitive
      BOLLINGER_BANDS: { multiplier: 1.5 }, // Tighter bands
      STOCHASTIC: { kPeriod: 9 } // Faster stochastic
    }
  }
};

/**
 * Timeframe-specific configurations
 */
export const TIMEFRAME_CONFIGS = {
  '1m': {
    RSI: { period: 7 },
    MACD: { fast: 5, slow: 13, signal: 4 },
    BOLLINGER_BANDS: { period: 10 }
  },
  
  '5m': {
    RSI: { period: 9 },
    MACD: { fast: 8, slow: 17, signal: 6 },
    BOLLINGER_BANDS: { period: 15 }
  },
  
  '15m': {
    RSI: { period: 12 },
    MACD: { fast: 10, slow: 21, signal: 7 },
    BOLLINGER_BANDS: { period: 18 }
  },
  
  '1h': {
    RSI: { period: 14 },
    MACD: { fast: 12, slow: 26, signal: 9 },
    BOLLINGER_BANDS: { period: 20 }
  },
  
  '4h': {
    RSI: { period: 16 },
    MACD: { fast: 15, slow: 30, signal: 10 },
    BOLLINGER_BANDS: { period: 25 }
  },
  
  '1d': {
    RSI: { period: 14 },
    MACD: { fast: 12, slow: 26, signal: 9 },
    BOLLINGER_BANDS: { period: 20 }
  }
};