/**
 * Feature Engineering Types
 * 
 * Type definitions for feature engineering and technical analysis.
 */

/**
 * Feature engineering configuration options
 */
export interface FeatureOptions {
  /** Target number of features to generate */
  targetFeatureCount: number;
  /** Enable advanced technical features */
  enableAdvancedFeatures: boolean;
  /** Enable microstructure features */
  enableMicrostructure: boolean;
  /** Enable volatility features */
  enableVolatilityFeatures: boolean;
  /** Enable statistical features */
  enableStatisticalFeatures: boolean;
  /** Enable harmonic and cyclical features */
  enableHarmonicFeatures: boolean;
  /** Enable cross-asset features */
  enableCrossAssetFeatures: boolean;
  /** Lookback periods for rolling calculations */
  lookbackPeriods: number[];
  /** Technical indicator configurations */
  technicalIndicators: TechnicalIndicatorConfig;
}

/**
 * Technical indicator configuration
 */
export interface TechnicalIndicatorConfig {
  /** RSI configuration */
  rsi: {
    periods: number[];
  };
  /** MACD configuration */
  macd: {
    fast: number;
    slow: number;
    signal: number;
  };
  /** Bollinger Bands configuration */
  bollinger: {
    period: number;
    multiplier: number;
  };
  /** Stochastic Oscillator configuration */
  stochastic: {
    kPeriod: number;
    dPeriod: number;
  };
  /** Williams %R configuration */
  williams: {
    period: number;
  };
  /** Commodity Channel Index configuration */
  cci: {
    period: number;
  };
}

/**
 * Feature matrix containing generated features
 */
export interface FeatureMatrix {
  /** Feature data matrix (rows = samples, columns = features) */
  data: number[][];
  /** Feature names */
  featureNames: string[];
  /** Feature metadata */
  metadata: FeatureMetadata[];
  /** Number of columns (features) */
  columns: number;
  /** Number of rows (samples) */
  rows: number;
}

/**
 * Metadata for individual features
 */
export interface FeatureMetadata {
  /** Feature name */
  name: string;
  /** Feature category */
  category: 'technical' | 'statistical' | 'microstructure' | 'volatility' | 'harmonic' | 'cross-asset';
  /** Feature description */
  description: string;
  /** Data type */
  dataType: 'numeric' | 'categorical' | 'binary';
  /** Value range */
  range?: {
    min: number;
    max: number;
  };
  /** Statistical properties */
  statistics?: {
    mean: number;
    std: number;
    skewness: number;
    kurtosis: number;
  };
  /** Missing value handling */
  missingValueStrategy: 'drop' | 'forward_fill' | 'backward_fill' | 'interpolate' | 'zero' | 'mean';
  /** Transformation applied */
  transformation?: 'none' | 'log' | 'sqrt' | 'normalize' | 'standardize';
}

/**
 * Technical indicator result structures
 */
export interface RSIResult {
  /** RSI values */
  values: number[];
  /** Overbought threshold */
  overbought: number;
  /** Oversold threshold */
  oversold: number;
}

export interface MACDResult {
  /** MACD line values */
  macd: number[];
  /** Signal line values */
  signal: number[];
  /** Histogram values */
  histogram: number[];
}

export interface BollingerBandsResult {
  /** Upper band values */
  upper: number[];
  /** Middle band (SMA) values */
  middle: number[];
  /** Lower band values */
  lower: number[];
  /** Bandwidth values */
  bandwidth: number[];
  /** %B values (position within bands) */
  percentB: number[];
}

export interface StochasticResult {
  /** %K values */
  k: number[];
  /** %D values */
  d: number[];
}

export interface IchimokuResult {
  /** Tenkan-sen (Conversion Line) */
  tenkanSen: number[];
  /** Kijun-sen (Base Line) */
  kijunSen: number[];
  /** Senkou Span A (Leading Span A) */
  senkouSpanA: number[];
  /** Senkou Span B (Leading Span B) */
  senkouSpanB: number[];
  /** Chikou Span (Lagging Span) */
  chikouSpan: number[];
}

export interface AroonResult {
  /** Aroon Up values */
  up: number[];
  /** Aroon Down values */
  down: number[];
  /** Aroon Oscillator values */
  oscillator: number[];
}

/**
 * Microstructure feature configuration
 */
export interface MicrostructureConfig {
  /** Enable bid-ask spread analysis */
  bidAskSpread: boolean;
  /** Enable order flow analysis */
  orderFlow: boolean;
  /** Enable price impact analysis */
  priceImpact: boolean;
  /** Enable liquidity analysis */
  liquidity: boolean;
  /** VWAP periods */
  vwapPeriods: number[];
  /** Order book depth levels */
  orderBookDepth: number;
}

/**
 * Volatility feature configuration
 */
export interface VolatilityConfig {
  /** Volatility estimators to use */
  estimators: ('close' | 'parkinson' | 'garmanKlass' | 'rogersSatchell' | 'yangZhang')[];
  /** Rolling window sizes */
  windows: number[];
  /** Enable GARCH modeling */
  enableGarch: boolean;
  /** Enable volatility clustering analysis */
  enableClustering: boolean;
  /** Enable volatility forecasting */
  enableForecasting: boolean;
}

/**
 * Statistical feature configuration
 */
export interface StatisticalConfig {
  /** Rolling statistics to calculate */
  statistics: ('mean' | 'std' | 'skewness' | 'kurtosis' | 'min' | 'max' | 'median' | 'quantile')[];
  /** Window sizes for rolling calculations */
  windows: number[];
  /** Quantile levels */
  quantiles: number[];
  /** Enable correlation analysis */
  enableCorrelation: boolean;
  /** Enable cointegration analysis */
  enableCointegration: boolean;
}

/**
 * Harmonic feature configuration
 */
export interface HarmonicConfig {
  /** Enable Fourier transform features */
  enableFourier: boolean;
  /** Enable wavelet transform features */
  enableWavelet: boolean;
  /** Enable cyclical pattern detection */
  enableCyclical: boolean;
  /** Frequency bands for analysis */
  frequencyBands: number[];
  /** Wavelet types */
  waveletTypes: string[];
}

/**
 * Feature selection configuration
 */
export interface FeatureSelectionConfig {
  /** Selection method */
  method: 'correlation' | 'mutual_information' | 'chi2' | 'f_test' | 'recursive' | 'lasso';
  /** Number of features to select */
  numFeatures: number;
  /** Selection threshold */
  threshold?: number;
  /** Cross-validation folds for selection */
  cvFolds: number;
}

/**
 * Feature transformation configuration
 */
export interface FeatureTransformationConfig {
  /** Scaling method */
  scaling: 'none' | 'standard' | 'minmax' | 'robust' | 'quantile';
  /** Dimensionality reduction method */
  dimensionalityReduction?: 'pca' | 'ica' | 'lda' | 'tsne' | 'umap';
  /** Number of components for dimensionality reduction */
  numComponents?: number;
  /** Handle missing values */
  handleMissing: boolean;
  /** Missing value strategy */
  missingStrategy: 'drop' | 'impute' | 'interpolate';
}

/**
 * Feature importance analysis
 */
export interface FeatureImportanceAnalysis {
  /** Feature importance scores */
  scores: number[];
  /** Feature rankings */
  rankings: number[];
  /** Feature names */
  names: string[];
  /** Importance method used */
  method: string;
  /** Statistical significance */
  significance?: number[];
}

/**
 * Feature correlation analysis
 */
export interface FeatureCorrelationAnalysis {
  /** Correlation matrix */
  correlationMatrix: number[][];
  /** Highly correlated feature pairs */
  highCorrelations: Array<{
    feature1: string;
    feature2: string;
    correlation: number;
  }>;
  /** Recommended features to remove */
  redundantFeatures: string[];
}

/**
 * Feature engineering pipeline
 */
export interface FeaturePipeline {
  /** Pipeline steps */
  steps: FeaturePipelineStep[];
  /** Pipeline configuration */
  config: FeatureOptions;
  /** Fitted pipeline state */
  fitted: boolean;
}

export interface FeaturePipelineStep {
  /** Step name */
  name: string;
  /** Step type */
  type: 'generator' | 'transformer' | 'selector';
  /** Step configuration */
  config: Record<string, unknown>;
  /** Step execution order */
  order: number;
}