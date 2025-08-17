/**
 * Default Configuration
 * 
 * Default settings and constants for the MeridianAlgo-JS library.
 */

import { PredictorOptions } from '../types/Prediction';
import { FeatureOptions } from '../types/Features';
import { OptimizerOptions } from '../types/Portfolio';

/**
 * Default predictor configuration
 */
export const DEFAULT_PREDICTOR_OPTIONS: Required<PredictorOptions> = {
  targetErrorRate: 0.01,
  ensembleSize: 10,
  featureCount: 1000,
  trainingRatio: 0.8,
  crossValidationFolds: 5,
  hyperparameterTuning: true,
  parallelProcessing: true,
  cacheFeatures: true,
  incrementalLearning: false,
  updateFrequency: 'batch',
  predictionHorizon: 1,
  confidenceThreshold: 0.8
};

/**
 * Default feature engineering configuration
 */
export const DEFAULT_FEATURE_OPTIONS: Required<FeatureOptions> = {
  targetFeatureCount: 1000,
  enableAdvancedFeatures: true,
  enableMicrostructure: true,
  enableVolatilityFeatures: true,
  enableStatisticalFeatures: true,
  enableHarmonicFeatures: true,
  enableCrossAssetFeatures: false,
  lookbackPeriods: [5, 10, 20, 50, 100, 200],
  technicalIndicators: {
    rsi: { periods: [7, 14, 21, 28] },
    macd: { fast: 12, slow: 26, signal: 9 },
    bollinger: { period: 20, multiplier: 2 },
    stochastic: { kPeriod: 14, dPeriod: 3 },
    williams: { period: 14 },
    cci: { period: 20 }
  }
};

/**
 * Default portfolio optimizer configuration
 */
export const DEFAULT_OPTIMIZER_OPTIONS: OptimizerOptions = {
  objective: 'sharpe',
  constraints: {
    minWeight: 0.0,
    maxWeight: 1.0,
    minTotalWeight: 0.99,
    maxTotalWeight: 1.01,
    longOnly: true
  },
  riskModel: 'historical',
  optimizationMethod: 'quadratic',
  rebalanceFrequency: 'monthly'
};

/**
 * Default library configuration
 */
export const DEFAULT_CONFIG = {
  predictor: DEFAULT_PREDICTOR_OPTIONS,
  features: DEFAULT_FEATURE_OPTIONS,
  optimizer: DEFAULT_OPTIMIZER_OPTIONS,
  
  // Performance settings
  performance: {
    enableParallelProcessing: true,
    maxWorkers: 4,
    cacheSize: 1000,
    memoryLimit: 512 * 1024 * 1024, // 512MB
    enableProfiling: false
  },
  
  // Validation settings
  validation: {
    strictMode: true,
    autoSanitize: true,
    warningsAsErrors: false
  },
  
  // Logging settings
  logging: {
    level: 'info',
    enableConsole: true,
    enableFile: false,
    maxLogSize: 10 * 1024 * 1024 // 10MB
  }
};

/**
 * Mathematical constants
 */
export const MATH_CONSTANTS = {
  EPSILON: 1e-10,
  PI: Math.PI,
  E: Math.E,
  GOLDEN_RATIO: (1 + Math.sqrt(5)) / 2,
  SQRT_2: Math.sqrt(2),
  SQRT_PI: Math.sqrt(Math.PI)
};

/**
 * Financial constants
 */
export const FINANCIAL_CONSTANTS = {
  TRADING_DAYS_PER_YEAR: 252,
  CALENDAR_DAYS_PER_YEAR: 365,
  HOURS_PER_TRADING_DAY: 6.5,
  MINUTES_PER_TRADING_DAY: 390,
  SECONDS_PER_TRADING_DAY: 23400,
  
  // Risk-free rates (approximate)
  US_TREASURY_10Y: 0.04, // 4%
  US_TREASURY_3M: 0.035,  // 3.5%
  
  // Market benchmarks
  SP500_ANNUAL_RETURN: 0.10,    // 10%
  SP500_ANNUAL_VOLATILITY: 0.16, // 16%
  
  // Common thresholds
  OVERBOUGHT_RSI: 70,
  OVERSOLD_RSI: 30,
  HIGH_VOLATILITY_THRESHOLD: 0.25, // 25%
  LOW_VOLATILITY_THRESHOLD: 0.10   // 10%
};

/**
 * Data validation constants
 */
export const VALIDATION_CONSTANTS = {
  MIN_DATA_POINTS: 50,
  MAX_MISSING_DATA_RATIO: 0.05, // 5%
  MAX_OUTLIER_RATIO: 0.02,       // 2%
  
  // Price validation
  MAX_DAILY_RETURN: 0.50,        // 50%
  MIN_PRICE: 0.001,
  MAX_PRICE: 1000000,
  
  // Volume validation
  MIN_VOLUME: 0,
  MAX_VOLUME_SPIKE: 10, // 10x average volume
  
  // Time validation
  MAX_TIME_GAP_HOURS: 72, // 3 days
  MIN_TIME_INTERVAL_MS: 1000 // 1 second
};

/**
 * Performance benchmarks
 */
export const PERFORMANCE_BENCHMARKS = {
  // Target processing times (milliseconds)
  FEATURE_GENERATION_TARGET: 100,
  PREDICTION_TARGET: 10,
  PORTFOLIO_OPTIMIZATION_TARGET: 1000,
  
  // Memory usage targets (bytes)
  MAX_FEATURE_MATRIX_SIZE: 100 * 1024 * 1024, // 100MB
  MAX_MODEL_SIZE: 50 * 1024 * 1024,           // 50MB
  
  // Accuracy targets
  MIN_PREDICTION_ACCURACY: 0.55,  // 55%
  TARGET_PREDICTION_ACCURACY: 0.65, // 65%
  EXCELLENT_PREDICTION_ACCURACY: 0.75 // 75%
};

/**
 * Error codes and messages
 */
export const ERROR_CODES = {
  // Data errors
  INVALID_DATA: 'INVALID_DATA',
  INSUFFICIENT_DATA: 'INSUFFICIENT_DATA',
  MISSING_DATA: 'MISSING_DATA',
  
  // Model errors
  MODEL_NOT_TRAINED: 'MODEL_NOT_TRAINED',
  TRAINING_FAILED: 'TRAINING_FAILED',
  PREDICTION_FAILED: 'PREDICTION_FAILED',
  
  // Configuration errors
  INVALID_CONFIG: 'INVALID_CONFIG',
  MISSING_CONFIG: 'MISSING_CONFIG',
  
  // Performance errors
  MEMORY_LIMIT_EXCEEDED: 'MEMORY_LIMIT_EXCEEDED',
  TIMEOUT_EXCEEDED: 'TIMEOUT_EXCEEDED',
  
  // Portfolio errors
  OPTIMIZATION_FAILED: 'OPTIMIZATION_FAILED',
  INVALID_WEIGHTS: 'INVALID_WEIGHTS',
  CONSTRAINT_VIOLATION: 'CONSTRAINT_VIOLATION'
};

/**
 * Feature categories
 */
export const FEATURE_CATEGORIES = {
  TECHNICAL: 'technical',
  STATISTICAL: 'statistical',
  MICROSTRUCTURE: 'microstructure',
  VOLATILITY: 'volatility',
  HARMONIC: 'harmonic',
  CROSS_ASSET: 'cross_asset',
  SENTIMENT: 'sentiment',
  FUNDAMENTAL: 'fundamental'
};

/**
 * Model types
 */
export const MODEL_TYPES = {
  RANDOM_FOREST: 'randomForest',
  NEURAL_NETWORK: 'neuralNetwork',
  SVM: 'svm',
  GRADIENT_BOOSTING: 'gradientBoosting',
  LINEAR_REGRESSION: 'linearRegression',
  ENSEMBLE: 'ensemble'
};

/**
 * Optimization objectives
 */
export const OPTIMIZATION_OBJECTIVES = {
  SHARPE: 'sharpe',
  RETURN: 'return',
  RISK: 'risk',
  SORTINO: 'sortino',
  CALMAR: 'calmar',
  CUSTOM: 'custom'
};

/**
 * Rebalancing frequencies
 */
export const REBALANCE_FREQUENCIES = {
  DAILY: 'daily',
  WEEKLY: 'weekly',
  MONTHLY: 'monthly',
  QUARTERLY: 'quarterly',
  ANNUALLY: 'annually',
  THRESHOLD: 'threshold'
};

/**
 * Market regimes
 */
export const MARKET_REGIMES = {
  BULL: 'bull',
  BEAR: 'bear',
  SIDEWAYS: 'sideways',
  VOLATILE: 'volatile',
  CALM: 'calm'
};

/**
 * Risk levels
 */
export const RISK_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  EXTREME: 'extreme'
};

/**
 * Alert severities
 */
export const ALERT_SEVERITIES = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
};