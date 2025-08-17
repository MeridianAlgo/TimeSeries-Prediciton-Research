/**
 * MeridianAlgo-JS v2.0 - Ultra-Precision Trading Library
 * 
 * Advanced algorithmic trading and financial analysis library with
 * ultra-precision machine learning capabilities for JavaScript/TypeScript.
 * 
 * @author MeridianAlgo Team
 * @version 2.0.0
 * @license MIT
 */

// Core Predictors
export { UltraPrecisionPredictor } from './predictors/UltraPrecisionPredictor';

// Feature Engineering
export { FeatureEngineer } from './features/FeatureEngineer';
export { TechnicalIndicators } from './indicators/TechnicalIndicators';

// Utilities
export { MathUtils } from './utils/MathUtils';
export { StatisticsUtils } from './utils/StatisticsUtils';
export { ValidationUtils } from './utils/ValidationUtils';

// Types
export { MarketData, ExtendedMarketData, TickData, OrderBook, ValidationResult } from './types/MarketData';
export { PredictorOptions, TrainingData, PredictionResult, ModelMetrics, TrainingResults } from './types/Prediction';
export { FeatureOptions, FeatureMatrix, FeatureMetadata } from './types/Features';

// Constants
export { DEFAULT_CONFIG } from './config/defaults';
export { INDICATORS_CONFIG } from './config/indicators';

/**
 * Library version
 */
export const VERSION = '2.0.0';