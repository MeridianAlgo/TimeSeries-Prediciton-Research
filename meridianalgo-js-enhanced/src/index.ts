/**
 * MeridianAlgo-JS Enhanced - Advanced Machine Learning for Financial Markets
 * 
 * Enhanced version building upon the original meridianalgo-js with:
 * - Ultra-precision prediction algorithms
 * - Advanced feature engineering (1000+ features)
 * - Real-time trading capabilities
 * - Portfolio optimization
 * - Risk management
 * - Market regime detection
 */

// Core prediction engines
export { UltraPrecisionPredictor } from './predictors/UltraPrecisionPredictor';
export { EnsemblePredictor } from './predictors/EnsemblePredictor';
export { NeuralNetworkPredictor } from './predictors/NeuralNetworkPredictor';
export { RealtimePredictor } from './predictors/RealtimePredictor';

// Feature engineering
export { FeatureEngineer } from './features/FeatureEngineer';
export { TechnicalIndicators } from './features/TechnicalIndicators';
export { AdvancedBollingerBands } from './features/AdvancedBollingerBands';
export { MultiRSISystem } from './features/MultiRSISystem';
export { MicrostructureAnalyzer } from './features/MicrostructureAnalyzer';
export { VolatilityAnalyzer } from './features/VolatilityAnalyzer';
export { HarmonicAnalyzer } from './features/HarmonicAnalyzer';

// Market analysis
export { MarketRegimeDetector } from './analysis/MarketRegimeDetector';
export { TrendAnalyzer } from './analysis/TrendAnalyzer';
export { SentimentAnalyzer } from './analysis/SentimentAnalyzer';
export { SeasonalityDetector } from './analysis/SeasonalityDetector';

// Portfolio management
export { PortfolioOptimizer } from './portfolio/PortfolioOptimizer';
export { RiskManager } from './portfolio/RiskManager';
export { PositionSizer } from './portfolio/PositionSizer';
export { BacktestEngine } from './portfolio/BacktestEngine';

// Utilities
export { DataProcessor } from './utils/DataProcessor';
export { ModelValidator } from './utils/ModelValidator';
export { PerformanceMetrics } from './utils/PerformanceMetrics';
export { DataGenerator } from './utils/DataGenerator';

// Types and interfaces
export * from './types/MarketData';
export * from './types/PredictionTypes';
export * from './types/FeatureTypes';
export * from './types/PortfolioTypes';
export * from './types/ConfigTypes';

// Constants and configurations
export { DEFAULT_CONFIG } from './config/DefaultConfig';
export { FEATURE_SETS } from './config/FeatureSets';
export { MODEL_PRESETS } from './config/ModelPresets';

// Version info
export const VERSION = '3.0.0';
export const BUILD_DATE = new Date().toISOString();

/**
 * Main entry point for quick setup
 */
export class MeridianAlgo {
  /**
   * Create a quick predictor with sensible defaults
   */
  static createPredictor(options?: Partial<any>) {
    return new UltraPrecisionPredictor({
      targetAccuracy: 0.01,
      features: ['bollinger', 'rsi', 'macd', 'volatility'],
      models: ['randomForest', 'neuralNetwork'],
      ...options
    });
  }

  /**
   * Create a feature engineer with advanced capabilities
   */
  static createFeatureEngineer(options?: Partial<any>) {
    return new FeatureEngineer({
      generators: [
        'advancedBollinger',
        'multiRSI', 
        'microstructure',
        'volatilityAnalysis',
        'harmonicAnalysis'
      ],
      ...options
    });
  }

  /**
   * Create a portfolio optimizer
   */
  static createPortfolioOptimizer(options?: Partial<any>) {
    return new PortfolioOptimizer({
      objective: 'sharpe',
      constraints: {
        maxWeight: 0.4,
        minWeight: 0.05
      },
      ...options
    });
  }
}