/**
 * Prediction Types
 * 
 * Type definitions for machine learning prediction models and results.
 */

/**
 * Configuration options for ultra-precision predictor
 */
export interface PredictorOptions {
  /** Target error rate (e.g., 0.01 for 1% error) */
  targetErrorRate: number;
  /** Number of models in the ensemble */
  ensembleSize: number;
  /** Target number of features to generate */
  featureCount: number;
  /** Ratio of data to use for training (rest for validation) */
  trainingRatio: number;
  /** Number of cross-validation folds */
  crossValidationFolds: number;
  /** Enable automatic hyperparameter tuning */
  hyperparameterTuning: boolean;
  /** Enable parallel processing */
  parallelProcessing: boolean;
  /** Enable feature caching */
  cacheFeatures: boolean;
  /** Enable incremental learning */
  incrementalLearning: boolean;
  /** Update frequency for real-time models */
  updateFrequency: 'tick' | 'batch' | 'realtime';
  /** Number of periods to predict ahead */
  predictionHorizon: number;
  /** Minimum confidence threshold for predictions */
  confidenceThreshold: number;
}

/**
 * Prediction result with confidence and metadata
 */
export interface PredictionResult {
  /** Predicted value */
  value: number;
  /** Confidence score (0-1) */
  confidence: number;
  /** Timestamp of prediction */
  timestamp: Date;
  /** Input features used for prediction */
  features: number[];
  /** Model version used */
  modelVersion: string;
  /** Additional metadata */
  metadata?: {
    /** Feature importance scores */
    featureImportance?: number[];
    /** Individual model predictions (for ensemble) */
    modelPredictions?: number[];
    /** Prediction interval bounds */
    bounds?: {
      lower: number;
      upper: number;
    };
  };
}

/**
 * Batch prediction results
 */
export interface BatchPredictionResult {
  /** Array of predictions */
  predictions: PredictionResult[];
  /** Overall batch statistics */
  statistics: {
    /** Mean prediction value */
    mean: number;
    /** Standard deviation of predictions */
    std: number;
    /** Minimum prediction value */
    min: number;
    /** Maximum prediction value */
    max: number;
    /** Average confidence */
    avgConfidence: number;
  };
  /** Processing time in milliseconds */
  processingTime: number;
}

/**
 * Model performance metrics
 */
export interface ModelMetrics {
  /** Mean Absolute Error */
  mae: number;
  /** Mean Squared Error */
  mse: number;
  /** Root Mean Squared Error */
  rmse: number;
  /** R-squared coefficient */
  r2: number;
  /** Directional accuracy (percentage of correct direction predictions) */
  directionalAccuracy: number;
  /** Number of samples used for evaluation */
  sampleCount: number;
  /** Additional custom metrics */
  custom?: Record<string, number>;
}

/**
 * Training data structure for machine learning models
 */
export interface TrainingData {
  /** Timestamp of the data point */
  timestamp: Date;
  /** Trading symbol */
  symbol: string;
  /** Opening price */
  open: number;
  /** Highest price during the period */
  high: number;
  /** Lowest price during the period */
  low: number;
  /** Closing price */
  close: number;
  /** Trading volume */
  volume: number;
  /** Pre-computed features for this data point */
  features?: number[];
  /** Target value for supervised learning */
  target?: number;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Training results and statistics
 */
export interface TrainingResults {
  /** Training performance metrics */
  trainingMetrics: ModelMetrics;
  /** Validation performance metrics */
  validationMetrics: ModelMetrics;
  /** Training duration in milliseconds */
  trainingTime: number;
  /** Number of features used */
  featureCount: number;
  /** Feature importance scores */
  featureImportance: number[];
  /** Cross-validation results */
  crossValidation?: {
    /** Mean CV score */
    meanScore: number;
    /** Standard deviation of CV scores */
    stdScore: number;
    /** Individual fold scores */
    foldScores: number[];
  };
  /** Hyperparameter optimization results */
  hyperparameterOptimization?: {
    /** Best parameters found */
    bestParams: Record<string, unknown>;
    /** Best score achieved */
    bestScore: number;
    /** Number of trials performed */
    trials: number;
  };
}

/**
 * Model configuration for individual algorithms
 */
export interface ModelConfig {
  /** Model type */
  type: 'randomForest' | 'neuralNetwork' | 'svm' | 'gradientBoosting' | 'linear';
  /** Model-specific parameters */
  parameters: Record<string, unknown>;
  /** Weight in ensemble (if applicable) */
  weight?: number;
  /** Whether this model is enabled */
  enabled: boolean;
}

/**
 * Ensemble configuration
 */
export interface EnsembleConfig {
  /** Individual model configurations */
  models: ModelConfig[];
  /** Ensemble combination method */
  combinationMethod: 'average' | 'weighted' | 'stacking' | 'voting';
  /** Meta-learner configuration (for stacking) */
  metaLearner?: ModelConfig;
  /** Dynamic weight adjustment */
  dynamicWeights: boolean;
}

/**
 * Feature importance information
 */
export interface FeatureImportance {
  /** Feature index */
  index: number;
  /** Feature name */
  name: string;
  /** Importance score */
  importance: number;
  /** Rank among all features */
  rank: number;
  /** Feature category */
  category?: string;
}

/**
 * Model serialization format
 */
export interface SerializedModel {
  /** Model version */
  version: string;
  /** Model type */
  type: string;
  /** Serialized model data */
  data: string;
  /** Model metadata */
  metadata: {
    /** Training timestamp */
    trainedAt: string;
    /** Training data size */
    trainingSize: number;
    /** Feature names */
    featureNames: string[];
    /** Performance metrics */
    metrics: ModelMetrics;
  };
  /** Model configuration */
  config: PredictorOptions;
}

/**
 * Real-time prediction configuration
 */
export interface RealtimePredictionConfig {
  /** Update interval in milliseconds */
  updateInterval: number;
  /** Buffer size for streaming data */
  bufferSize: number;
  /** Minimum data points required for prediction */
  minDataPoints: number;
  /** Enable automatic model retraining */
  autoRetrain: boolean;
  /** Retraining trigger conditions */
  retrainTriggers: {
    /** Retrain when accuracy drops below threshold */
    accuracyThreshold?: number;
    /** Retrain after specified time interval */
    timeInterval?: number;
    /** Retrain after specified number of predictions */
    predictionCount?: number;
  };
}

/**
 * Prediction monitoring and alerting
 */
export interface PredictionMonitoring {
  /** Enable monitoring */
  enabled: boolean;
  /** Metrics to monitor */
  metrics: string[];
  /** Alert thresholds */
  thresholds: Record<string, number>;
  /** Alert callback function */
  onAlert?: (alert: PredictionAlert) => void;
}

export interface PredictionAlert {
  /** Alert type */
  type: 'accuracy' | 'confidence' | 'latency' | 'error';
  /** Alert severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Alert message */
  message: string;
  /** Metric value that triggered the alert */
  value: number;
  /** Threshold that was exceeded */
  threshold: number;
  /** Timestamp of the alert */
  timestamp: Date;
}

/**
 * Model comparison results
 */
export interface ModelComparison {
  /** Models being compared */
  models: string[];
  /** Comparison metrics */
  metrics: Record<string, number[]>;
  /** Statistical significance tests */
  significance?: Record<string, {
    pValue: number;
    isSignificant: boolean;
  }>;
  /** Recommended model */
  recommendation: string;
}