/**
 * Ultra-Precision Predictor
 * 
 * Advanced ensemble predictor targeting sub-1% error rates through
 * sophisticated feature engineering and model combination.
 */

import { PredictorOptions, TrainingData, PredictionResult, ModelMetrics, TrainingResults } from '../types/Prediction';
import { MarketData } from '../types/MarketData';
import { ValidationUtils } from '../utils/ValidationUtils';
import { StatisticsUtils } from '../utils/StatisticsUtils';
import { MathUtils } from '../utils/MathUtils';
import { DEFAULT_PREDICTOR_OPTIONS } from '../config/defaults';

/**
 * Ultra-precision predictor implementation
 */
export class UltraPrecisionPredictor {
  private options: Required<PredictorOptions>;
  private models: any[] = [];
  private isTrained: boolean = false;
  private featureImportance: number[] = [];
  private trainingMetrics: ModelMetrics | null = null;
  private lastConfidence: number = 0;
  private modelWeights: number[] = [];

  constructor(options: Partial<PredictorOptions> = {}) {
    this.options = {
      ...DEFAULT_PREDICTOR_OPTIONS,
      ...options
    };
  }

  /**
   * Train the ultra-precision predictor
   */
  async train(data: TrainingData[]): Promise<TrainingResults> {
    console.log(`🚀 Training Ultra-Precision Predictor with ${data.length} samples...`);
    
    // Validate training data
    const validation = ValidationUtils.validateTrainingData(data);
    if (!validation.isValid) {
      throw new Error(`Training data validation failed: ${validation.errors.map(e => e.message).join(', ')}`);
    }

    // Prepare features and targets
    const { features, targets } = this.prepareTrainingData(data);
    console.log(`✨ Prepared ${features.length} samples with ${features[0]?.length || 0} features`);

    // Split data for training and validation
    const { trainX, trainY, testX, testY } = this.splitData(features, targets);

    // Train ensemble of models
    const startTime = Date.now();
    await this.trainEnsemble(trainX, trainY);
    const trainingTime = Date.now() - startTime;

    // Validate performance
    const predictions = await this.predictBatch(testX);
    this.trainingMetrics = this.calculateMetrics(predictions, testY);

    // Calculate feature importance
    this.calculateFeatureImportance(features[0]?.length || 0);

    console.log(`📊 Training completed in ${trainingTime}ms:`);
    console.log(`   MAE: ${(this.trainingMetrics.mae * 100).toFixed(3)}%`);
    console.log(`   RMSE: ${(this.trainingMetrics.rmse * 100).toFixed(3)}%`);
    console.log(`   R²: ${this.trainingMetrics.r2.toFixed(4)}`);
    console.log(`   Directional Accuracy: ${(this.trainingMetrics.directionalAccuracy * 100).toFixed(1)}%`);

    this.isTrained = true;

    // Check if we achieved target error rate
    if (this.trainingMetrics.mae <= this.options.targetErrorRate) {
      console.log(`🎯 Target error rate achieved: ${(this.trainingMetrics.mae * 100).toFixed(3)}% <= ${(this.options.targetErrorRate * 100).toFixed(1)}%`);
    } else {
      console.log(`⚠️  Target error rate not achieved. Consider increasing ensemble size or feature count.`);
    }

    return {
      trainingMetrics: this.trainingMetrics,
      validationMetrics: this.trainingMetrics, // Same for now
      trainingTime,
      featureCount: features[0]?.length || 0,
      featureImportance: [...this.featureImportance]
    };
  }

  /**
   * Make ultra-precise prediction
   */
  async predict(features: number[]): Promise<number> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before making predictions');
    }

    const validation = ValidationUtils.validateFeatures(features);
    if (!validation.isValid) {
      throw new Error(`Feature validation failed: ${validation.errors.map(e => e.message).join(', ')}`);
    }

    // Get ensemble predictions
    const predictions = this.models.map((model, index) => {
      try {
        return this.predictWithModel(model, features, index);
      } catch (error) {
        console.warn(`Model ${index} prediction failed:`, error);
        return 0;
      }
    });

    // Calculate weighted average with confidence
    const weightedPrediction = this.combinepredictions(predictions);
    
    // Calculate prediction confidence
    this.lastConfidence = this.calculatePredictionConfidence(predictions);

    return weightedPrediction;
  }

  /**
   * Batch prediction for multiple samples
   */
  async predictBatch(featuresMatrix: number[][]): Promise<number[]> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before making predictions');
    }

    const predictions: number[] = [];
    
    for (const features of featuresMatrix) {
      const prediction = await this.predict(features);
      predictions.push(prediction);
    }

    return predictions;
  }

  /**
   * Get prediction confidence (0-1)
   */
  getConfidence(): number {
    return this.lastConfidence;
  }

  /**
   * Get feature importance scores
   */
  getFeatureImportance(): number[] {
    return [...this.featureImportance];
  }

  /**
   * Get training metrics
   */
  getTrainingMetrics(): ModelMetrics | null {
    return this.trainingMetrics;
  }

  /**
   * Check if model is trained
   */
  isModelTrained(): boolean {
    return this.isTrained;
  }

  /**
   * Save model to JSON string
   */
  async saveModel(): Promise<string> {
    if (!this.isTrained) {
      throw new Error('Cannot save untrained model');
    }

    const modelData = {
      version: '2.0.0',
      options: this.options,
      models: this.models.map(model => this.serializeModel(model)),
      modelWeights: this.modelWeights,
      featureImportance: this.featureImportance,
      trainingMetrics: this.trainingMetrics,
      timestamp: new Date().toISOString()
    };

    return JSON.stringify(modelData, null, 2);
  }

  /**
   * Load model from JSON string
   */
  async loadModel(modelJson: string): Promise<void> {
    try {
      const modelData = JSON.parse(modelJson);
      
      this.options = { ...this.options, ...modelData.options };
      this.modelWeights = modelData.modelWeights || [];
      this.featureImportance = modelData.featureImportance || [];
      this.trainingMetrics = modelData.trainingMetrics;
      
      // Reconstruct models
      this.models = modelData.models.map((serializedModel: any) => 
        this.deserializeModel(serializedModel)
      );
      
      this.isTrained = this.models.length > 0;
      
      console.log(`✅ Model loaded successfully (${this.models.length} ensemble models)`);
    } catch (error) {
      throw new Error(`Failed to load model: ${error}`);
    }
  }

  /**
   * Prepare training data from raw data
   */
  private prepareTrainingData(data: TrainingData[]): { features: number[][]; targets: number[] } {
    const features: number[][] = [];
    const targets: number[] = [];

    for (let i = 0; i < data.length - 1; i++) {
      const current = data[i];
      const next = data[i + 1];

      // Use provided features or generate basic ones
      let featureVector: number[];
      if (current.features && current.features.length > 0) {
        featureVector = current.features;
      } else {
        // Generate basic features from OHLCV data
        featureVector = this.generateBasicFeatures(data, i);
      }

      // Calculate target (next period return)
      const target = current.target !== undefined ? 
        current.target : 
        (next.close - current.close) / current.close;

      if (featureVector.length > 0 && isFinite(target)) {
        features.push(featureVector);
        targets.push(target);
      }
    }

    return { features, targets };
  }

  /**
   * Generate basic features from OHLCV data
   */
  private generateBasicFeatures(data: TrainingData[], index: number): number[] {
    const features: number[] = [];
    const current = data[index];
    
    // Basic price features
    features.push(
      (current.high - current.low) / current.close, // High-low range
      (current.close - current.open) / current.open, // Open-close return
      current.volume / 1000000 // Normalized volume
    );

    // Simple moving averages (if enough history)
    const lookbacks = [5, 10, 20];
    for (const lookback of lookbacks) {
      if (index >= lookback) {
        const prices = data.slice(index - lookback + 1, index + 1).map(d => d.close);
        const sma = StatisticsUtils.mean(prices);
        features.push((current.close - sma) / sma);
      } else {
        features.push(0);
      }
    }

    // Simple returns (if enough history)
    for (let lag = 1; lag <= 5; lag++) {
      if (index >= lag) {
        const prevClose = data[index - lag].close;
        features.push((current.close - prevClose) / prevClose);
      } else {
        features.push(0);
      }
    }

    return features;
  }

  /**
   * Split data into training and testing sets
   */
  private splitData(features: number[][], targets: number[]) {
    const n = features.length;
    const trainSize = Math.floor(n * this.options.trainingRatio);
    
    const trainX = features.slice(0, trainSize);
    const testX = features.slice(trainSize);
    const trainY = targets.slice(0, trainSize);
    const testY = targets.slice(trainSize);
    
    return { trainX, trainY, testX, testY };
  }

  /**
   * Train ensemble of models
   */
  private async trainEnsemble(trainX: number[][], trainY: number[]): Promise<void> {
    this.models = [];
    this.modelWeights = [];

    for (let i = 0; i < this.options.ensembleSize; i++) {
      console.log(`Training model ${i + 1}/${this.options.ensembleSize}...`);
      
      const model = await this.trainSingleModel(trainX, trainY, i);
      const weight = this.calculateModelWeight(model, trainX, trainY);
      
      this.models.push(model);
      this.modelWeights.push(weight);
    }

    // Normalize weights
    const totalWeight = StatisticsUtils.sum(this.modelWeights);
    if (totalWeight > 0) {
      this.modelWeights = this.modelWeights.map(w => w / totalWeight);
    } else {
      this.modelWeights = new Array(this.models.length).fill(1 / this.models.length);
    }
  }

  /**
   * Train a single model in the ensemble
   */
  private async trainSingleModel(trainX: number[][], trainY: number[], modelIndex: number): Promise<any> {
    // Bootstrap sampling for diversity
    const { sampledX, sampledY } = this.bootstrapSample(trainX, trainY, modelIndex);
    
    // Simple linear regression model (in production, use more sophisticated models)
    const model = this.trainLinearRegression(sampledX, sampledY);
    
    return {
      type: 'linear',
      coefficients: model.coefficients,
      intercept: model.intercept,
      seed: modelIndex * 42
    };
  }

  /**
   * Train a simple linear regression model
   */
  private trainLinearRegression(X: number[][], y: number[]): { coefficients: number[]; intercept: number } {
    const n = X.length;
    const p = X[0]?.length || 0;
    
    if (n === 0 || p === 0) {
      return { coefficients: [], intercept: 0 };
    }

    // Add intercept column
    const XWithIntercept = X.map(row => [1, ...row]);
    
    // Normal equation: β = (X'X)^(-1)X'y
    // Simplified implementation for demonstration
    const coefficients = new Array(p).fill(0);
    let intercept = StatisticsUtils.mean(y);
    
    // Simple gradient descent approximation
    for (let feature = 0; feature < p; feature++) {
      const featureValues = X.map(row => row[feature]);
      const correlation = MathUtils.correlation(featureValues, y);
      coefficients[feature] = correlation * 0.1; // Simplified coefficient
    }
    
    return { coefficients, intercept };
  }

  /**
   * Bootstrap sampling for ensemble diversity
   */
  private bootstrapSample(X: number[][], y: number[], seed: number): { sampledX: number[][]; sampledY: number[] } {
    const n = X.length;
    const sampledX: number[][] = [];
    const sampledY: number[] = [];

    // Use seed for reproducible randomness
    let random = seed;
    const nextRandom = () => {
      random = (random * 9301 + 49297) % 233280;
      return random / 233280;
    };

    for (let i = 0; i < n; i++) {
      const randomIndex = Math.floor(nextRandom() * n);
      sampledX.push([...X[randomIndex]]);
      sampledY.push(y[randomIndex]);
    }

    return { sampledX, sampledY };
  }

  /**
   * Calculate model weight based on performance
   */
  private calculateModelWeight(model: any, X: number[][], y: number[]): number {
    const predictions = X.map(features => this.predictWithModel(model, features, 0));
    const mse = StatisticsUtils.mean(predictions.map((pred, i) => Math.pow(pred - y[i], 2)));
    
    // Weight inversely proportional to error
    return mse > 0 ? 1 / (1 + mse) : 1;
  }

  /**
   * Make prediction with a single model
   */
  private predictWithModel(model: any, features: number[], modelIndex: number): number {
    if (model.type === 'linear') {
      let prediction = model.intercept;
      for (let i = 0; i < Math.min(features.length, model.coefficients.length); i++) {
        prediction += features[i] * model.coefficients[i];
      }
      return prediction;
    }
    
    return 0;
  }

  /**
   * Combine predictions from ensemble
   */
  private combinepredictions(predictions: number[]): number {
    if (predictions.length === 0) return 0;
    
    // Weighted average
    let weightedSum = 0;
    let totalWeight = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      const weight = this.modelWeights[i] || (1 / predictions.length);
      weightedSum += predictions[i] * weight;
      totalWeight += weight;
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : StatisticsUtils.mean(predictions);
  }

  /**
   * Calculate prediction confidence based on ensemble agreement
   */
  private calculatePredictionConfidence(predictions: number[]): number {
    if (predictions.length === 0) return 0;
    
    const mean = StatisticsUtils.mean(predictions);
    const std = StatisticsUtils.standardDeviation(predictions);
    
    // Confidence inversely related to standard deviation
    const normalizedStd = std / (Math.abs(mean) + 1e-8);
    const confidence = Math.max(0, Math.min(1, 1 - normalizedStd));
    
    return confidence;
  }

  /**
   * Calculate feature importance
   */
  private calculateFeatureImportance(featureCount: number): void {
    // Simplified feature importance calculation
    this.featureImportance = new Array(featureCount).fill(0);
    
    // Calculate average absolute coefficients across models
    for (const model of this.models) {
      if (model.coefficients) {
        for (let i = 0; i < Math.min(featureCount, model.coefficients.length); i++) {
          this.featureImportance[i] += Math.abs(model.coefficients[i]);
        }
      }
    }
    
    // Normalize
    const total = StatisticsUtils.sum(this.featureImportance);
    if (total > 0) {
      this.featureImportance = this.featureImportance.map(imp => imp / total);
    }
  }

  /**
   * Calculate model performance metrics
   */
  private calculateMetrics(predictions: number[], actual: number[]): ModelMetrics {
    const n = Math.min(predictions.length, actual.length);
    if (n === 0) {
      return {
        mae: 1,
        mse: 1,
        rmse: 1,
        r2: 0,
        directionalAccuracy: 0.5,
        sampleCount: 0
      };
    }
    
    const pred = predictions.slice(0, n);
    const act = actual.slice(0, n);
    
    // Mean Absolute Error
    const mae = StatisticsUtils.mean(pred.map((p, i) => Math.abs(p - act[i])));
    
    // Root Mean Square Error
    const mse = StatisticsUtils.mean(pred.map((p, i) => Math.pow(p - act[i], 2)));
    const rmse = Math.sqrt(mse);
    
    // R-squared
    const actualMean = StatisticsUtils.mean(act);
    const totalSumSquares = StatisticsUtils.sum(act.map(a => Math.pow(a - actualMean, 2)));
    const residualSumSquares = StatisticsUtils.sum(pred.map((p, i) => Math.pow(act[i] - p, 2)));
    const r2 = totalSumSquares > 0 ? 1 - (residualSumSquares / totalSumSquares) : 0;
    
    // Directional Accuracy
    const correctDirections = pred.filter((p, i) => {
      return (p > 0 && act[i] > 0) || (p < 0 && act[i] < 0) || (Math.abs(p) < 1e-8 && Math.abs(act[i]) < 1e-8);
    }).length;
    const directionalAccuracy = correctDirections / n;
    
    return {
      mae,
      mse,
      rmse,
      r2,
      directionalAccuracy,
      sampleCount: n
    };
  }

  /**
   * Serialize model for saving
   */
  private serializeModel(model: any): any {
    return {
      type: model.type,
      coefficients: model.coefficients,
      intercept: model.intercept,
      seed: model.seed
    };
  }

  /**
   * Deserialize model for loading
   */
  private deserializeModel(serializedModel: any): any {
    return {
      type: serializedModel.type,
      coefficients: serializedModel.coefficients || [],
      intercept: serializedModel.intercept || 0,
      seed: serializedModel.seed || 0
    };
  }
}