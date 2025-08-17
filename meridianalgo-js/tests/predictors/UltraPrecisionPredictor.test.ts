/**
 * Ultra-Precision Predictor Tests
 */

import { UltraPrecisionPredictor } from '../../src/predictors/UltraPrecisionPredictor';
import { MarketData } from '../../src/types/MarketData';

describe('UltraPrecisionPredictor', () => {
  let predictor: UltraPrecisionPredictor;
  let testData: MarketData[];

  beforeEach(() => {
    predictor = new UltraPrecisionPredictor({
      targetErrorRate: 0.05,
      ensembleSize: 3,
      featureCount: 50,
      trainingRatio: 0.8
    });
    
    testData = generateTestMarketData(200, 100);
  });

  describe('Constructor', () => {
    it('should initialize with default options', () => {
      const defaultPredictor = new UltraPrecisionPredictor();
      expect(defaultPredictor).toBeInstanceOf(UltraPrecisionPredictor);
    });

    it('should merge custom options with defaults', () => {
      const customPredictor = new UltraPrecisionPredictor({
        targetErrorRate: 0.02,
        ensembleSize: 5
      });
      expect(customPredictor).toBeInstanceOf(UltraPrecisionPredictor);
    });
  });

  describe('Training', () => {
    it('should train successfully with valid data', async () => {
      const trainingData = testData.map((data, index) => ({
        ...data,
        target: index < testData.length - 1 ? 
          (testData[index + 1].close - data.close) / data.close : 0
      }));

      const results = await predictor.train(trainingData.slice(0, -20));
      
      expect(results).toBeDefined();
      expect(results.trainingMetrics).toBeDefined();
      expect(results.trainingTime).toBeGreaterThan(0);
      expect(predictor.isModelTrained()).toBe(true);
    });

    it('should throw error with insufficient data', async () => {
      const insufficientData = testData.slice(0, 10);
      
      await expect(predictor.train(insufficientData)).rejects.toThrow();
    });

    it('should validate training data', async () => {
      const invalidData = [
        { ...testData[0], close: NaN }
      ];
      
      await expect(predictor.train(invalidData)).rejects.toThrow();
    });
  });

  describe('Prediction', () => {
    beforeEach(async () => {
      const trainingData = testData.map((data, index) => ({
        ...data,
        target: index < testData.length - 1 ? 
          (testData[index + 1].close - data.close) / data.close : 0
      }));

      await predictor.train(trainingData.slice(0, -20));
    });

    it('should make predictions after training', async () => {
      const features = [0.1, 0.2, 0.3, 0.4, 0.5];
      const prediction = await predictor.predict(features);
      
      expect(typeof prediction).toBe('number');
      expect(isFinite(prediction)).toBe(true);
    });

    it('should return confidence score', async () => {
      const features = [0.1, 0.2, 0.3, 0.4, 0.5];
      await predictor.predict(features);
      
      const confidence = predictor.getConfidence();
      expect(confidence).toBeGreaterThanOrEqual(0);
      expect(confidence).toBeLessThanOrEqual(1);
    });

    it('should throw error when predicting without training', async () => {
      const untrainedPredictor = new UltraPrecisionPredictor();
      const features = [0.1, 0.2, 0.3];
      
      await expect(untrainedPredictor.predict(features)).rejects.toThrow();
    });

    it('should validate feature input', async () => {
      const invalidFeatures = [NaN, 0.2, 0.3];
      
      await expect(predictor.predict(invalidFeatures)).rejects.toThrow();
    });
  });

  describe('Batch Prediction', () => {
    beforeEach(async () => {
      const trainingData = testData.map((data, index) => ({
        ...data,
        target: index < testData.length - 1 ? 
          (testData[index + 1].close - data.close) / data.close : 0
      }));

      await predictor.train(trainingData.slice(0, -20));
    });

    it('should make batch predictions', async () => {
      const featuresMatrix = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
      ];
      
      const predictions = await predictor.predictBatch(featuresMatrix);
      
      expect(predictions).toHaveLength(3);
      expect(predictions.every(p => typeof p === 'number' && isFinite(p))).toBe(true);
    });
  });

  describe('Model Persistence', () => {
    beforeEach(async () => {
      const trainingData = testData.map((data, index) => ({
        ...data,
        target: index < testData.length - 1 ? 
          (testData[index + 1].close - data.close) / data.close : 0
      }));

      await predictor.train(trainingData.slice(0, -20));
    });

    it('should save and load model', async () => {
      const modelJson = await predictor.saveModel();
      expect(typeof modelJson).toBe('string');
      
      const newPredictor = new UltraPrecisionPredictor();
      await newPredictor.loadModel(modelJson);
      
      expect(newPredictor.isModelTrained()).toBe(true);
    });

    it('should throw error when saving untrained model', async () => {
      const untrainedPredictor = new UltraPrecisionPredictor();
      
      await expect(untrainedPredictor.saveModel()).rejects.toThrow();
    });
  });

  describe('Feature Importance', () => {
    beforeEach(async () => {
      const trainingData = testData.map((data, index) => ({
        ...data,
        features: [0.1, 0.2, 0.3, 0.4, 0.5],
        target: index < testData.length - 1 ? 
          (testData[index + 1].close - data.close) / data.close : 0
      }));

      await predictor.train(trainingData.slice(0, -20));
    });

    it('should return feature importance scores', () => {
      const importance = predictor.getFeatureImportance();
      
      expect(Array.isArray(importance)).toBe(true);
      expect(importance.length).toBeGreaterThan(0);
      expect(importance.every(score => typeof score === 'number' && isFinite(score))).toBe(true);
    });
  });

  describe('Training Metrics', () => {
    it('should return training metrics after training', async () => {
      const trainingData = testData.map((data, index) => ({
        ...data,
        target: index < testData.length - 1 ? 
          (testData[index + 1].close - data.close) / data.close : 0
      }));

      await predictor.train(trainingData.slice(0, -20));
      
      const metrics = predictor.getTrainingMetrics();
      
      expect(metrics).toBeDefined();
      expect(metrics!.mae).toBeGreaterThanOrEqual(0);
      expect(metrics!.rmse).toBeGreaterThanOrEqual(0);
      expect(metrics!.r2).toBeGreaterThanOrEqual(-1);
      expect(metrics!.directionalAccuracy).toBeGreaterThanOrEqual(0);
      expect(metrics!.directionalAccuracy).toBeLessThanOrEqual(1);
    });

    it('should return null metrics before training', () => {
      const metrics = predictor.getTrainingMetrics();
      expect(metrics).toBeNull();
    });
  });
});