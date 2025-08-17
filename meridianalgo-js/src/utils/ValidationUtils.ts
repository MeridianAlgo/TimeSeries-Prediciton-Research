/**
 * Validation Utilities
 * 
 * Comprehensive data validation and error checking utilities.
 */

import { MarketData, ValidationResult, ValidationError, ValidationWarning } from '../types/MarketData';
import { TrainingData } from '../types/Prediction';

/**
 * Validation utility functions
 */
export class ValidationUtils {
  /**
   * Validate market data array
   */
  static validateMarketData(data: MarketData[]): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!Array.isArray(data)) {
      errors.push({
        code: 'INVALID_TYPE',
        message: 'Data must be an array',
        severity: 'critical'
      });
      return { isValid: false, errors, warnings };
    }

    if (data.length === 0) {
      errors.push({
        code: 'EMPTY_DATA',
        message: 'Data array cannot be empty',
        severity: 'critical'
      });
      return { isValid: false, errors, warnings };
    }

    // Validate each data point
    data.forEach((item, index) => {
      this.validateSingleMarketData(item, index, errors, warnings);
    });

    // Check for chronological order
    this.validateChronologicalOrder(data, errors, warnings);

    // Check for data gaps
    this.validateDataGaps(data, warnings);

    // Check for outliers
    this.validateOutliers(data, warnings);

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate single market data point
   */
  private static validateSingleMarketData(
    item: MarketData, 
    index: number, 
    errors: ValidationError[], 
    warnings: ValidationWarning[]
  ): void {
    const prefix = `Item ${index}`;

    // Check required fields
    if (!item.timestamp) {
      errors.push({
        code: 'MISSING_TIMESTAMP',
        message: `${prefix}: Missing timestamp`,
        field: 'timestamp',
        severity: 'high'
      });
    }

    if (!item.symbol || typeof item.symbol !== 'string') {
      errors.push({
        code: 'INVALID_SYMBOL',
        message: `${prefix}: Invalid or missing symbol`,
        field: 'symbol',
        value: item.symbol,
        severity: 'high'
      });
    }

    // Validate numeric fields
    const numericFields = ['open', 'high', 'low', 'close', 'volume'];
    numericFields.forEach(field => {
      const value = (item as any)[field];
      if (typeof value !== 'number' || !isFinite(value)) {
        errors.push({
          code: 'INVALID_NUMERIC_FIELD',
          message: `${prefix}: ${field} must be a finite number`,
          field,
          value,
          severity: 'high'
        });
      } else if (value < 0 && field !== 'close') { // Close can be negative for some instruments
        errors.push({
          code: 'NEGATIVE_VALUE',
          message: `${prefix}: ${field} cannot be negative`,
          field,
          value,
          severity: 'medium'
        });
      }
    });

    // Validate OHLC relationships
    if (typeof item.high === 'number' && typeof item.low === 'number') {
      if (item.high < item.low) {
        errors.push({
          code: 'INVALID_HIGH_LOW',
          message: `${prefix}: High price cannot be less than low price`,
          severity: 'high'
        });
      }
    }

    if (typeof item.open === 'number' && typeof item.high === 'number' && typeof item.low === 'number') {
      if (item.open > item.high || item.open < item.low) {
        warnings.push({
          code: 'OPEN_OUT_OF_RANGE',
          message: `${prefix}: Open price is outside high-low range`,
          field: 'open',
          value: item.open
        });
      }
    }

    if (typeof item.close === 'number' && typeof item.high === 'number' && typeof item.low === 'number') {
      if (item.close > item.high || item.close < item.low) {
        warnings.push({
          code: 'CLOSE_OUT_OF_RANGE',
          message: `${prefix}: Close price is outside high-low range`,
          field: 'close',
          value: item.close
        });
      }
    }

    // Validate optional fields
    if (item.vwap !== undefined) {
      if (typeof item.vwap !== 'number' || !isFinite(item.vwap)) {
        warnings.push({
          code: 'INVALID_VWAP',
          message: `${prefix}: VWAP must be a finite number`,
          field: 'vwap',
          value: item.vwap
        });
      }
    }

    if (item.trades !== undefined) {
      if (!Number.isInteger(item.trades) || item.trades < 0) {
        warnings.push({
          code: 'INVALID_TRADES',
          message: `${prefix}: Trades must be a non-negative integer`,
          field: 'trades',
          value: item.trades
        });
      }
    }
  }

  /**
   * Validate chronological order
   */
  private static validateChronologicalOrder(
    data: MarketData[], 
    errors: ValidationError[], 
    warnings: ValidationWarning[]
  ): void {
    for (let i = 1; i < data.length; i++) {
      const prevTime = new Date(data[i - 1].timestamp).getTime();
      const currTime = new Date(data[i].timestamp).getTime();

      if (currTime < prevTime) {
        errors.push({
          code: 'NON_CHRONOLOGICAL',
          message: `Data is not in chronological order at index ${i}`,
          severity: 'medium'
        });
      } else if (currTime === prevTime) {
        warnings.push({
          code: 'DUPLICATE_TIMESTAMP',
          message: `Duplicate timestamp at index ${i}`,
          value: data[i].timestamp
        });
      }
    }
  }

  /**
   * Validate data gaps
   */
  private static validateDataGaps(data: MarketData[], warnings: ValidationWarning[]): void {
    if (data.length < 2) return;

    const intervals: number[] = [];
    for (let i = 1; i < data.length; i++) {
      const prevTime = new Date(data[i - 1].timestamp).getTime();
      const currTime = new Date(data[i].timestamp).getTime();
      intervals.push(currTime - prevTime);
    }

    // Calculate expected interval (mode of intervals)
    const intervalCounts = new Map<number, number>();
    intervals.forEach(interval => {
      intervalCounts.set(interval, (intervalCounts.get(interval) || 0) + 1);
    });

    let expectedInterval = 0;
    let maxCount = 0;
    for (const [interval, count] of intervalCounts) {
      if (count > maxCount) {
        maxCount = count;
        expectedInterval = interval;
      }
    }

    // Check for gaps larger than 2x expected interval
    intervals.forEach((interval, index) => {
      if (interval > expectedInterval * 2) {
        warnings.push({
          code: 'DATA_GAP',
          message: `Large data gap detected between index ${index} and ${index + 1}`,
          value: interval
        });
      }
    });
  }

  /**
   * Validate outliers in price data
   */
  private static validateOutliers(data: MarketData[], warnings: ValidationWarning[]): void {
    if (data.length < 10) return; // Need sufficient data for outlier detection

    const returns = [];
    for (let i = 1; i < data.length; i++) {
      const prevClose = data[i - 1].close;
      const currClose = data[i].close;
      if (prevClose > 0) {
        returns.push((currClose - prevClose) / prevClose);
      }
    }

    if (returns.length === 0) return;

    // Calculate z-scores for returns
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    const std = Math.sqrt(variance);

    if (std === 0) return;

    returns.forEach((ret, index) => {
      const zScore = Math.abs((ret - mean) / std);
      if (zScore > 3) { // 3-sigma rule
        warnings.push({
          code: 'PRICE_OUTLIER',
          message: `Potential price outlier detected at index ${index + 1}`,
          value: ret
        });
      }
    });
  }

  /**
   * Validate training data
   */
  static validateTrainingData(data: TrainingData[]): ValidationResult {
    // First validate as market data
    const marketDataResult = this.validateMarketData(data);
    
    // Additional validation for training data
    const errors = [...marketDataResult.errors];
    const warnings = [...marketDataResult.warnings];

    data.forEach((item, index) => {
      if (item.features !== undefined) {
        if (!Array.isArray(item.features)) {
          errors.push({
            code: 'INVALID_FEATURES',
            message: `Item ${index}: Features must be an array`,
            field: 'features',
            severity: 'medium'
          });
        } else {
          // Check for invalid feature values
          item.features.forEach((feature, featureIndex) => {
            if (typeof feature !== 'number' || !isFinite(feature)) {
              errors.push({
                code: 'INVALID_FEATURE_VALUE',
                message: `Item ${index}: Feature ${featureIndex} must be a finite number`,
                field: `features[${featureIndex}]`,
                value: feature,
                severity: 'medium'
              });
            }
          });
        }
      }

      if (item.target !== undefined) {
        if (typeof item.target !== 'number' || !isFinite(item.target)) {
          errors.push({
            code: 'INVALID_TARGET',
            message: `Item ${index}: Target must be a finite number`,
            field: 'target',
            value: item.target,
            severity: 'medium'
          });
        }
      }
    });

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate feature array
   */
  static validateFeatures(features: number[]): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!Array.isArray(features)) {
      errors.push({
        code: 'INVALID_TYPE',
        message: 'Features must be an array',
        severity: 'critical'
      });
      return { isValid: false, errors, warnings };
    }

    if (features.length === 0) {
      errors.push({
        code: 'EMPTY_FEATURES',
        message: 'Features array cannot be empty',
        severity: 'high'
      });
      return { isValid: false, errors, warnings };
    }

    features.forEach((feature, index) => {
      if (typeof feature !== 'number') {
        errors.push({
          code: 'INVALID_FEATURE_TYPE',
          message: `Feature ${index} must be a number`,
          field: `features[${index}]`,
          value: feature,
          severity: 'high'
        });
      } else if (!isFinite(feature)) {
        errors.push({
          code: 'INVALID_FEATURE_VALUE',
          message: `Feature ${index} must be finite`,
          field: `features[${index}]`,
          value: feature,
          severity: 'high'
        });
      }
    });

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate portfolio weights
   */
  static validatePortfolioWeights(weights: number[]): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!Array.isArray(weights)) {
      errors.push({
        code: 'INVALID_TYPE',
        message: 'Weights must be an array',
        severity: 'critical'
      });
      return { isValid: false, errors, warnings };
    }

    if (weights.length === 0) {
      errors.push({
        code: 'EMPTY_WEIGHTS',
        message: 'Weights array cannot be empty',
        severity: 'high'
      });
      return { isValid: false, errors, warnings };
    }

    let sum = 0;
    weights.forEach((weight, index) => {
      if (typeof weight !== 'number' || !isFinite(weight)) {
        errors.push({
          code: 'INVALID_WEIGHT_VALUE',
          message: `Weight ${index} must be a finite number`,
          field: `weights[${index}]`,
          value: weight,
          severity: 'high'
        });
      } else {
        sum += weight;
        if (weight < 0) {
          warnings.push({
            code: 'NEGATIVE_WEIGHT',
            message: `Weight ${index} is negative (short position)`,
            field: `weights[${index}]`,
            value: weight
          });
        }
      }
    });

    // Check if weights sum to approximately 1
    const tolerance = 1e-6;
    if (Math.abs(sum - 1) > tolerance) {
      if (Math.abs(sum - 1) > 0.01) {
        errors.push({
          code: 'WEIGHTS_SUM_ERROR',
          message: `Weights sum to ${sum}, should sum to 1.0`,
          severity: 'medium'
        });
      } else {
        warnings.push({
          code: 'WEIGHTS_SUM_WARNING',
          message: `Weights sum to ${sum}, should sum to 1.0`,
          value: sum
        });
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate date range
   */
  static validateDateRange(startDate: Date, endDate: Date): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!(startDate instanceof Date) || isNaN(startDate.getTime())) {
      errors.push({
        code: 'INVALID_START_DATE',
        message: 'Start date must be a valid Date object',
        field: 'startDate',
        severity: 'high'
      });
    }

    if (!(endDate instanceof Date) || isNaN(endDate.getTime())) {
      errors.push({
        code: 'INVALID_END_DATE',
        message: 'End date must be a valid Date object',
        field: 'endDate',
        severity: 'high'
      });
    }

    if (errors.length === 0) {
      if (startDate >= endDate) {
        errors.push({
          code: 'INVALID_DATE_RANGE',
          message: 'Start date must be before end date',
          severity: 'high'
        });
      }

      const now = new Date();
      if (endDate > now) {
        warnings.push({
          code: 'FUTURE_END_DATE',
          message: 'End date is in the future',
          field: 'endDate',
          value: endDate
        });
      }

      const daysDiff = (endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24);
      if (daysDiff < 1) {
        warnings.push({
          code: 'SHORT_DATE_RANGE',
          message: 'Date range is less than 1 day',
          value: daysDiff
        });
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate numeric parameter
   */
  static validateNumericParameter(
    value: unknown, 
    name: string, 
    options: {
      min?: number;
      max?: number;
      integer?: boolean;
      positive?: boolean;
    } = {}
  ): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (typeof value !== 'number') {
      errors.push({
        code: 'INVALID_TYPE',
        message: `${name} must be a number`,
        field: name,
        value,
        severity: 'high'
      });
      return { isValid: false, errors, warnings };
    }

    if (!isFinite(value)) {
      errors.push({
        code: 'INVALID_VALUE',
        message: `${name} must be finite`,
        field: name,
        value,
        severity: 'high'
      });
      return { isValid: false, errors, warnings };
    }

    if (options.integer && !Number.isInteger(value)) {
      errors.push({
        code: 'NOT_INTEGER',
        message: `${name} must be an integer`,
        field: name,
        value,
        severity: 'medium'
      });
    }

    if (options.positive && value <= 0) {
      errors.push({
        code: 'NOT_POSITIVE',
        message: `${name} must be positive`,
        field: name,
        value,
        severity: 'medium'
      });
    }

    if (options.min !== undefined && value < options.min) {
      errors.push({
        code: 'BELOW_MINIMUM',
        message: `${name} must be at least ${options.min}`,
        field: name,
        value,
        severity: 'medium'
      });
    }

    if (options.max !== undefined && value > options.max) {
      errors.push({
        code: 'ABOVE_MAXIMUM',
        message: `${name} must be at most ${options.max}`,
        field: name,
        value,
        severity: 'medium'
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Sanitize and clean market data
   */
  static sanitizeMarketData(data: MarketData[]): MarketData[] {
    return data
      .filter(item => {
        // Remove items with invalid basic structure
        return item && 
               typeof item.open === 'number' && isFinite(item.open) &&
               typeof item.high === 'number' && isFinite(item.high) &&
               typeof item.low === 'number' && isFinite(item.low) &&
               typeof item.close === 'number' && isFinite(item.close) &&
               typeof item.volume === 'number' && isFinite(item.volume) &&
               item.high >= item.low;
      })
      .map(item => ({
        ...item,
        // Ensure timestamp is a Date object
        timestamp: new Date(item.timestamp),
        // Ensure numeric fields are properly typed
        open: Number(item.open),
        high: Number(item.high),
        low: Number(item.low),
        close: Number(item.close),
        volume: Number(item.volume),
        // Clean optional fields
        vwap: item.vwap !== undefined ? Number(item.vwap) : undefined,
        trades: item.trades !== undefined ? Math.floor(Number(item.trades)) : undefined
      }))
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()); // Ensure chronological order
  }
}