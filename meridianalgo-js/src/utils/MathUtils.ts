/**
 * Mathematical Utilities
 * 
 * Core mathematical functions and utilities for financial calculations.
 */

/**
 * Mathematical utility functions
 */
export class MathUtils {
  /**
   * Calculate the natural logarithm with safety checks
   */
  static safeLog(value: number): number {
    if (value <= 0) {
      throw new Error(`Cannot calculate log of non-positive value: ${value}`);
    }
    return Math.log(value);
  }

  /**
   * Calculate square root with safety checks
   */
  static safeSqrt(value: number): number {
    if (value < 0) {
      throw new Error(`Cannot calculate square root of negative value: ${value}`);
    }
    return Math.sqrt(value);
  }

  /**
   * Calculate percentage change between two values
   */
  static percentageChange(oldValue: number, newValue: number): number {
    if (oldValue === 0) {
      return newValue === 0 ? 0 : Infinity;
    }
    return (newValue - oldValue) / Math.abs(oldValue);
  }

  /**
   * Calculate log returns
   */
  static logReturn(price1: number, price2: number): number {
    if (price1 <= 0 || price2 <= 0) {
      throw new Error('Prices must be positive for log return calculation');
    }
    return Math.log(price2 / price1);
  }

  /**
   * Calculate simple returns
   */
  static simpleReturn(price1: number, price2: number): number {
    if (price1 === 0) {
      throw new Error('Initial price cannot be zero for simple return calculation');
    }
    return (price2 - price1) / price1;
  }

  /**
   * Calculate compound annual growth rate (CAGR)
   */
  static cagr(beginValue: number, endValue: number, periods: number): number {
    if (beginValue <= 0 || endValue <= 0) {
      throw new Error('Values must be positive for CAGR calculation');
    }
    if (periods <= 0) {
      throw new Error('Periods must be positive for CAGR calculation');
    }
    return Math.pow(endValue / beginValue, 1 / periods) - 1;
  }

  /**
   * Calculate annualized return
   */
  static annualizeReturn(totalReturn: number, periods: number, periodsPerYear: number = 252): number {
    return Math.pow(1 + totalReturn, periodsPerYear / periods) - 1;
  }

  /**
   * Calculate annualized volatility
   */
  static annualizeVolatility(volatility: number, periodsPerYear: number = 252): number {
    return volatility * Math.sqrt(periodsPerYear);
  }

  /**
   * Linear interpolation
   */
  static linearInterpolate(x0: number, y0: number, x1: number, y1: number, x: number): number {
    if (x1 === x0) {
      return y0;
    }
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }

  /**
   * Clamp value between min and max
   */
  static clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  /**
   * Check if number is approximately equal (within tolerance)
   */
  static isApproximatelyEqual(a: number, b: number, tolerance: number = 1e-10): boolean {
    return Math.abs(a - b) < tolerance;
  }

  /**
   * Round to specified decimal places
   */
  static roundTo(value: number, decimals: number): number {
    const factor = Math.pow(10, decimals);
    return Math.round(value * factor) / factor;
  }

  /**
   * Calculate factorial
   */
  static factorial(n: number): number {
    if (n < 0 || !Number.isInteger(n)) {
      throw new Error('Factorial is only defined for non-negative integers');
    }
    if (n === 0 || n === 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) {
      result *= i;
    }
    return result;
  }

  /**
   * Calculate combination (n choose k)
   */
  static combination(n: number, k: number): number {
    if (k > n || k < 0 || !Number.isInteger(n) || !Number.isInteger(k)) {
      throw new Error('Invalid parameters for combination calculation');
    }
    if (k === 0 || k === n) return 1;
    
    // Use the more efficient formula: C(n,k) = n! / (k! * (n-k)!)
    // But calculate it iteratively to avoid large factorials
    let result = 1;
    for (let i = 0; i < k; i++) {
      result = result * (n - i) / (i + 1);
    }
    return Math.round(result);
  }

  /**
   * Calculate permutation (n P k)
   */
  static permutation(n: number, k: number): number {
    if (k > n || k < 0 || !Number.isInteger(n) || !Number.isInteger(k)) {
      throw new Error('Invalid parameters for permutation calculation');
    }
    let result = 1;
    for (let i = 0; i < k; i++) {
      result *= (n - i);
    }
    return result;
  }

  /**
   * Calculate greatest common divisor
   */
  static gcd(a: number, b: number): number {
    a = Math.abs(Math.floor(a));
    b = Math.abs(Math.floor(b));
    while (b !== 0) {
      const temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  }

  /**
   * Calculate least common multiple
   */
  static lcm(a: number, b: number): number {
    return Math.abs(a * b) / this.gcd(a, b);
  }

  /**
   * Generate array of numbers from start to end with step
   */
  static range(start: number, end: number, step: number = 1): number[] {
    const result: number[] = [];
    if (step > 0) {
      for (let i = start; i < end; i += step) {
        result.push(i);
      }
    } else if (step < 0) {
      for (let i = start; i > end; i += step) {
        result.push(i);
      }
    }
    return result;
  }

  /**
   * Generate linearly spaced array
   */
  static linspace(start: number, end: number, num: number): number[] {
    if (num <= 0) {
      throw new Error('Number of points must be positive');
    }
    if (num === 1) {
      return [start];
    }
    
    const result: number[] = [];
    const step = (end - start) / (num - 1);
    for (let i = 0; i < num; i++) {
      result.push(start + i * step);
    }
    return result;
  }

  /**
   * Generate logarithmically spaced array
   */
  static logspace(start: number, end: number, num: number, base: number = 10): number[] {
    const linearPoints = this.linspace(start, end, num);
    return linearPoints.map(x => Math.pow(base, x));
  }

  /**
   * Calculate moving average
   */
  static movingAverage(data: number[], window: number): number[] {
    if (window <= 0 || window > data.length) {
      throw new Error('Invalid window size for moving average');
    }
    
    const result: number[] = [];
    for (let i = window - 1; i < data.length; i++) {
      const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / window);
    }
    return result;
  }

  /**
   * Calculate exponential moving average
   */
  static exponentialMovingAverage(data: number[], alpha: number): number[] {
    if (alpha <= 0 || alpha > 1) {
      throw new Error('Alpha must be between 0 and 1 for EMA calculation');
    }
    
    const result: number[] = [];
    let ema = data[0];
    result.push(ema);
    
    for (let i = 1; i < data.length; i++) {
      ema = alpha * data[i] + (1 - alpha) * ema;
      result.push(ema);
    }
    return result;
  }

  /**
   * Calculate weighted moving average
   */
  static weightedMovingAverage(data: number[], weights: number[]): number[] {
    if (weights.length === 0) {
      throw new Error('Weights array cannot be empty');
    }
    
    const window = weights.length;
    const weightSum = weights.reduce((a, b) => a + b, 0);
    
    if (Math.abs(weightSum) < 1e-10) {
      throw new Error('Sum of weights cannot be zero');
    }
    
    const result: number[] = [];
    for (let i = window - 1; i < data.length; i++) {
      let weightedSum = 0;
      for (let j = 0; j < window; j++) {
        weightedSum += data[i - window + 1 + j] * weights[j];
      }
      result.push(weightedSum / weightSum);
    }
    return result;
  }

  /**
   * Calculate rolling correlation
   */
  static rollingCorrelation(x: number[], y: number[], window: number): number[] {
    if (x.length !== y.length) {
      throw new Error('Arrays must have the same length for correlation calculation');
    }
    if (window <= 1 || window > x.length) {
      throw new Error('Invalid window size for rolling correlation');
    }
    
    const result: number[] = [];
    for (let i = window - 1; i < x.length; i++) {
      const xWindow = x.slice(i - window + 1, i + 1);
      const yWindow = y.slice(i - window + 1, i + 1);
      
      const correlation = this.correlation(xWindow, yWindow);
      result.push(correlation);
    }
    return result;
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  static correlation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) {
      throw new Error('Arrays must have the same non-zero length for correlation calculation');
    }
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    if (Math.abs(denominator) < 1e-10) {
      return 0; // No correlation when denominator is zero
    }
    
    return numerator / denominator;
  }

  /**
   * Calculate covariance
   */
  static covariance(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) {
      throw new Error('Arrays must have the same non-zero length for covariance calculation');
    }
    
    const n = x.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    
    let covar = 0;
    for (let i = 0; i < n; i++) {
      covar += (x[i] - meanX) * (y[i] - meanY);
    }
    
    return covar / (n - 1);
  }

  /**
   * Calculate beta coefficient
   */
  static beta(returns: number[], marketReturns: number[]): number {
    const covar = this.covariance(returns, marketReturns);
    const marketVariance = this.variance(marketReturns);
    
    if (Math.abs(marketVariance) < 1e-10) {
      throw new Error('Market variance is zero, cannot calculate beta');
    }
    
    return covar / marketVariance;
  }

  /**
   * Calculate variance
   */
  static variance(data: number[]): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate variance of empty array');
    }
    
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const squaredDiffs = data.map(x => Math.pow(x - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / (data.length - 1);
  }

  /**
   * Normalize array to [0, 1] range
   */
  static normalize(data: number[]): number[] {
    if (data.length === 0) {
      return [];
    }
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    
    if (range === 0) {
      return data.map(() => 0);
    }
    
    return data.map(x => (x - min) / range);
  }

  /**
   * Standardize array (z-score normalization)
   */
  static standardize(data: number[]): number[] {
    if (data.length === 0) {
      return [];
    }
    
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const std = Math.sqrt(this.variance(data));
    
    if (std === 0) {
      return data.map(() => 0);
    }
    
    return data.map(x => (x - mean) / std);
  }
}