/**
 * Statistical Utilities
 * 
 * Advanced statistical functions for financial analysis and risk management.
 */

/**
 * Statistical utility functions
 */
export class StatisticsUtils {
  /**
   * Calculate mean (average)
   */
  static mean(data: number[]): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate mean of empty array');
    }
    return data.reduce((sum, value) => sum + value, 0) / data.length;
  }

  /**
   * Calculate median
   */
  static median(data: number[]): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate median of empty array');
    }
    
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    } else {
      return sorted[mid];
    }
  }

  /**
   * Calculate mode (most frequent value)
   */
  static mode(data: number[]): number[] {
    if (data.length === 0) {
      throw new Error('Cannot calculate mode of empty array');
    }
    
    const frequency: Map<number, number> = new Map();
    let maxFreq = 0;
    
    for (const value of data) {
      const freq = (frequency.get(value) || 0) + 1;
      frequency.set(value, freq);
      maxFreq = Math.max(maxFreq, freq);
    }
    
    const modes: number[] = [];
    for (const [value, freq] of frequency) {
      if (freq === maxFreq) {
        modes.push(value);
      }
    }
    
    return modes;
  }

  /**
   * Calculate standard deviation
   */
  static standardDeviation(data: number[], sample: boolean = true): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate standard deviation of empty array');
    }
    if (sample && data.length === 1) {
      throw new Error('Cannot calculate sample standard deviation with only one data point');
    }
    
    const mean = this.mean(data);
    const squaredDiffs = data.map(x => Math.pow(x - mean, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / (sample ? data.length - 1 : data.length);
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate variance
   */
  static variance(data: number[], sample: boolean = true): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate variance of empty array');
    }
    if (sample && data.length === 1) {
      throw new Error('Cannot calculate sample variance with only one data point');
    }
    
    const mean = this.mean(data);
    const squaredDiffs = data.map(x => Math.pow(x - mean, 2));
    
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / (sample ? data.length - 1 : data.length);
  }

  /**
   * Calculate skewness (measure of asymmetry)
   */
  static skewness(data: number[]): number {
    if (data.length < 3) {
      throw new Error('Need at least 3 data points to calculate skewness');
    }
    
    const mean = this.mean(data);
    const std = this.standardDeviation(data);
    const n = data.length;
    
    if (std === 0) {
      return 0;
    }
    
    const cubedDeviations = data.map(x => Math.pow((x - mean) / std, 3));
    const sum = cubedDeviations.reduce((a, b) => a + b, 0);
    
    return (n / ((n - 1) * (n - 2))) * sum;
  }

  /**
   * Calculate kurtosis (measure of tail heaviness)
   */
  static kurtosis(data: number[], excess: boolean = true): number {
    if (data.length < 4) {
      throw new Error('Need at least 4 data points to calculate kurtosis');
    }
    
    const mean = this.mean(data);
    const std = this.standardDeviation(data);
    const n = data.length;
    
    if (std === 0) {
      return excess ? -3 : 0;
    }
    
    const fourthMoments = data.map(x => Math.pow((x - mean) / std, 4));
    const sum = fourthMoments.reduce((a, b) => a + b, 0);
    
    const kurtosisValue = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum - 
                         (3 * Math.pow(n - 1, 2) / ((n - 2) * (n - 3)));
    
    return excess ? kurtosisValue : kurtosisValue + 3;
  }

  /**
   * Calculate quantile (percentile)
   */
  static quantile(data: number[], q: number): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate quantile of empty array');
    }
    if (q < 0 || q > 1) {
      throw new Error('Quantile must be between 0 and 1');
    }
    
    const sorted = [...data].sort((a, b) => a - b);
    const index = q * (sorted.length - 1);
    
    if (Number.isInteger(index)) {
      return sorted[index];
    } else {
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      const weight = index - lower;
      return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
  }

  /**
   * Calculate interquartile range (IQR)
   */
  static interquartileRange(data: number[]): number {
    const q1 = this.quantile(data, 0.25);
    const q3 = this.quantile(data, 0.75);
    return q3 - q1;
  }

  /**
   * Detect outliers using IQR method
   */
  static detectOutliers(data: number[], multiplier: number = 1.5): { outliers: number[]; indices: number[] } {
    const q1 = this.quantile(data, 0.25);
    const q3 = this.quantile(data, 0.75);
    const iqr = q3 - q1;
    const lowerBound = q1 - multiplier * iqr;
    const upperBound = q3 + multiplier * iqr;
    
    const outliers: number[] = [];
    const indices: number[] = [];
    
    data.forEach((value, index) => {
      if (value < lowerBound || value > upperBound) {
        outliers.push(value);
        indices.push(index);
      }
    });
    
    return { outliers, indices };
  }

  /**
   * Calculate z-scores
   */
  static zScores(data: number[]): number[] {
    const mean = this.mean(data);
    const std = this.standardDeviation(data);
    
    if (std === 0) {
      return data.map(() => 0);
    }
    
    return data.map(x => (x - mean) / std);
  }

  /**
   * Calculate rolling statistics
   */
  static rollingStatistic(
    data: number[], 
    window: number, 
    statistic: 'mean' | 'std' | 'var' | 'min' | 'max' | 'median' | 'skewness' | 'kurtosis'
  ): number[] {
    if (window <= 0 || window > data.length) {
      throw new Error('Invalid window size');
    }
    
    const result: number[] = [];
    
    for (let i = window - 1; i < data.length; i++) {
      const windowData = data.slice(i - window + 1, i + 1);
      
      switch (statistic) {
        case 'mean':
          result.push(this.mean(windowData));
          break;
        case 'std':
          result.push(this.standardDeviation(windowData));
          break;
        case 'var':
          result.push(this.variance(windowData));
          break;
        case 'min':
          result.push(Math.min(...windowData));
          break;
        case 'max':
          result.push(Math.max(...windowData));
          break;
        case 'median':
          result.push(this.median(windowData));
          break;
        case 'skewness':
          result.push(windowData.length >= 3 ? this.skewness(windowData) : 0);
          break;
        case 'kurtosis':
          result.push(windowData.length >= 4 ? this.kurtosis(windowData) : 0);
          break;
        default:
          throw new Error(`Unknown statistic: ${statistic}`);
      }
    }
    
    return result;
  }

  /**
   * Calculate Value at Risk (VaR)
   */
  static valueAtRisk(returns: number[], confidenceLevel: number = 0.95): number {
    if (confidenceLevel <= 0 || confidenceLevel >= 1) {
      throw new Error('Confidence level must be between 0 and 1');
    }
    
    return -this.quantile(returns, 1 - confidenceLevel);
  }

  /**
   * Calculate Expected Shortfall (Conditional VaR)
   */
  static expectedShortfall(returns: number[], confidenceLevel: number = 0.95): number {
    const varValue = this.valueAtRisk(returns, confidenceLevel);
    const tailReturns = returns.filter(r => r <= -varValue);
    
    if (tailReturns.length === 0) {
      return varValue;
    }
    
    return -this.mean(tailReturns);
  }

  /**
   * Calculate maximum drawdown
   */
  static maxDrawdown(cumulativeReturns: number[]): { maxDrawdown: number; peak: number; trough: number } {
    if (cumulativeReturns.length === 0) {
      throw new Error('Cannot calculate max drawdown of empty array');
    }
    
    let peak = cumulativeReturns[0];
    let maxDrawdown = 0;
    let peakIndex = 0;
    let troughIndex = 0;
    
    for (let i = 1; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] > peak) {
        peak = cumulativeReturns[i];
        peakIndex = i;
      }
      
      const drawdown = (peak - cumulativeReturns[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        troughIndex = i;
      }
    }
    
    return {
      maxDrawdown,
      peak: peakIndex,
      trough: troughIndex
    };
  }

  /**
   * Calculate Sharpe ratio
   */
  static sharpeRatio(returns: number[], riskFreeRate: number = 0): number {
    const excessReturns = returns.map(r => r - riskFreeRate);
    const meanExcessReturn = this.mean(excessReturns);
    const std = this.standardDeviation(excessReturns);
    
    if (std === 0) {
      return meanExcessReturn > 0 ? Infinity : (meanExcessReturn < 0 ? -Infinity : 0);
    }
    
    return meanExcessReturn / std;
  }

  /**
   * Calculate Sortino ratio
   */
  static sortinoRatio(returns: number[], riskFreeRate: number = 0, targetReturn: number = 0): number {
    const excessReturns = returns.map(r => r - riskFreeRate);
    const meanExcessReturn = this.mean(excessReturns);
    
    const downsideReturns = returns.filter(r => r < targetReturn);
    if (downsideReturns.length === 0) {
      return meanExcessReturn > 0 ? Infinity : 0;
    }
    
    const downsideDeviation = Math.sqrt(
      downsideReturns.reduce((sum, r) => sum + Math.pow(r - targetReturn, 2), 0) / downsideReturns.length
    );
    
    if (downsideDeviation === 0) {
      return meanExcessReturn > 0 ? Infinity : 0;
    }
    
    return meanExcessReturn / downsideDeviation;
  }

  /**
   * Calculate Calmar ratio
   */
  static calmarRatio(returns: number[]): number {
    const cumulativeReturns = this.cumulativeSum(returns);
    const totalReturn = cumulativeReturns[cumulativeReturns.length - 1];
    const { maxDrawdown } = this.maxDrawdown(cumulativeReturns);
    
    if (maxDrawdown === 0) {
      return totalReturn > 0 ? Infinity : 0;
    }
    
    return totalReturn / maxDrawdown;
  }

  /**
   * Calculate cumulative sum
   */
  static cumulativeSum(data: number[]): number[] {
    const result: number[] = [];
    let sum = 0;
    
    for (const value of data) {
      sum += value;
      result.push(sum);
    }
    
    return result;
  }

  /**
   * Calculate cumulative product
   */
  static cumulativeProduct(data: number[]): number[] {
    const result: number[] = [];
    let product = 1;
    
    for (const value of data) {
      product *= (1 + value);
      result.push(product - 1);
    }
    
    return result;
  }

  /**
   * Calculate sum
   */
  static sum(data: number[]): number {
    return data.reduce((sum, value) => sum + value, 0);
  }

  /**
   * Calculate product
   */
  static product(data: number[]): number {
    return data.reduce((product, value) => product * value, 1);
  }

  /**
   * Calculate range (max - min)
   */
  static range(data: number[]): number {
    if (data.length === 0) {
      throw new Error('Cannot calculate range of empty array');
    }
    return Math.max(...data) - Math.min(...data);
  }

  /**
   * Calculate coefficient of variation
   */
  static coefficientOfVariation(data: number[]): number {
    const mean = this.mean(data);
    const std = this.standardDeviation(data);
    
    if (mean === 0) {
      throw new Error('Cannot calculate coefficient of variation when mean is zero');
    }
    
    return std / Math.abs(mean);
  }

  /**
   * Perform Jarque-Bera test for normality
   */
  static jarqueBeraTest(data: number[]): { statistic: number; pValue: number; isNormal: boolean } {
    if (data.length < 4) {
      throw new Error('Need at least 4 data points for Jarque-Bera test');
    }
    
    const n = data.length;
    const skew = this.skewness(data);
    const kurt = this.kurtosis(data, true); // excess kurtosis
    
    const jb = (n / 6) * (Math.pow(skew, 2) + Math.pow(kurt, 2) / 4);
    
    // Approximate p-value using chi-square distribution with 2 degrees of freedom
    // This is a simplified approximation
    const pValue = 1 - this.chiSquareCDF(jb, 2);
    
    return {
      statistic: jb,
      pValue,
      isNormal: pValue > 0.05 // 5% significance level
    };
  }

  /**
   * Approximate chi-square CDF (simplified implementation)
   */
  private static chiSquareCDF(x: number, df: number): number {
    if (x <= 0) return 0;
    if (df === 2) {
      return 1 - Math.exp(-x / 2);
    }
    // For other degrees of freedom, use a simple approximation
    // In a production environment, you'd want a more accurate implementation
    return Math.min(1, x / (2 * df));
  }

  /**
   * Calculate autocorrelation at given lag
   */
  static autocorrelation(data: number[], lag: number): number {
    if (lag >= data.length || lag < 0) {
      throw new Error('Invalid lag for autocorrelation calculation');
    }
    
    const n = data.length - lag;
    const x1 = data.slice(0, n);
    const x2 = data.slice(lag, lag + n);
    
    const mean1 = this.mean(x1);
    const mean2 = this.mean(x2);
    
    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;
    
    for (let i = 0; i < n; i++) {
      const diff1 = x1[i] - mean1;
      const diff2 = x2[i] - mean2;
      numerator += diff1 * diff2;
      denominator1 += diff1 * diff1;
      denominator2 += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(denominator1 * denominator2);
    
    if (denominator === 0) {
      return 0;
    }
    
    return numerator / denominator;
  }

  /**
   * Calculate multiple autocorrelations
   */
  static autocorrelationFunction(data: number[], maxLag: number): number[] {
    const result: number[] = [];
    
    for (let lag = 0; lag <= maxLag; lag++) {
      if (lag === 0) {
        result.push(1); // Autocorrelation at lag 0 is always 1
      } else {
        result.push(this.autocorrelation(data, lag));
      }
    }
    
    return result;
  }
}