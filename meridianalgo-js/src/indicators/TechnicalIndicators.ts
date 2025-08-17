/**
 * Technical Indicators
 * 
 * Comprehensive collection of technical analysis indicators for financial markets.
 */

import { MarketData } from '../types/MarketData';
import { MACDResult, BollingerBandsResult, StochasticResult } from '../types/Features';
import { MathUtils } from '../utils/MathUtils';
import { StatisticsUtils } from '../utils/StatisticsUtils';

/**
 * Technical indicator calculations
 */
export class TechnicalIndicators {
  /**
   * Simple Moving Average (SMA)
   */
  static sma(data: number[], period: number): number[] {
    if (period <= 0 || period > data.length) {
      throw new Error('Invalid period for SMA calculation');
    }
    
    return MathUtils.movingAverage(data, period);
  }

  /**
   * Exponential Moving Average (EMA)
   */
  static ema(data: number[], period: number): number[] {
    if (period <= 0) {
      throw new Error('Period must be positive for EMA calculation');
    }
    
    const alpha = 2 / (period + 1);
    return MathUtils.exponentialMovingAverage(data, alpha);
  }

  /**
   * Weighted Moving Average (WMA)
   */
  static wma(data: number[], period: number): number[] {
    if (period <= 0 || period > data.length) {
      throw new Error('Invalid period for WMA calculation');
    }
    
    const weights = Array.from({ length: period }, (_, i) => i + 1);
    return MathUtils.weightedMovingAverage(data, weights);
  }

  /**
   * Relative Strength Index (RSI)
   */
  static rsi(data: number[], period: number = 14): number[] {
    if (period <= 0 || data.length < period + 1) {
      throw new Error('Insufficient data or invalid period for RSI calculation');
    }
    
    const changes: number[] = [];
    for (let i = 1; i < data.length; i++) {
      changes.push(data[i] - data[i - 1]);
    }
    
    const gains = changes.map(change => Math.max(change, 0));
    const losses = changes.map(change => Math.max(-change, 0));
    
    const avgGains = this.sma(gains, period);
    const avgLosses = this.sma(losses, period);
    
    const rsiValues: number[] = [];
    for (let i = 0; i < avgGains.length; i++) {
      if (avgLosses[i] === 0) {
        rsiValues.push(100);
      } else {
        const rs = avgGains[i] / avgLosses[i];
        rsiValues.push(100 - (100 / (1 + rs)));
      }
    }
    
    return rsiValues;
  }

  /**
   * Moving Average Convergence Divergence (MACD)
   */
  static macd(data: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): MACDResult {
    if (fastPeriod >= slowPeriod) {
      throw new Error('Fast period must be less than slow period for MACD');
    }
    
    const fastEMA = this.ema(data, fastPeriod);
    const slowEMA = this.ema(data, slowPeriod);
    
    // Align arrays (slowEMA is shorter)
    const startIndex = slowPeriod - fastPeriod;
    const alignedFastEMA = fastEMA.slice(startIndex);
    
    const macdLine = alignedFastEMA.map((fast, i) => fast - slowEMA[i]);
    const signalLine = this.ema(macdLine, signalPeriod);
    
    // Align MACD line with signal line
    const alignedMACDLine = macdLine.slice(macdLine.length - signalLine.length);
    const histogram = alignedMACDLine.map((macd, i) => macd - signalLine[i]);
    
    return {
      macd: alignedMACDLine,
      signal: signalLine,
      histogram
    };
  }

  /**
   * Bollinger Bands
   */
  static bollingerBands(data: number[], period: number = 20, multiplier: number = 2): BollingerBandsResult {
    if (period <= 0 || period > data.length) {
      throw new Error('Invalid period for Bollinger Bands calculation');
    }
    
    const smaValues = this.sma(data, period);
    const upper: number[] = [];
    const lower: number[] = [];
    const bandwidth: number[] = [];
    const percentB: number[] = [];
    
    for (let i = 0; i < smaValues.length; i++) {
      const dataSlice = data.slice(i, i + period);
      const std = StatisticsUtils.standardDeviation(dataSlice);
      
      const upperBand = smaValues[i] + multiplier * std;
      const lowerBand = smaValues[i] - multiplier * std;
      
      upper.push(upperBand);
      lower.push(lowerBand);
      bandwidth.push((upperBand - lowerBand) / smaValues[i]);
      
      const currentPrice = data[i + period - 1];
      percentB.push((currentPrice - lowerBand) / (upperBand - lowerBand));
    }
    
    return {
      upper,
      middle: smaValues,
      lower,
      bandwidth,
      percentB
    };
  }

  /**
   * Stochastic Oscillator
   */
  static stochastic(high: number[], low: number[], close: number[], kPeriod: number = 14, dPeriod: number = 3): StochasticResult {
    if (high.length !== low.length || low.length !== close.length) {
      throw new Error('High, low, and close arrays must have the same length');
    }
    
    const k: number[] = [];
    
    for (let i = kPeriod - 1; i < close.length; i++) {
      const highestHigh = Math.max(...high.slice(i - kPeriod + 1, i + 1));
      const lowestLow = Math.min(...low.slice(i - kPeriod + 1, i + 1));
      
      if (highestHigh === lowestLow) {
        k.push(50); // Avoid division by zero
      } else {
        k.push(((close[i] - lowestLow) / (highestHigh - lowestLow)) * 100);
      }
    }
    
    const d = this.sma(k, dPeriod);
    
    return {
      k: k.slice(k.length - d.length), // Align with D values
      d
    };
  }

  /**
   * Williams %R
   */
  static williamsR(high: number[], low: number[], close: number[], period: number = 14): number[] {
    if (high.length !== low.length || low.length !== close.length) {
      throw new Error('High, low, and close arrays must have the same length');
    }
    
    const williamsR: number[] = [];
    
    for (let i = period - 1; i < close.length; i++) {
      const highestHigh = Math.max(...high.slice(i - period + 1, i + 1));
      const lowestLow = Math.min(...low.slice(i - period + 1, i + 1));
      
      if (highestHigh === lowestLow) {
        williamsR.push(-50); // Avoid division by zero
      } else {
        williamsR.push(((highestHigh - close[i]) / (highestHigh - lowestLow)) * -100);
      }
    }
    
    return williamsR;
  }

  /**
   * Commodity Channel Index (CCI)
   */
  static cci(high: number[], low: number[], close: number[], period: number = 20): number[] {
    if (high.length !== low.length || low.length !== close.length) {
      throw new Error('High, low, and close arrays must have the same length');
    }
    
    // Calculate Typical Price
    const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
    
    const cci: number[] = [];
    
    for (let i = period - 1; i < typicalPrice.length; i++) {
      const tpSlice = typicalPrice.slice(i - period + 1, i + 1);
      const smaTP = StatisticsUtils.mean(tpSlice);
      
      // Calculate Mean Deviation
      const meanDeviation = tpSlice.reduce((sum, tp) => sum + Math.abs(tp - smaTP), 0) / period;
      
      if (meanDeviation === 0) {
        cci.push(0);
      } else {
        cci.push((typicalPrice[i] - smaTP) / (0.015 * meanDeviation));
      }
    }
    
    return cci;
  }

  /**
   * Average True Range (ATR)
   */
  static atr(high: number[], low: number[], close: number[], period: number = 14): number[] {
    if (high.length !== low.length || low.length !== close.length) {
      throw new Error('High, low, and close arrays must have the same length');
    }
    
    const trueRanges: number[] = [];
    
    for (let i = 1; i < high.length; i++) {
      const tr1 = high[i] - low[i];
      const tr2 = Math.abs(high[i] - close[i - 1]);
      const tr3 = Math.abs(low[i] - close[i - 1]);
      
      trueRanges.push(Math.max(tr1, tr2, tr3));
    }
    
    return this.sma(trueRanges, period);
  }

  /**
   * Average Directional Index (ADX)
   */
  static adx(high: number[], low: number[], close: number[], period: number = 14): { adx: number[]; plusDI: number[]; minusDI: number[] } {
    if (high.length !== low.length || low.length !== close.length) {
      throw new Error('High, low, and close arrays must have the same length');
    }
    
    const plusDM: number[] = [];
    const minusDM: number[] = [];
    const trueRanges: number[] = [];
    
    for (let i = 1; i < high.length; i++) {
      const highDiff = high[i] - high[i - 1];
      const lowDiff = low[i - 1] - low[i];
      
      plusDM.push(highDiff > lowDiff && highDiff > 0 ? highDiff : 0);
      minusDM.push(lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0);
      
      const tr1 = high[i] - low[i];
      const tr2 = Math.abs(high[i] - close[i - 1]);
      const tr3 = Math.abs(low[i] - close[i - 1]);
      trueRanges.push(Math.max(tr1, tr2, tr3));
    }
    
    const smoothedPlusDM = this.sma(plusDM, period);
    const smoothedMinusDM = this.sma(minusDM, period);
    const smoothedTR = this.sma(trueRanges, period);
    
    const plusDI = smoothedPlusDM.map((dm, i) => (dm / smoothedTR[i]) * 100);
    const minusDI = smoothedMinusDM.map((dm, i) => (dm / smoothedTR[i]) * 100);
    
    const dx = plusDI.map((plus, i) => {
      const sum = plus + minusDI[i];
      return sum === 0 ? 0 : (Math.abs(plus - minusDI[i]) / sum) * 100;
    });
    
    const adx = this.sma(dx, period);
    
    return {
      adx,
      plusDI: plusDI.slice(plusDI.length - adx.length),
      minusDI: minusDI.slice(minusDI.length - adx.length)
    };
  }

  /**
   * Money Flow Index (MFI)
   */
  static mfi(high: number[], low: number[], close: number[], volume: number[], period: number = 14): number[] {
    if (high.length !== low.length || low.length !== close.length || close.length !== volume.length) {
      throw new Error('All arrays must have the same length');
    }
    
    const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
    const rawMoneyFlow = typicalPrice.map((tp, i) => tp * volume[i]);
    
    const mfi: number[] = [];
    
    for (let i = period; i < typicalPrice.length; i++) {
      let positiveFlow = 0;
      let negativeFlow = 0;
      
      for (let j = i - period + 1; j <= i; j++) {
        if (typicalPrice[j] > typicalPrice[j - 1]) {
          positiveFlow += rawMoneyFlow[j];
        } else if (typicalPrice[j] < typicalPrice[j - 1]) {
          negativeFlow += rawMoneyFlow[j];
        }
      }
      
      if (negativeFlow === 0) {
        mfi.push(100);
      } else {
        const moneyRatio = positiveFlow / negativeFlow;
        mfi.push(100 - (100 / (1 + moneyRatio)));
      }
    }
    
    return mfi;
  }

  /**
   * On-Balance Volume (OBV)
   */
  static obv(close: number[], volume: number[]): number[] {
    if (close.length !== volume.length) {
      throw new Error('Close and volume arrays must have the same length');
    }
    
    const obv: number[] = [volume[0]];
    
    for (let i = 1; i < close.length; i++) {
      if (close[i] > close[i - 1]) {
        obv.push(obv[i - 1] + volume[i]);
      } else if (close[i] < close[i - 1]) {
        obv.push(obv[i - 1] - volume[i]);
      } else {
        obv.push(obv[i - 1]);
      }
    }
    
    return obv;
  }

  /**
   * Volume Weighted Average Price (VWAP)
   */
  static vwap(high: number[], low: number[], close: number[], volume: number[]): number[] {
    if (high.length !== low.length || low.length !== close.length || close.length !== volume.length) {
      throw new Error('All arrays must have the same length');
    }
    
    const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
    const vwap: number[] = [];
    
    let cumulativeTPV = 0; // Cumulative Typical Price * Volume
    let cumulativeVolume = 0;
    
    for (let i = 0; i < typicalPrice.length; i++) {
      cumulativeTPV += typicalPrice[i] * volume[i];
      cumulativeVolume += volume[i];
      
      vwap.push(cumulativeVolume === 0 ? typicalPrice[i] : cumulativeTPV / cumulativeVolume);
    }
    
    return vwap;
  }

  /**
   * Momentum
   */
  static momentum(data: number[], period: number = 10): number[] {
    if (period <= 0 || period >= data.length) {
      throw new Error('Invalid period for momentum calculation');
    }
    
    const momentum: number[] = [];
    
    for (let i = period; i < data.length; i++) {
      momentum.push(data[i] - data[i - period]);
    }
    
    return momentum;
  }

  /**
   * Rate of Change (ROC)
   */
  static roc(data: number[], period: number = 10): number[] {
    if (period <= 0 || period >= data.length) {
      throw new Error('Invalid period for ROC calculation');
    }
    
    const roc: number[] = [];
    
    for (let i = period; i < data.length; i++) {
      if (data[i - period] === 0) {
        roc.push(0);
      } else {
        roc.push(((data[i] - data[i - period]) / data[i - period]) * 100);
      }
    }
    
    return roc;
  }

  /**
   * Standard Deviation
   */
  static standardDeviation(data: number[], period: number): number[] {
    if (period <= 0 || period > data.length) {
      throw new Error('Invalid period for standard deviation calculation');
    }
    
    return StatisticsUtils.rollingStatistic(data, period, 'std');
  }

  /**
   * Variance
   */
  static variance(data: number[], period: number): number[] {
    if (period <= 0 || period > data.length) {
      throw new Error('Invalid period for variance calculation');
    }
    
    return StatisticsUtils.rollingStatistic(data, period, 'var');
  }

  /**
   * Linear Regression Slope
   */
  static linearRegressionSlope(data: number[], period: number): number[] {
    if (period <= 1 || period > data.length) {
      throw new Error('Invalid period for linear regression slope calculation');
    }
    
    const slopes: number[] = [];
    
    for (let i = period - 1; i < data.length; i++) {
      const y = data.slice(i - period + 1, i + 1);
      const x = Array.from({ length: period }, (_, idx) => idx);
      
      const n = period;
      const sumX = x.reduce((sum, val) => sum + val, 0);
      const sumY = y.reduce((sum, val) => sum + val, 0);
      const sumXY = x.reduce((sum, val, idx) => sum + val * y[idx], 0);
      const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
      
      const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
      slopes.push(slope);
    }
    
    return slopes;
  }

  /**
   * Pivot Points (Standard)
   */
  static pivotPoints(high: number, low: number, close: number): {
    pivot: number;
    r1: number;
    r2: number;
    r3: number;
    s1: number;
    s2: number;
    s3: number;
  } {
    const pivot = (high + low + close) / 3;
    
    return {
      pivot,
      r1: 2 * pivot - low,
      r2: pivot + (high - low),
      r3: high + 2 * (pivot - low),
      s1: 2 * pivot - high,
      s2: pivot - (high - low),
      s3: low - 2 * (high - pivot)
    };
  }

  /**
   * Helper method to extract prices from MarketData array
   */
  static extractPrices(data: MarketData[], priceType: 'open' | 'high' | 'low' | 'close' | 'volume'): number[] {
    return data.map(item => item[priceType]);
  }

  /**
   * Helper method to extract OHLCV arrays from MarketData
   */
  static extractOHLCV(data: MarketData[]): {
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
  } {
    return {
      open: data.map(item => item.open),
      high: data.map(item => item.high),
      low: data.map(item => item.low),
      close: data.map(item => item.close),
      volume: data.map(item => item.volume)
    };
  }
}