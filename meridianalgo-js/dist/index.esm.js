/**
 * Validation Utilities
 *
 * Comprehensive data validation and error checking utilities.
 */
/**
 * Validation utility functions
 */
class ValidationUtils {
    /**
     * Validate market data array
     */
    static validateMarketData(data) {
        const errors = [];
        const warnings = [];
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
    static validateSingleMarketData(item, index, errors, warnings) {
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
            const value = item[field];
            if (typeof value !== 'number' || !isFinite(value)) {
                errors.push({
                    code: 'INVALID_NUMERIC_FIELD',
                    message: `${prefix}: ${field} must be a finite number`,
                    field,
                    value,
                    severity: 'high'
                });
            }
            else if (value < 0 && field !== 'close') { // Close can be negative for some instruments
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
    static validateChronologicalOrder(data, errors, warnings) {
        for (let i = 1; i < data.length; i++) {
            const prevTime = new Date(data[i - 1].timestamp).getTime();
            const currTime = new Date(data[i].timestamp).getTime();
            if (currTime < prevTime) {
                errors.push({
                    code: 'NON_CHRONOLOGICAL',
                    message: `Data is not in chronological order at index ${i}`,
                    severity: 'medium'
                });
            }
            else if (currTime === prevTime) {
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
    static validateDataGaps(data, warnings) {
        if (data.length < 2)
            return;
        const intervals = [];
        for (let i = 1; i < data.length; i++) {
            const prevTime = new Date(data[i - 1].timestamp).getTime();
            const currTime = new Date(data[i].timestamp).getTime();
            intervals.push(currTime - prevTime);
        }
        // Calculate expected interval (mode of intervals)
        const intervalCounts = new Map();
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
    static validateOutliers(data, warnings) {
        if (data.length < 10)
            return; // Need sufficient data for outlier detection
        const returns = [];
        for (let i = 1; i < data.length; i++) {
            const prevClose = data[i - 1].close;
            const currClose = data[i].close;
            if (prevClose > 0) {
                returns.push((currClose - prevClose) / prevClose);
            }
        }
        if (returns.length === 0)
            return;
        // Calculate z-scores for returns
        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
        const std = Math.sqrt(variance);
        if (std === 0)
            return;
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
    static validateTrainingData(data) {
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
                }
                else {
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
    static validateFeatures(features) {
        const errors = [];
        const warnings = [];
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
            }
            else if (!isFinite(feature)) {
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
    static validatePortfolioWeights(weights) {
        const errors = [];
        const warnings = [];
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
            }
            else {
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
            }
            else {
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
    static validateDateRange(startDate, endDate) {
        const errors = [];
        const warnings = [];
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
    static validateNumericParameter(value, name, options = {}) {
        const errors = [];
        const warnings = [];
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
    static sanitizeMarketData(data) {
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

/**
 * Statistical Utilities
 *
 * Advanced statistical functions for financial analysis and risk management.
 */
/**
 * Statistical utility functions
 */
class StatisticsUtils {
    /**
     * Calculate mean (average)
     */
    static mean(data) {
        if (data.length === 0) {
            throw new Error('Cannot calculate mean of empty array');
        }
        return data.reduce((sum, value) => sum + value, 0) / data.length;
    }
    /**
     * Calculate median
     */
    static median(data) {
        if (data.length === 0) {
            throw new Error('Cannot calculate median of empty array');
        }
        const sorted = [...data].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2;
        }
        else {
            return sorted[mid];
        }
    }
    /**
     * Calculate mode (most frequent value)
     */
    static mode(data) {
        if (data.length === 0) {
            throw new Error('Cannot calculate mode of empty array');
        }
        const frequency = new Map();
        let maxFreq = 0;
        for (const value of data) {
            const freq = (frequency.get(value) || 0) + 1;
            frequency.set(value, freq);
            maxFreq = Math.max(maxFreq, freq);
        }
        const modes = [];
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
    static standardDeviation(data, sample = true) {
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
    static variance(data, sample = true) {
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
    static skewness(data) {
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
    static kurtosis(data, excess = true) {
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
    static quantile(data, q) {
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
        }
        else {
            const lower = Math.floor(index);
            const upper = Math.ceil(index);
            const weight = index - lower;
            return sorted[lower] * (1 - weight) + sorted[upper] * weight;
        }
    }
    /**
     * Calculate interquartile range (IQR)
     */
    static interquartileRange(data) {
        const q1 = this.quantile(data, 0.25);
        const q3 = this.quantile(data, 0.75);
        return q3 - q1;
    }
    /**
     * Detect outliers using IQR method
     */
    static detectOutliers(data, multiplier = 1.5) {
        const q1 = this.quantile(data, 0.25);
        const q3 = this.quantile(data, 0.75);
        const iqr = q3 - q1;
        const lowerBound = q1 - multiplier * iqr;
        const upperBound = q3 + multiplier * iqr;
        const outliers = [];
        const indices = [];
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
    static zScores(data) {
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
    static rollingStatistic(data, window, statistic) {
        if (window <= 0 || window > data.length) {
            throw new Error('Invalid window size');
        }
        const result = [];
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
    static valueAtRisk(returns, confidenceLevel = 0.95) {
        if (confidenceLevel <= 0 || confidenceLevel >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }
        return -this.quantile(returns, 1 - confidenceLevel);
    }
    /**
     * Calculate Expected Shortfall (Conditional VaR)
     */
    static expectedShortfall(returns, confidenceLevel = 0.95) {
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
    static maxDrawdown(cumulativeReturns) {
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
    static sharpeRatio(returns, riskFreeRate = 0) {
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
    static sortinoRatio(returns, riskFreeRate = 0, targetReturn = 0) {
        const excessReturns = returns.map(r => r - riskFreeRate);
        const meanExcessReturn = this.mean(excessReturns);
        const downsideReturns = returns.filter(r => r < targetReturn);
        if (downsideReturns.length === 0) {
            return meanExcessReturn > 0 ? Infinity : 0;
        }
        const downsideDeviation = Math.sqrt(downsideReturns.reduce((sum, r) => sum + Math.pow(r - targetReturn, 2), 0) / downsideReturns.length);
        if (downsideDeviation === 0) {
            return meanExcessReturn > 0 ? Infinity : 0;
        }
        return meanExcessReturn / downsideDeviation;
    }
    /**
     * Calculate Calmar ratio
     */
    static calmarRatio(returns) {
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
    static cumulativeSum(data) {
        const result = [];
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
    static cumulativeProduct(data) {
        const result = [];
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
    static sum(data) {
        return data.reduce((sum, value) => sum + value, 0);
    }
    /**
     * Calculate product
     */
    static product(data) {
        return data.reduce((product, value) => product * value, 1);
    }
    /**
     * Calculate range (max - min)
     */
    static range(data) {
        if (data.length === 0) {
            throw new Error('Cannot calculate range of empty array');
        }
        return Math.max(...data) - Math.min(...data);
    }
    /**
     * Calculate coefficient of variation
     */
    static coefficientOfVariation(data) {
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
    static jarqueBeraTest(data) {
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
    static chiSquareCDF(x, df) {
        if (x <= 0)
            return 0;
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
    static autocorrelation(data, lag) {
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
    static autocorrelationFunction(data, maxLag) {
        const result = [];
        for (let lag = 0; lag <= maxLag; lag++) {
            if (lag === 0) {
                result.push(1); // Autocorrelation at lag 0 is always 1
            }
            else {
                result.push(this.autocorrelation(data, lag));
            }
        }
        return result;
    }
}

/**
 * Mathematical Utilities
 *
 * Core mathematical functions and utilities for financial calculations.
 */
/**
 * Mathematical utility functions
 */
class MathUtils {
    /**
     * Calculate the natural logarithm with safety checks
     */
    static safeLog(value) {
        if (value <= 0) {
            throw new Error(`Cannot calculate log of non-positive value: ${value}`);
        }
        return Math.log(value);
    }
    /**
     * Calculate square root with safety checks
     */
    static safeSqrt(value) {
        if (value < 0) {
            throw new Error(`Cannot calculate square root of negative value: ${value}`);
        }
        return Math.sqrt(value);
    }
    /**
     * Calculate percentage change between two values
     */
    static percentageChange(oldValue, newValue) {
        if (oldValue === 0) {
            return newValue === 0 ? 0 : Infinity;
        }
        return (newValue - oldValue) / Math.abs(oldValue);
    }
    /**
     * Calculate log returns
     */
    static logReturn(price1, price2) {
        if (price1 <= 0 || price2 <= 0) {
            throw new Error('Prices must be positive for log return calculation');
        }
        return Math.log(price2 / price1);
    }
    /**
     * Calculate simple returns
     */
    static simpleReturn(price1, price2) {
        if (price1 === 0) {
            throw new Error('Initial price cannot be zero for simple return calculation');
        }
        return (price2 - price1) / price1;
    }
    /**
     * Calculate compound annual growth rate (CAGR)
     */
    static cagr(beginValue, endValue, periods) {
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
    static annualizeReturn(totalReturn, periods, periodsPerYear = 252) {
        return Math.pow(1 + totalReturn, periodsPerYear / periods) - 1;
    }
    /**
     * Calculate annualized volatility
     */
    static annualizeVolatility(volatility, periodsPerYear = 252) {
        return volatility * Math.sqrt(periodsPerYear);
    }
    /**
     * Linear interpolation
     */
    static linearInterpolate(x0, y0, x1, y1, x) {
        if (x1 === x0) {
            return y0;
        }
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
    /**
     * Clamp value between min and max
     */
    static clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }
    /**
     * Check if number is approximately equal (within tolerance)
     */
    static isApproximatelyEqual(a, b, tolerance = 1e-10) {
        return Math.abs(a - b) < tolerance;
    }
    /**
     * Round to specified decimal places
     */
    static roundTo(value, decimals) {
        const factor = Math.pow(10, decimals);
        return Math.round(value * factor) / factor;
    }
    /**
     * Calculate factorial
     */
    static factorial(n) {
        if (n < 0 || !Number.isInteger(n)) {
            throw new Error('Factorial is only defined for non-negative integers');
        }
        if (n === 0 || n === 1)
            return 1;
        let result = 1;
        for (let i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    /**
     * Calculate combination (n choose k)
     */
    static combination(n, k) {
        if (k > n || k < 0 || !Number.isInteger(n) || !Number.isInteger(k)) {
            throw new Error('Invalid parameters for combination calculation');
        }
        if (k === 0 || k === n)
            return 1;
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
    static permutation(n, k) {
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
    static gcd(a, b) {
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
    static lcm(a, b) {
        return Math.abs(a * b) / this.gcd(a, b);
    }
    /**
     * Generate array of numbers from start to end with step
     */
    static range(start, end, step = 1) {
        const result = [];
        if (step > 0) {
            for (let i = start; i < end; i += step) {
                result.push(i);
            }
        }
        else if (step < 0) {
            for (let i = start; i > end; i += step) {
                result.push(i);
            }
        }
        return result;
    }
    /**
     * Generate linearly spaced array
     */
    static linspace(start, end, num) {
        if (num <= 0) {
            throw new Error('Number of points must be positive');
        }
        if (num === 1) {
            return [start];
        }
        const result = [];
        const step = (end - start) / (num - 1);
        for (let i = 0; i < num; i++) {
            result.push(start + i * step);
        }
        return result;
    }
    /**
     * Generate logarithmically spaced array
     */
    static logspace(start, end, num, base = 10) {
        const linearPoints = this.linspace(start, end, num);
        return linearPoints.map(x => Math.pow(base, x));
    }
    /**
     * Calculate moving average
     */
    static movingAverage(data, window) {
        if (window <= 0 || window > data.length) {
            throw new Error('Invalid window size for moving average');
        }
        const result = [];
        for (let i = window - 1; i < data.length; i++) {
            const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / window);
        }
        return result;
    }
    /**
     * Calculate exponential moving average
     */
    static exponentialMovingAverage(data, alpha) {
        if (alpha <= 0 || alpha > 1) {
            throw new Error('Alpha must be between 0 and 1 for EMA calculation');
        }
        const result = [];
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
    static weightedMovingAverage(data, weights) {
        if (weights.length === 0) {
            throw new Error('Weights array cannot be empty');
        }
        const window = weights.length;
        const weightSum = weights.reduce((a, b) => a + b, 0);
        if (Math.abs(weightSum) < 1e-10) {
            throw new Error('Sum of weights cannot be zero');
        }
        const result = [];
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
    static rollingCorrelation(x, y, window) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have the same length for correlation calculation');
        }
        if (window <= 1 || window > x.length) {
            throw new Error('Invalid window size for rolling correlation');
        }
        const result = [];
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
    static correlation(x, y) {
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
    static covariance(x, y) {
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
    static beta(returns, marketReturns) {
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
    static variance(data) {
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
    static normalize(data) {
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
    static standardize(data) {
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

/**
 * Default Configuration
 *
 * Default settings and constants for the MeridianAlgo-JS library.
 */
/**
 * Default predictor configuration
 */
const DEFAULT_PREDICTOR_OPTIONS = {
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
const DEFAULT_FEATURE_OPTIONS = {
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
const DEFAULT_OPTIMIZER_OPTIONS = {
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
const DEFAULT_CONFIG = {
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
 * Ultra-Precision Predictor
 *
 * Advanced ensemble predictor targeting sub-1% error rates through
 * sophisticated feature engineering and model combination.
 */
/**
 * Ultra-precision predictor implementation
 */
class UltraPrecisionPredictor {
    constructor(options = {}) {
        this.models = [];
        this.isTrained = false;
        this.featureImportance = [];
        this.trainingMetrics = null;
        this.lastConfidence = 0;
        this.modelWeights = [];
        this.options = {
            ...DEFAULT_PREDICTOR_OPTIONS,
            ...options
        };
    }
    /**
     * Train the ultra-precision predictor
     */
    async train(data) {
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
        }
        else {
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
    async predict(features) {
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
            }
            catch (error) {
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
    async predictBatch(featuresMatrix) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before making predictions');
        }
        const predictions = [];
        for (const features of featuresMatrix) {
            const prediction = await this.predict(features);
            predictions.push(prediction);
        }
        return predictions;
    }
    /**
     * Get prediction confidence (0-1)
     */
    getConfidence() {
        return this.lastConfidence;
    }
    /**
     * Get feature importance scores
     */
    getFeatureImportance() {
        return [...this.featureImportance];
    }
    /**
     * Get training metrics
     */
    getTrainingMetrics() {
        return this.trainingMetrics;
    }
    /**
     * Check if model is trained
     */
    isModelTrained() {
        return this.isTrained;
    }
    /**
     * Save model to JSON string
     */
    async saveModel() {
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
    async loadModel(modelJson) {
        try {
            const modelData = JSON.parse(modelJson);
            this.options = { ...this.options, ...modelData.options };
            this.modelWeights = modelData.modelWeights || [];
            this.featureImportance = modelData.featureImportance || [];
            this.trainingMetrics = modelData.trainingMetrics;
            // Reconstruct models
            this.models = modelData.models.map((serializedModel) => this.deserializeModel(serializedModel));
            this.isTrained = this.models.length > 0;
            console.log(`✅ Model loaded successfully (${this.models.length} ensemble models)`);
        }
        catch (error) {
            throw new Error(`Failed to load model: ${error}`);
        }
    }
    /**
     * Prepare training data from raw data
     */
    prepareTrainingData(data) {
        const features = [];
        const targets = [];
        for (let i = 0; i < data.length - 1; i++) {
            const current = data[i];
            const next = data[i + 1];
            // Use provided features or generate basic ones
            let featureVector;
            if (current.features && current.features.length > 0) {
                featureVector = current.features;
            }
            else {
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
    generateBasicFeatures(data, index) {
        const features = [];
        const current = data[index];
        // Basic price features
        features.push((current.high - current.low) / current.close, // High-low range
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
            }
            else {
                features.push(0);
            }
        }
        // Simple returns (if enough history)
        for (let lag = 1; lag <= 5; lag++) {
            if (index >= lag) {
                const prevClose = data[index - lag].close;
                features.push((current.close - prevClose) / prevClose);
            }
            else {
                features.push(0);
            }
        }
        return features;
    }
    /**
     * Split data into training and testing sets
     */
    splitData(features, targets) {
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
    async trainEnsemble(trainX, trainY) {
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
        }
        else {
            this.modelWeights = new Array(this.models.length).fill(1 / this.models.length);
        }
    }
    /**
     * Train a single model in the ensemble
     */
    async trainSingleModel(trainX, trainY, modelIndex) {
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
    trainLinearRegression(X, y) {
        const n = X.length;
        const p = X[0]?.length || 0;
        if (n === 0 || p === 0) {
            return { coefficients: [], intercept: 0 };
        }
        // Add intercept column
        X.map(row => [1, ...row]);
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
    bootstrapSample(X, y, seed) {
        const n = X.length;
        const sampledX = [];
        const sampledY = [];
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
    calculateModelWeight(model, X, y) {
        const predictions = X.map(features => this.predictWithModel(model, features, 0));
        const mse = StatisticsUtils.mean(predictions.map((pred, i) => Math.pow(pred - y[i], 2)));
        // Weight inversely proportional to error
        return mse > 0 ? 1 / (1 + mse) : 1;
    }
    /**
     * Make prediction with a single model
     */
    predictWithModel(model, features, modelIndex) {
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
    combinepredictions(predictions) {
        if (predictions.length === 0)
            return 0;
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
    calculatePredictionConfidence(predictions) {
        if (predictions.length === 0)
            return 0;
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
    calculateFeatureImportance(featureCount) {
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
    calculateMetrics(predictions, actual) {
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
    serializeModel(model) {
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
    deserializeModel(serializedModel) {
        return {
            type: serializedModel.type,
            coefficients: serializedModel.coefficients || [],
            intercept: serializedModel.intercept || 0,
            seed: serializedModel.seed || 0
        };
    }
}

/**
 * Technical Indicators
 *
 * Comprehensive collection of technical analysis indicators for financial markets.
 */
/**
 * Technical indicator calculations
 */
class TechnicalIndicators {
    /**
     * Simple Moving Average (SMA)
     */
    static sma(data, period) {
        if (period <= 0 || period > data.length) {
            throw new Error('Invalid period for SMA calculation');
        }
        return MathUtils.movingAverage(data, period);
    }
    /**
     * Exponential Moving Average (EMA)
     */
    static ema(data, period) {
        if (period <= 0) {
            throw new Error('Period must be positive for EMA calculation');
        }
        const alpha = 2 / (period + 1);
        return MathUtils.exponentialMovingAverage(data, alpha);
    }
    /**
     * Weighted Moving Average (WMA)
     */
    static wma(data, period) {
        if (period <= 0 || period > data.length) {
            throw new Error('Invalid period for WMA calculation');
        }
        const weights = Array.from({ length: period }, (_, i) => i + 1);
        return MathUtils.weightedMovingAverage(data, weights);
    }
    /**
     * Relative Strength Index (RSI)
     */
    static rsi(data, period = 14) {
        if (period <= 0 || data.length < period + 1) {
            throw new Error('Insufficient data or invalid period for RSI calculation');
        }
        const changes = [];
        for (let i = 1; i < data.length; i++) {
            changes.push(data[i] - data[i - 1]);
        }
        const gains = changes.map(change => Math.max(change, 0));
        const losses = changes.map(change => Math.max(-change, 0));
        const avgGains = this.sma(gains, period);
        const avgLosses = this.sma(losses, period);
        const rsiValues = [];
        for (let i = 0; i < avgGains.length; i++) {
            if (avgLosses[i] === 0) {
                rsiValues.push(100);
            }
            else {
                const rs = avgGains[i] / avgLosses[i];
                rsiValues.push(100 - (100 / (1 + rs)));
            }
        }
        return rsiValues;
    }
    /**
     * Moving Average Convergence Divergence (MACD)
     */
    static macd(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
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
    static bollingerBands(data, period = 20, multiplier = 2) {
        if (period <= 0 || period > data.length) {
            throw new Error('Invalid period for Bollinger Bands calculation');
        }
        const smaValues = this.sma(data, period);
        const upper = [];
        const lower = [];
        const bandwidth = [];
        const percentB = [];
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
    static stochastic(high, low, close, kPeriod = 14, dPeriod = 3) {
        if (high.length !== low.length || low.length !== close.length) {
            throw new Error('High, low, and close arrays must have the same length');
        }
        const k = [];
        for (let i = kPeriod - 1; i < close.length; i++) {
            const highestHigh = Math.max(...high.slice(i - kPeriod + 1, i + 1));
            const lowestLow = Math.min(...low.slice(i - kPeriod + 1, i + 1));
            if (highestHigh === lowestLow) {
                k.push(50); // Avoid division by zero
            }
            else {
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
    static williamsR(high, low, close, period = 14) {
        if (high.length !== low.length || low.length !== close.length) {
            throw new Error('High, low, and close arrays must have the same length');
        }
        const williamsR = [];
        for (let i = period - 1; i < close.length; i++) {
            const highestHigh = Math.max(...high.slice(i - period + 1, i + 1));
            const lowestLow = Math.min(...low.slice(i - period + 1, i + 1));
            if (highestHigh === lowestLow) {
                williamsR.push(-50); // Avoid division by zero
            }
            else {
                williamsR.push(((highestHigh - close[i]) / (highestHigh - lowestLow)) * -100);
            }
        }
        return williamsR;
    }
    /**
     * Commodity Channel Index (CCI)
     */
    static cci(high, low, close, period = 20) {
        if (high.length !== low.length || low.length !== close.length) {
            throw new Error('High, low, and close arrays must have the same length');
        }
        // Calculate Typical Price
        const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
        const cci = [];
        for (let i = period - 1; i < typicalPrice.length; i++) {
            const tpSlice = typicalPrice.slice(i - period + 1, i + 1);
            const smaTP = StatisticsUtils.mean(tpSlice);
            // Calculate Mean Deviation
            const meanDeviation = tpSlice.reduce((sum, tp) => sum + Math.abs(tp - smaTP), 0) / period;
            if (meanDeviation === 0) {
                cci.push(0);
            }
            else {
                cci.push((typicalPrice[i] - smaTP) / (0.015 * meanDeviation));
            }
        }
        return cci;
    }
    /**
     * Average True Range (ATR)
     */
    static atr(high, low, close, period = 14) {
        if (high.length !== low.length || low.length !== close.length) {
            throw new Error('High, low, and close arrays must have the same length');
        }
        const trueRanges = [];
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
    static adx(high, low, close, period = 14) {
        if (high.length !== low.length || low.length !== close.length) {
            throw new Error('High, low, and close arrays must have the same length');
        }
        const plusDM = [];
        const minusDM = [];
        const trueRanges = [];
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
    static mfi(high, low, close, volume, period = 14) {
        if (high.length !== low.length || low.length !== close.length || close.length !== volume.length) {
            throw new Error('All arrays must have the same length');
        }
        const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
        const rawMoneyFlow = typicalPrice.map((tp, i) => tp * volume[i]);
        const mfi = [];
        for (let i = period; i < typicalPrice.length; i++) {
            let positiveFlow = 0;
            let negativeFlow = 0;
            for (let j = i - period + 1; j <= i; j++) {
                if (typicalPrice[j] > typicalPrice[j - 1]) {
                    positiveFlow += rawMoneyFlow[j];
                }
                else if (typicalPrice[j] < typicalPrice[j - 1]) {
                    negativeFlow += rawMoneyFlow[j];
                }
            }
            if (negativeFlow === 0) {
                mfi.push(100);
            }
            else {
                const moneyRatio = positiveFlow / negativeFlow;
                mfi.push(100 - (100 / (1 + moneyRatio)));
            }
        }
        return mfi;
    }
    /**
     * On-Balance Volume (OBV)
     */
    static obv(close, volume) {
        if (close.length !== volume.length) {
            throw new Error('Close and volume arrays must have the same length');
        }
        const obv = [volume[0]];
        for (let i = 1; i < close.length; i++) {
            if (close[i] > close[i - 1]) {
                obv.push(obv[i - 1] + volume[i]);
            }
            else if (close[i] < close[i - 1]) {
                obv.push(obv[i - 1] - volume[i]);
            }
            else {
                obv.push(obv[i - 1]);
            }
        }
        return obv;
    }
    /**
     * Volume Weighted Average Price (VWAP)
     */
    static vwap(high, low, close, volume) {
        if (high.length !== low.length || low.length !== close.length || close.length !== volume.length) {
            throw new Error('All arrays must have the same length');
        }
        const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
        const vwap = [];
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
    static momentum(data, period = 10) {
        if (period <= 0 || period >= data.length) {
            throw new Error('Invalid period for momentum calculation');
        }
        const momentum = [];
        for (let i = period; i < data.length; i++) {
            momentum.push(data[i] - data[i - period]);
        }
        return momentum;
    }
    /**
     * Rate of Change (ROC)
     */
    static roc(data, period = 10) {
        if (period <= 0 || period >= data.length) {
            throw new Error('Invalid period for ROC calculation');
        }
        const roc = [];
        for (let i = period; i < data.length; i++) {
            if (data[i - period] === 0) {
                roc.push(0);
            }
            else {
                roc.push(((data[i] - data[i - period]) / data[i - period]) * 100);
            }
        }
        return roc;
    }
    /**
     * Standard Deviation
     */
    static standardDeviation(data, period) {
        if (period <= 0 || period > data.length) {
            throw new Error('Invalid period for standard deviation calculation');
        }
        return StatisticsUtils.rollingStatistic(data, period, 'std');
    }
    /**
     * Variance
     */
    static variance(data, period) {
        if (period <= 0 || period > data.length) {
            throw new Error('Invalid period for variance calculation');
        }
        return StatisticsUtils.rollingStatistic(data, period, 'var');
    }
    /**
     * Linear Regression Slope
     */
    static linearRegressionSlope(data, period) {
        if (period <= 1 || period > data.length) {
            throw new Error('Invalid period for linear regression slope calculation');
        }
        const slopes = [];
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
    static pivotPoints(high, low, close) {
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
    static extractPrices(data, priceType) {
        return data.map(item => item[priceType]);
    }
    /**
     * Helper method to extract OHLCV arrays from MarketData
     */
    static extractOHLCV(data) {
        return {
            open: data.map(item => item.open),
            high: data.map(item => item.high),
            low: data.map(item => item.low),
            close: data.map(item => item.close),
            volume: data.map(item => item.volume)
        };
    }
}

/**
 * Advanced Feature Engineer
 *
 * Generates 1000+ sophisticated features from basic OHLCV market data
 * using advanced technical analysis, statistical methods, and machine learning.
 */
/**
 * Advanced feature engineering implementation
 */
class FeatureEngineer {
    constructor(options = {}) {
        this.featureNames = [];
        this.featureMetadata = [];
        this.options = {
            ...DEFAULT_FEATURE_OPTIONS,
            ...options
        };
    }
    /**
     * Generate comprehensive feature matrix from market data
     */
    generateFeatures(data) {
        if (data.length < 50) {
            throw new Error('Insufficient data for feature generation. Need at least 50 periods.');
        }
        console.log(`🔧 Generating advanced features from ${data.length} data points...`);
        this.featureNames = [];
        this.featureMetadata = [];
        const features = [];
        // Initialize feature matrix
        for (let i = 0; i < data.length; i++) {
            features.push([]);
        }
        // 1. Basic OHLCV Features
        this.addBasicFeatures(data, features);
        // 2. Technical Indicators
        this.addTechnicalIndicators(data, features);
        // 3. Statistical Features
        if (this.options.enableStatisticalFeatures) {
            this.addStatisticalFeatures(data, features);
        }
        // 4. Volatility Features
        if (this.options.enableVolatilityFeatures) {
            this.addVolatilityFeatures(data, features);
        }
        // 5. Cross-sectional Features
        this.addCrossSectionalFeatures(data, features);
        // 6. Pattern Recognition Features
        this.addPatternFeatures(data, features);
        // 7. Harmonic Features
        if (this.options.enableHarmonicFeatures) {
            this.addHarmonicFeatures(data, features);
        }
        console.log(`✨ Generated ${this.featureNames.length} features`);
        return {
            data: features,
            featureNames: [...this.featureNames],
            metadata: [...this.featureMetadata],
            columns: this.featureNames.length,
            rows: features.length
        };
    }
    /**
     * Get feature names
     */
    getFeatureNames() {
        return [...this.featureNames];
    }
    /**
     * Get feature metadata
     */
    getFeatureMetadata() {
        return [...this.featureMetadata];
    }
    /**
     * Add basic OHLCV-derived features
     */
    addBasicFeatures(data, features) {
        const closes = data.map(d => d.close);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        const opens = data.map(d => d.open);
        const volumes = data.map(d => d.volume);
        // Price-based features
        const returns = this.calculateReturns(closes);
        const logReturns = this.calculateLogReturns(closes);
        const hlRatio = highs.map((h, i) => h / lows[i]);
        const ocRatio = opens.map((o, i) => o / closes[i]);
        const bodySize = opens.map((o, i) => Math.abs(closes[i] - o) / o);
        const upperShadow = highs.map((h, i) => (h - Math.max(opens[i], closes[i])) / closes[i]);
        const lowerShadow = lows.map((l, i) => (Math.min(opens[i], closes[i]) - l) / closes[i]);
        this.addFeatureColumn(features, returns, 'returns', 'Basic price returns');
        this.addFeatureColumn(features, logReturns, 'log_returns', 'Logarithmic returns');
        this.addFeatureColumn(features, hlRatio, 'hl_ratio', 'High/Low ratio');
        this.addFeatureColumn(features, ocRatio, 'oc_ratio', 'Open/Close ratio');
        this.addFeatureColumn(features, bodySize, 'body_size', 'Candle body size');
        this.addFeatureColumn(features, upperShadow, 'upper_shadow', 'Upper shadow size');
        this.addFeatureColumn(features, lowerShadow, 'lower_shadow', 'Lower shadow size');
        // Volume-based features
        const volumeReturns = this.calculateReturns(volumes);
        const priceVolumeCorr = this.calculateRollingCorrelation(returns, volumeReturns, 20);
        const volumeMA = this.calculateMovingAverage(volumes, 20);
        const volumeRatio = volumes.map((v, i) => i >= 20 ? v / volumeMA[i - 20] : 1);
        this.addFeatureColumn(features, volumeReturns, 'volume_returns', 'Volume returns');
        this.addFeatureColumn(features, priceVolumeCorr, 'price_volume_corr', 'Price-volume correlation');
        this.addFeatureColumn(features, volumeRatio, 'volume_ratio', 'Volume ratio to MA');
    }
    /**
     * Add technical indicator features
     */
    addTechnicalIndicators(data, features) {
        const closes = data.map(d => d.close);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        const volumes = data.map(d => d.volume);
        // RSI with multiple periods
        for (const period of this.options.technicalIndicators.rsi.periods) {
            const rsi = TechnicalIndicators.rsi(closes, period);
            const rsiVelocity = this.calculateVelocity(rsi);
            const rsiAcceleration = this.calculateVelocity(rsiVelocity);
            this.addFeatureColumn(features, rsi, `rsi_${period}`, `RSI with period ${period}`);
            this.addFeatureColumn(features, rsiVelocity, `rsi_velocity_${period}`, `RSI velocity ${period}`);
            this.addFeatureColumn(features, rsiAcceleration, `rsi_acceleration_${period}`, `RSI acceleration ${period}`);
        }
        // MACD
        const macd = TechnicalIndicators.macd(closes, this.options.technicalIndicators.macd.fast, this.options.technicalIndicators.macd.slow, this.options.technicalIndicators.macd.signal);
        this.addFeatureColumn(features, macd.macd, 'macd_line', 'MACD line');
        this.addFeatureColumn(features, macd.signal, 'macd_signal', 'MACD signal');
        this.addFeatureColumn(features, macd.histogram, 'macd_histogram', 'MACD histogram');
        // Bollinger Bands
        const bb = TechnicalIndicators.bollingerBands(closes, this.options.technicalIndicators.bollinger.period, this.options.technicalIndicators.bollinger.multiplier);
        this.addFeatureColumn(features, bb.upper, 'bb_upper', 'Bollinger upper band');
        this.addFeatureColumn(features, bb.middle, 'bb_middle', 'Bollinger middle band');
        this.addFeatureColumn(features, bb.lower, 'bb_lower', 'Bollinger lower band');
        this.addFeatureColumn(features, bb.percentB, 'bb_position', 'Bollinger band position');
        this.addFeatureColumn(features, bb.bandwidth, 'bb_width', 'Bollinger band width');
        // Stochastic Oscillator
        const stoch = TechnicalIndicators.stochastic(highs, lows, closes, this.options.technicalIndicators.stochastic.kPeriod, this.options.technicalIndicators.stochastic.dPeriod);
        this.addFeatureColumn(features, stoch.k, 'stoch_k', 'Stochastic %K');
        this.addFeatureColumn(features, stoch.d, 'stoch_d', 'Stochastic %D');
        // Williams %R
        const williams = TechnicalIndicators.williamsR(highs, lows, closes, this.options.technicalIndicators.williams.period);
        this.addFeatureColumn(features, williams, 'williams_r', 'Williams %R');
        // Commodity Channel Index
        const cci = TechnicalIndicators.cci(highs, lows, closes, this.options.technicalIndicators.cci.period);
        this.addFeatureColumn(features, cci, 'cci', 'Commodity Channel Index');
        // ATR
        const atr = TechnicalIndicators.atr(highs, lows, closes, 14);
        this.addFeatureColumn(features, atr, 'atr', 'Average True Range');
        // ADX
        const adx = TechnicalIndicators.adx(highs, lows, closes, 14);
        this.addFeatureColumn(features, adx.adx, 'adx', 'Average Directional Index');
        this.addFeatureColumn(features, adx.plusDI, 'plus_di', 'Plus Directional Indicator');
        this.addFeatureColumn(features, adx.minusDI, 'minus_di', 'Minus Directional Indicator');
        // Volume indicators
        const obv = TechnicalIndicators.obv(closes, volumes);
        const mfi = TechnicalIndicators.mfi(highs, lows, closes, volumes, 14);
        this.addFeatureColumn(features, obv, 'obv', 'On-Balance Volume');
        this.addFeatureColumn(features, mfi, 'mfi', 'Money Flow Index');
    }
    /**
     * Add statistical features
     */
    addStatisticalFeatures(data, features) {
        const closes = data.map(d => d.close);
        const returns = this.calculateReturns(closes);
        // Rolling statistics for different windows
        for (const window of this.options.lookbackPeriods) {
            if (window <= data.length) {
                // Rolling mean
                const rollingMean = StatisticsUtils.rollingStatistic(returns, window, 'mean');
                this.addFeatureColumn(features, rollingMean, `rolling_mean_${window}`, `Rolling mean ${window}`);
                // Rolling standard deviation
                const rollingStd = StatisticsUtils.rollingStatistic(returns, window, 'std');
                this.addFeatureColumn(features, rollingStd, `rolling_std_${window}`, `Rolling std ${window}`);
                // Rolling skewness
                const rollingSkew = StatisticsUtils.rollingStatistic(returns, window, 'skewness');
                this.addFeatureColumn(features, rollingSkew, `rolling_skew_${window}`, `Rolling skewness ${window}`);
                // Rolling kurtosis
                const rollingKurt = StatisticsUtils.rollingStatistic(returns, window, 'kurtosis');
                this.addFeatureColumn(features, rollingKurt, `rolling_kurt_${window}`, `Rolling kurtosis ${window}`);
                // Rolling min/max
                const rollingMin = StatisticsUtils.rollingStatistic(closes, window, 'min');
                const rollingMax = StatisticsUtils.rollingStatistic(closes, window, 'max');
                this.addFeatureColumn(features, rollingMin, `rolling_min_${window}`, `Rolling min ${window}`);
                this.addFeatureColumn(features, rollingMax, `rolling_max_${window}`, `Rolling max ${window}`);
            }
        }
        // Autocorrelation features
        for (let lag = 1; lag <= 10; lag++) {
            const autocorr = this.calculateRollingAutocorrelation(returns, lag, 50);
            this.addFeatureColumn(features, autocorr, `autocorr_${lag}`, `Autocorrelation lag ${lag}`);
        }
    }
    /**
     * Add volatility features
     */
    addVolatilityFeatures(data, features) {
        const closes = data.map(d => d.close);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        const returns = this.calculateReturns(closes);
        // Realized volatility (different estimators)
        for (const window of [10, 20, 50]) {
            // Close-to-close volatility
            const ccVol = this.calculateRollingVolatility(returns, window);
            this.addFeatureColumn(features, ccVol, `cc_vol_${window}`, `Close-to-close volatility ${window}`);
            // Parkinson volatility (high-low)
            const parkVol = this.calculateParkinsonVolatility(highs, lows, window);
            this.addFeatureColumn(features, parkVol, `park_vol_${window}`, `Parkinson volatility ${window}`);
            // Volatility of volatility
            const volOfVol = this.calculateRollingVolatility(ccVol, Math.min(window, 20));
            this.addFeatureColumn(features, volOfVol, `vol_of_vol_${window}`, `Volatility of volatility ${window}`);
        }
        // GARCH-like features
        const garchVol = this.calculateGARCHVolatility(returns);
        this.addFeatureColumn(features, garchVol, 'garch_vol', 'GARCH-like volatility');
        // Volatility regime indicators
        const volRegime = this.detectVolatilityRegime(returns, 50);
        this.addFeatureColumn(features, volRegime, 'vol_regime', 'Volatility regime');
    }
    /**
     * Add cross-sectional features
     */
    addCrossSectionalFeatures(data, features) {
        const closes = data.map(d => d.close);
        const volumes = data.map(d => d.volume);
        const returns = this.calculateReturns(closes);
        // Rank-based features
        for (const window of [20, 50]) {
            const returnRanks = this.calculateRollingRanks(returns, window);
            const volumeRanks = this.calculateRollingRanks(volumes, window);
            this.addFeatureColumn(features, returnRanks, `return_rank_${window}`, `Return rank ${window}`);
            this.addFeatureColumn(features, volumeRanks, `volume_rank_${window}`, `Volume rank ${window}`);
        }
        // Z-score features
        for (const window of [20, 50]) {
            const returnZScores = this.calculateRollingZScores(returns, window);
            const volumeZScores = this.calculateRollingZScores(volumes, window);
            this.addFeatureColumn(features, returnZScores, `return_zscore_${window}`, `Return z-score ${window}`);
            this.addFeatureColumn(features, volumeZScores, `volume_zscore_${window}`, `Volume z-score ${window}`);
        }
    }
    /**
     * Add pattern recognition features
     */
    addPatternFeatures(data, features) {
        const closes = data.map(d => d.close);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        const opens = data.map(d => d.open);
        // Candlestick patterns
        const doji = this.detectDoji(opens, closes);
        const hammer = this.detectHammer(opens, highs, lows, closes);
        const engulfing = this.detectEngulfing(opens, closes);
        this.addFeatureColumn(features, doji, 'doji', 'Doji pattern');
        this.addFeatureColumn(features, hammer, 'hammer', 'Hammer pattern');
        this.addFeatureColumn(features, engulfing, 'engulfing', 'Engulfing pattern');
        // Support/Resistance levels
        const supportResistance = this.detectSupportResistance(closes, 20);
        this.addFeatureColumn(features, supportResistance.support, 'support_level', 'Support level');
        this.addFeatureColumn(features, supportResistance.resistance, 'resistance_level', 'Resistance level');
        // Trend patterns
        const trendStrength = this.calculateTrendStrength(closes, 20);
        this.addFeatureColumn(features, trendStrength, 'trend_strength', 'Trend strength');
    }
    /**
     * Add harmonic features
     */
    addHarmonicFeatures(data, features) {
        const closes = data.map(d => d.close);
        const returns = this.calculateReturns(closes);
        // Fourier transform features (simplified)
        const fourierFeatures = this.calculateFourierFeatures(returns, 50);
        for (let i = 0; i < fourierFeatures.length; i++) {
            this.addFeatureColumn(features, fourierFeatures[i], `fourier_${i}`, `Fourier component ${i}`);
        }
        // Cyclical features
        const cyclicalFeatures = this.calculateCyclicalFeatures(closes);
        for (let i = 0; i < cyclicalFeatures.length; i++) {
            this.addFeatureColumn(features, cyclicalFeatures[i], `cyclical_${i}`, `Cyclical component ${i}`);
        }
    }
    /**
     * Helper method to add a feature column
     */
    addFeatureColumn(features, values, name, description) {
        // Pad with zeros if values array is shorter
        const paddedValues = new Array(features.length).fill(0);
        const startIndex = Math.max(0, features.length - values.length);
        for (let i = 0; i < values.length && startIndex + i < features.length; i++) {
            paddedValues[startIndex + i] = isFinite(values[i]) ? values[i] : 0;
        }
        // Add to each row
        for (let i = 0; i < features.length; i++) {
            features[i].push(paddedValues[i]);
        }
        // Add metadata
        this.featureNames.push(name);
        this.featureMetadata.push({
            name,
            category: 'technical',
            description,
            dataType: 'numeric',
            missingValueStrategy: 'zero'
        });
    }
    /**
     * Calculate returns
     */
    calculateReturns(prices) {
        const returns = [];
        for (let i = 1; i < prices.length; i++) {
            if (prices[i - 1] !== 0) {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            }
            else {
                returns.push(0);
            }
        }
        return returns;
    }
    /**
     * Calculate log returns
     */
    calculateLogReturns(prices) {
        const logReturns = [];
        for (let i = 1; i < prices.length; i++) {
            if (prices[i - 1] > 0 && prices[i] > 0) {
                logReturns.push(Math.log(prices[i] / prices[i - 1]));
            }
            else {
                logReturns.push(0);
            }
        }
        return logReturns;
    }
    /**
     * Calculate velocity (first difference)
     */
    calculateVelocity(values) {
        const velocity = [];
        for (let i = 1; i < values.length; i++) {
            velocity.push(values[i] - values[i - 1]);
        }
        return velocity;
    }
    /**
     * Calculate moving average
     */
    calculateMovingAverage(data, window) {
        return TechnicalIndicators.sma(data, window);
    }
    /**
     * Calculate rolling correlation
     */
    calculateRollingCorrelation(x, y, window) {
        return MathUtils.rollingCorrelation(x, y, window);
    }
    /**
     * Calculate rolling volatility
     */
    calculateRollingVolatility(returns, window) {
        return StatisticsUtils.rollingStatistic(returns, window, 'std');
    }
    /**
     * Calculate Parkinson volatility
     */
    calculateParkinsonVolatility(highs, lows, window) {
        const logHL = highs.map((h, i) => Math.log(h / lows[i]));
        const parkVol = [];
        for (let i = window - 1; i < logHL.length; i++) {
            const slice = logHL.slice(i - window + 1, i + 1);
            const variance = slice.reduce((sum, val) => sum + val * val, 0) / (4 * Math.log(2) * window);
            parkVol.push(Math.sqrt(variance));
        }
        return parkVol;
    }
    /**
     * Calculate GARCH-like volatility
     */
    calculateGARCHVolatility(returns) {
        const garchVol = [];
        let variance = 0.01; // Initial variance
        const alpha = 0.1; // ARCH parameter
        const beta = 0.85; // GARCH parameter
        const omega = 0.000001; // Constant
        for (const ret of returns) {
            variance = omega + alpha * ret * ret + beta * variance;
            garchVol.push(Math.sqrt(variance));
        }
        return garchVol;
    }
    /**
     * Detect volatility regime
     */
    detectVolatilityRegime(returns, window) {
        const vol = this.calculateRollingVolatility(returns, window);
        const volMean = StatisticsUtils.mean(vol);
        const volStd = StatisticsUtils.standardDeviation(vol);
        return vol.map(v => {
            if (v > volMean + volStd)
                return 2; // High volatility
            if (v < volMean - volStd)
                return 0; // Low volatility
            return 1; // Normal volatility
        });
    }
    /**
     * Calculate rolling ranks
     */
    calculateRollingRanks(data, window) {
        const ranks = [];
        for (let i = window - 1; i < data.length; i++) {
            const slice = data.slice(i - window + 1, i + 1);
            const currentValue = data[i];
            const rank = slice.filter(val => val <= currentValue).length / window;
            ranks.push(rank);
        }
        return ranks;
    }
    /**
     * Calculate rolling z-scores
     */
    calculateRollingZScores(data, window) {
        const zScores = [];
        for (let i = window - 1; i < data.length; i++) {
            const slice = data.slice(i - window + 1, i + 1);
            const mean = StatisticsUtils.mean(slice);
            const std = StatisticsUtils.standardDeviation(slice);
            if (std > 0) {
                zScores.push((data[i] - mean) / std);
            }
            else {
                zScores.push(0);
            }
        }
        return zScores;
    }
    /**
     * Calculate rolling autocorrelation
     */
    calculateRollingAutocorrelation(data, lag, window) {
        const autocorr = [];
        for (let i = window - 1; i < data.length - lag; i++) {
            const slice = data.slice(i - window + 1, i + 1);
            const laggedSlice = data.slice(i - window + 1 + lag, i + 1 + lag);
            if (slice.length === laggedSlice.length) {
                const correlation = MathUtils.correlation(slice, laggedSlice);
                autocorr.push(correlation);
            }
            else {
                autocorr.push(0);
            }
        }
        return autocorr;
    }
    /**
     * Detect Doji candlestick pattern
     */
    detectDoji(opens, closes) {
        return opens.map((open, i) => {
            const bodySize = Math.abs(closes[i] - open) / open;
            return bodySize < 0.001 ? 1 : 0; // Doji if body is very small
        });
    }
    /**
     * Detect Hammer candlestick pattern
     */
    detectHammer(opens, highs, lows, closes) {
        return opens.map((open, i) => {
            const bodySize = Math.abs(closes[i] - open);
            const lowerShadow = Math.min(open, closes[i]) - lows[i];
            const upperShadow = highs[i] - Math.max(open, closes[i]);
            // Hammer: small body, long lower shadow, short upper shadow
            return (lowerShadow > 2 * bodySize && upperShadow < bodySize) ? 1 : 0;
        });
    }
    /**
     * Detect Engulfing pattern
     */
    detectEngulfing(opens, closes) {
        const pattern = [0]; // First candle can't be engulfing
        for (let i = 1; i < opens.length; i++) {
            const prevBody = Math.abs(closes[i - 1] - opens[i - 1]);
            const currBody = Math.abs(closes[i] - opens[i]);
            // Bullish engulfing
            if (closes[i - 1] < opens[i - 1] && closes[i] > opens[i] &&
                opens[i] < closes[i - 1] && closes[i] > opens[i - 1] &&
                currBody > prevBody) {
                pattern.push(1);
            }
            // Bearish engulfing
            else if (closes[i - 1] > opens[i - 1] && closes[i] < opens[i] &&
                opens[i] > closes[i - 1] && closes[i] < opens[i - 1] &&
                currBody > prevBody) {
                pattern.push(-1);
            }
            else {
                pattern.push(0);
            }
        }
        return pattern;
    }
    /**
     * Detect support and resistance levels
     */
    detectSupportResistance(closes, window) {
        const support = [];
        const resistance = [];
        for (let i = window; i < closes.length; i++) {
            const slice = closes.slice(i - window, i);
            const currentPrice = closes[i];
            // Support: lowest price in window
            const supportLevel = Math.min(...slice);
            support.push(supportLevel / currentPrice);
            // Resistance: highest price in window
            const resistanceLevel = Math.max(...slice);
            resistance.push(resistanceLevel / currentPrice);
        }
        return { support, resistance };
    }
    /**
     * Calculate trend strength
     */
    calculateTrendStrength(closes, window) {
        const trendStrength = [];
        for (let i = window - 1; i < closes.length; i++) {
            const slice = closes.slice(i - window + 1, i + 1);
            const x = Array.from({ length: window }, (_, idx) => idx);
            // Linear regression slope as trend strength
            const correlation = MathUtils.correlation(x, slice);
            trendStrength.push(correlation);
        }
        return trendStrength;
    }
    /**
     * Calculate Fourier features (simplified)
     */
    calculateFourierFeatures(data, window) {
        const features = [[], []]; // Real and imaginary parts
        for (let i = window - 1; i < data.length; i++) {
            const slice = data.slice(i - window + 1, i + 1);
            // Simple DFT for first few frequencies
            let realPart = 0;
            let imagPart = 0;
            for (let k = 0; k < slice.length; k++) {
                const angle = -2 * Math.PI * k / slice.length;
                realPart += slice[k] * Math.cos(angle);
                imagPart += slice[k] * Math.sin(angle);
            }
            features[0].push(realPart / slice.length);
            features[1].push(imagPart / slice.length);
        }
        return features;
    }
    /**
     * Calculate cyclical features
     */
    calculateCyclicalFeatures(closes) {
        const features = [];
        // Daily, weekly, monthly cycles (simplified)
        const cycles = [5, 20, 60]; // 5-day, 20-day, 60-day cycles
        for (const cycle of cycles) {
            const cyclicalFeature = [];
            for (let i = 0; i < closes.length; i++) {
                const phase = (2 * Math.PI * i) / cycle;
                cyclicalFeature.push(Math.sin(phase));
            }
            features.push(cyclicalFeature);
        }
        return features;
    }
}

/**
 * Indicators Configuration
 *
 * Configuration settings for technical indicators and their parameters.
 */
/**
 * Technical indicator configurations
 */
const INDICATORS_CONFIG = {
    // Moving Averages
    SMA: {
        defaultPeriods: [5, 10, 20, 50, 100, 200],
        minPeriod: 2,
        maxPeriod: 500
    },
    EMA: {
        defaultPeriods: [5, 10, 20, 50, 100, 200],
        minPeriod: 2,
        maxPeriod: 500
    },
    WMA: {
        defaultPeriods: [5, 10, 20, 50, 100],
        minPeriod: 2,
        maxPeriod: 200
    },
    // Momentum Indicators
    RSI: {
        defaultPeriod: 14,
        alternativePeriods: [7, 9, 14, 21, 25],
        overboughtLevel: 70,
        oversoldLevel: 30,
        minPeriod: 2,
        maxPeriod: 100
    },
    MACD: {
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
        alternativeSettings: [
            { fast: 5, slow: 35, signal: 5 },
            { fast: 8, slow: 17, signal: 9 },
            { fast: 12, slow: 26, signal: 9 }
        ]
    },
    STOCHASTIC: {
        kPeriod: 14,
        dPeriod: 3,
        smoothing: 3,
        overboughtLevel: 80,
        oversoldLevel: 20,
        alternativeSettings: [
            { k: 5, d: 3 },
            { k: 14, d: 3 },
            { k: 21, d: 5 }
        ]
    },
    WILLIAMS_R: {
        defaultPeriod: 14,
        alternativePeriods: [7, 14, 21],
        overboughtLevel: -20,
        oversoldLevel: -80
    },
    CCI: {
        defaultPeriod: 20,
        alternativePeriods: [14, 20, 50],
        overboughtLevel: 100,
        oversoldLevel: -100,
        constant: 0.015
    },
    // Volatility Indicators
    BOLLINGER_BANDS: {
        period: 20,
        multiplier: 2,
        alternativeSettings: [
            { period: 10, multiplier: 1.9 },
            { period: 20, multiplier: 2.0 },
            { period: 50, multiplier: 2.1 }
        ]
    },
    ATR: {
        defaultPeriod: 14,
        alternativePeriods: [7, 14, 21, 50],
        minPeriod: 2,
        maxPeriod: 100
    },
    // Trend Indicators
    ADX: {
        defaultPeriod: 14,
        alternativePeriods: [7, 14, 21],
        trendThreshold: 25,
        strongTrendThreshold: 40
    },
    PARABOLIC_SAR: {
        accelerationFactor: 0.02,
        maxAcceleration: 0.20,
        alternativeSettings: [
            { af: 0.01, max: 0.10 },
            { af: 0.02, max: 0.20 },
            { af: 0.03, max: 0.30 }
        ]
    },
    // Volume Indicators
    OBV: {
    // No parameters needed
    },
    MFI: {
        defaultPeriod: 14,
        alternativePeriods: [10, 14, 20],
        overboughtLevel: 80,
        oversoldLevel: 20
    },
    VWAP: {
        // Typically calculated from session start
        resetPeriod: 'session'
    },
    // Oscillators
    MOMENTUM: {
        defaultPeriod: 10,
        alternativePeriods: [5, 10, 20, 50]
    },
    ROC: {
        defaultPeriod: 10,
        alternativePeriods: [5, 10, 20, 50]
    },
    // Statistical Indicators
    STANDARD_DEVIATION: {
        defaultPeriod: 20,
        alternativePeriods: [10, 20, 50]
    },
    VARIANCE: {
        defaultPeriod: 20,
        alternativePeriods: [10, 20, 50]
    },
    LINEAR_REGRESSION_SLOPE: {
        defaultPeriod: 14,
        alternativePeriods: [7, 14, 21, 50]
    }
};

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
/**
 * Library version
 */
const VERSION = '2.0.0';

export { DEFAULT_CONFIG, FeatureEngineer, INDICATORS_CONFIG, MathUtils, StatisticsUtils, TechnicalIndicators, UltraPrecisionPredictor, VERSION, ValidationUtils };
//# sourceMappingURL=index.esm.js.map
