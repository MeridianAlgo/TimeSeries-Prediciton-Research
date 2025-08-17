# Migration Guide: v1.x to v2.0

This guide helps you migrate from MeridianAlgo-JS v1.x to v2.0.

## Overview

Version 2.0 is a complete rewrite of the library with significant improvements in:
- **Ultra-precision machine learning capabilities**
- **Advanced feature engineering**
- **Comprehensive TypeScript support**
- **Modern build system and architecture**

## Breaking Changes

### 1. Import Structure

**v1.x:**
```javascript
const meridianalgo = require('meridianalgo-js');
```

**v2.0:**
```javascript
// ES6 modules
import { TechnicalIndicators, UltraPrecisionPredictor, FeatureEngineer } from 'meridianalgo-js';

// CommonJS
const { TechnicalIndicators, UltraPrecisionPredictor, FeatureEngineer } = require('meridianalgo-js');
```

### 2. Technical Indicators

**v1.x:**
```javascript
const sma = meridianalgo.sma(prices, 20);
```

**v2.0:**
```javascript
const sma = TechnicalIndicators.sma(prices, 20);
```

### 3. New Features

Version 2.0 introduces powerful new capabilities:

#### Ultra-Precision Predictor
```javascript
import { UltraPrecisionPredictor } from 'meridianalgo-js';

const predictor = new UltraPrecisionPredictor({
  targetErrorRate: 0.01, // Target 1% error rate
  ensembleSize: 10,
  featureCount: 1000
});

await predictor.train(trainingData);
const prediction = await predictor.predict(features);
```

#### Advanced Feature Engineering
```javascript
import { FeatureEngineer } from 'meridianalgo-js';

const engineer = new FeatureEngineer({
  targetFeatureCount: 1000,
  enableAdvancedFeatures: true
});

const features = engineer.generateFeatures(marketData);
```

## Migration Steps

### Step 1: Update Dependencies

```bash
npm install meridianalgo-js@^2.0.0
```

### Step 2: Update Imports

Replace old imports with new modular imports:

```javascript
// Old
const meridianalgo = require('meridianalgo-js');

// New
const { TechnicalIndicators, UltraPrecisionPredictor } = require('meridianalgo-js');
```

### Step 3: Update Function Calls

Update function calls to use the new class-based API:

```javascript
// Old
const rsi = meridianalgo.rsi(prices, 14);

// New
const rsi = TechnicalIndicators.rsi(prices, 14);
```

### Step 4: Leverage New Features

Take advantage of new ultra-precision capabilities:

```javascript
// New in v2.0 - Ultra-precision prediction
const predictor = new UltraPrecisionPredictor();
await predictor.train(data);
const prediction = await predictor.predict(features);

// New in v2.0 - Advanced feature engineering
const engineer = new FeatureEngineer();
const features = engineer.generateFeatures(marketData);
```

## Compatibility

### What's Compatible
- All technical indicators from v1.x are available in v2.0
- Basic calculation results remain the same
- Core mathematical functions are preserved

### What's Changed
- Import structure (modular imports)
- Function organization (class-based)
- Enhanced error handling
- Improved TypeScript support

## Getting Help

If you encounter issues during migration:

1. Check the [examples](./examples/) directory for working code samples
2. Review the [API documentation](./README.md#api-reference)
3. Open an issue on [GitHub](https://github.com/meridianalgo/meridianalgo-js/issues)

## Benefits of Upgrading

- **Ultra-precision predictions** with sub-1% error rates
- **1000+ advanced features** from basic market data
- **Full TypeScript support** with complete type definitions
- **Better performance** and memory efficiency
- **Modern architecture** for scalability
- **Comprehensive testing** and validation