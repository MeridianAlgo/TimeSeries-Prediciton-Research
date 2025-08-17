/**
 * Basic Prediction Example
 * 
 * Demonstrates basic usage of the MeridianAlgo-JS library for price prediction.
 */

const { 
  UltraPrecisionPredictor, 
  FeatureEngineer, 
  TechnicalIndicators 
} = require('../dist/index.js');

// Sample market data (in a real application, you'd load this from an API or file)
const sampleData = [
  { timestamp: new Date('2024-01-01'), symbol: 'AAPL', open: 150, high: 155, low: 148, close: 153, volume: 1000000 },
  { timestamp: new Date('2024-01-02'), symbol: 'AAPL', open: 153, high: 158, low: 151, close: 156, volume: 1200000 },
  { timestamp: new Date('2024-01-03'), symbol: 'AAPL', open: 156, high: 160, low: 154, close: 159, volume: 1100000 },
  { timestamp: new Date('2024-01-04'), symbol: 'AAPL', open: 159, high: 162, low: 157, close: 161, volume: 1300000 },
  { timestamp: new Date('2024-01-05'), symbol: 'AAPL', open: 161, high: 165, low: 159, close: 163, volume: 1150000 },
  // Add more data points...
];

// Generate more sample data for demonstration
function generateSampleData(basePrice = 150, days = 100) {
  const data = [];
  let price = basePrice;
  
  for (let i = 0; i < days; i++) {
    const date = new Date('2024-01-01');
    date.setDate(date.getDate() + i);
    
    // Simple random walk with slight upward bias
    const change = (Math.random() - 0.48) * 0.05; // Slight upward bias
    price = price * (1 + change);
    
    const volatility = 0.02;
    const high = price * (1 + Math.random() * volatility);
    const low = price * (1 - Math.random() * volatility);
    const open = price * (1 + (Math.random() - 0.5) * volatility * 0.5);
    const close = price;
    const volume = Math.floor(1000000 + Math.random() * 500000);
    
    data.push({
      timestamp: date,
      symbol: 'AAPL',
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume
    });
  }
  
  return data;
}

async function basicPredictionExample() {
  console.log('🚀 MeridianAlgo-JS Basic Prediction Example');
  console.log('==========================================');
  
  try {
    // Generate sample data
    const marketData = generateSampleData(150, 200);
    console.log(`📊 Generated ${marketData.length} data points`);
    
    // 1. Calculate basic technical indicators
    console.log('\n📈 Calculating Technical Indicators...');
    const closes = TechnicalIndicators.extractPrices(marketData, 'close');
    const highs = TechnicalIndicators.extractPrices(marketData, 'high');
    const lows = TechnicalIndicators.extractPrices(marketData, 'low');
    const volumes = TechnicalIndicators.extractPrices(marketData, 'volume');
    
    const rsi = TechnicalIndicators.rsi(closes, 14);
    const macd = TechnicalIndicators.macd(closes);
    const bollinger = TechnicalIndicators.bollingerBands(closes, 20, 2);
    
    console.log(`   RSI (last 5): ${rsi.slice(-5).map(v => v.toFixed(2)).join(', ')}`);
    console.log(`   MACD (last 3): ${macd.macd.slice(-3).map(v => v.toFixed(4)).join(', ')}`);
    console.log(`   Bollinger Upper (last 3): ${bollinger.upper.slice(-3).map(v => v.toFixed(2)).join(', ')}`);
    
    // 2. Initialize feature engineer
    console.log('\n🔧 Initializing Feature Engineer...');
    const featureEngineer = new FeatureEngineer({
      targetFeatureCount: 100, // Reduced for demo
      enableAdvancedFeatures: true,
      enableMicrostructure: false, // Disabled for basic example
      enableVolatilityFeatures: true
    });
    
    // 3. Generate features
    console.log('✨ Generating Features...');
    const features = featureEngineer.generateFeatures(marketData);
    console.log(`   Generated ${features.featureNames.length} features`);
    console.log(`   Feature matrix size: ${features.rows} x ${features.columns}`);
    
    // 4. Initialize ultra-precision predictor
    console.log('\n🧠 Initializing Ultra-Precision Predictor...');
    const predictor = new UltraPrecisionPredictor({
      targetErrorRate: 0.02, // 2% target error
      ensembleSize: 5,       // Reduced for demo
      featureCount: 100,     // Match feature engineer
      trainingRatio: 0.8
    });
    
    // 5. Prepare training data
    console.log('📚 Preparing Training Data...');
    const trainingData = marketData.map((data, index) => ({
      ...data,
      features: features.data[index] || [],
      target: index < marketData.length - 1 ? 
        (marketData[index + 1].close - data.close) / data.close : 0
    })).filter(d => d.features.length > 0);
    
    console.log(`   Training samples: ${trainingData.length}`);
    
    // 6. Train the model
    console.log('\n🎯 Training Model...');
    await predictor.train(trainingData.slice(0, -20)); // Reserve last 20 for testing
    
    const metrics = predictor.getTrainingMetrics();
    if (metrics) {
      console.log(`   Training MAE: ${(metrics.mae * 100).toFixed(3)}%`);
      console.log(`   Training R²: ${metrics.r2.toFixed(4)}`);
      console.log(`   Directional Accuracy: ${(metrics.directionalAccuracy * 100).toFixed(1)}%`);
    }
    
    // 7. Make predictions
    console.log('\n🔮 Making Predictions...');
    const testData = trainingData.slice(-20);
    const predictions = [];
    
    for (let i = 0; i < Math.min(5, testData.length); i++) {
      const prediction = await predictor.predict(testData[i].features);
      const confidence = predictor.getConfidence();
      const actual = testData[i].target;
      
      predictions.push({
        predicted: prediction,
        actual: actual,
        confidence: confidence,
        error: Math.abs(prediction - actual)
      });
      
      console.log(`   Prediction ${i + 1}:`);
      console.log(`     Predicted: ${(prediction * 100).toFixed(3)}%`);
      console.log(`     Actual: ${(actual * 100).toFixed(3)}%`);
      console.log(`     Confidence: ${(confidence * 100).toFixed(1)}%`);
      console.log(`     Error: ${(Math.abs(prediction - actual) * 100).toFixed(3)}%`);
    }
    
    // 8. Calculate prediction statistics
    console.log('\n📊 Prediction Statistics:');
    const avgError = predictions.reduce((sum, p) => sum + p.error, 0) / predictions.length;
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
    
    console.log(`   Average Error: ${(avgError * 100).toFixed(3)}%`);
    console.log(`   Average Confidence: ${(avgConfidence * 100).toFixed(1)}%`);
    
    // 9. Feature importance
    console.log('\n🎯 Top 10 Most Important Features:');
    const featureImportance = predictor.getFeatureImportance();
    const topFeatures = featureImportance
      .map((importance, index) => ({ importance, name: features.featureNames[index] || `Feature_${index}` }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);
    
    topFeatures.forEach((feature, index) => {
      console.log(`   ${index + 1}. ${feature.name}: ${(feature.importance * 100).toFixed(2)}%`);
    });
    
    console.log('\n✅ Basic prediction example completed successfully!');
    
  } catch (error) {
    console.error('❌ Error in basic prediction example:', error.message);
    console.error(error.stack);
  }
}

// Run the example
if (require.main === module) {
  basicPredictionExample();
}

module.exports = { basicPredictionExample };