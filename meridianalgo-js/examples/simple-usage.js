/**
 * Simple Usage Example
 * 
 * Basic demonstration of MeridianAlgo-JS functionality.
 */

const { TechnicalIndicators, VERSION } = require('../dist/index.js');

console.log(`🚀 MeridianAlgo-JS v${VERSION} - Simple Usage Example`);
console.log('='.repeat(50));

// Sample price data
const prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113];

console.log('\n📊 Sample Price Data:');
console.log(prices.join(', '));

// Calculate Simple Moving Average
console.log('\n📈 Technical Indicators:');
try {
  const sma5 = TechnicalIndicators.sma(prices, 5);
  console.log(`SMA(5): ${sma5.map(v => v.toFixed(2)).join(', ')}`);
  
  const sma10 = TechnicalIndicators.sma(prices, 10);
  console.log(`SMA(10): ${sma10.map(v => v.toFixed(2)).join(', ')}`);
  
  // Calculate RSI
  const rsi = TechnicalIndicators.rsi(prices, 14);
  console.log(`RSI(14): ${rsi.map(v => v.toFixed(2)).join(', ')}`);
  
  console.log('\n✅ Simple usage example completed successfully!');
  
} catch (error) {
  console.error('❌ Error:', error.message);
}

console.log('\n📚 For more examples, check the examples/ directory');
console.log('📖 Documentation: https://github.com/meridianalgo/meridianalgo-js#readme');