/**
 * Test Setup
 * 
 * Global test configuration and setup for Jest.
 */

// Extend Jest matchers
expect.extend({
  toBeCloseToArray(received: number[], expected: number[], precision: number = 2) {
    const pass = received.length === expected.length && 
      received.every((val, i) => Math.abs(val - expected[i]) < Math.pow(10, -precision));
    
    if (pass) {
      return {
        message: () => `expected ${received} not to be close to ${expected}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be close to ${expected}`,
        pass: false,
      };
    }
  },
});

// Global test timeout
jest.setTimeout(30000);

// Mock console methods in tests to reduce noise
global.console = {
  ...console,
  log: jest.fn(),
  warn: jest.fn(),
  error: console.error, // Keep error for debugging
};

// Global test data generators
global.generateTestMarketData = (count: number = 100, basePrice: number = 100) => {
  const data = [];
  let price = basePrice;
  
  for (let i = 0; i < count; i++) {
    const date = new Date('2024-01-01');
    date.setDate(date.getDate() + i);
    
    const change = (Math.random() - 0.5) * 0.02; // 2% max change
    price = price * (1 + change);
    
    const volatility = 0.01;
    const high = price * (1 + Math.random() * volatility);
    const low = price * (1 - Math.random() * volatility);
    const open = price * (1 + (Math.random() - 0.5) * volatility * 0.5);
    const close = price;
    const volume = Math.floor(1000000 + Math.random() * 500000);
    
    data.push({
      timestamp: date,
      symbol: 'TEST',
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume
    });
  }
  
  return data;
};

// Global types for TypeScript
export {};