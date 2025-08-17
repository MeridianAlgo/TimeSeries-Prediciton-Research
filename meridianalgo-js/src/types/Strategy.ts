/**
 * Trading Strategy Types
 * 
 * Type definitions for trading strategies, signals, and execution.
 */

/**
 * Trading signal
 */
export interface TradingSignal {
  /** Signal ID */
  id: string;
  /** Asset symbol */
  symbol: string;
  /** Signal action */
  action: 'buy' | 'sell' | 'hold';
  /** Signal strength (0-1) */
  strength: number;
  /** Confidence level (0-1) */
  confidence: number;
  /** Recommended quantity */
  quantity?: number;
  /** Target price */
  targetPrice?: number;
  /** Stop loss price */
  stopLoss?: number;
  /** Take profit price */
  takeProfit?: number;
  /** Signal timestamp */
  timestamp: Date;
  /** Strategy that generated the signal */
  strategy: string;
  /** Signal metadata */
  metadata?: {
    /** Technical indicators used */
    indicators?: Record<string, number>;
    /** Market conditions */
    marketConditions?: string[];
    /** Risk factors */
    riskFactors?: string[];
  };
}

/**
 * Base strategy interface
 */
export interface IStrategy {
  /** Strategy name */
  name: string;
  /** Strategy description */
  description: string;
  /** Strategy parameters */
  parameters: Record<string, unknown>;
  /** Initialize strategy */
  initialize(config: StrategyConfig): Promise<void>;
  /** Process new market data and generate signals */
  onData(data: import('./MarketData').MarketData[]): Promise<TradingSignal[]>;
  /** Handle position updates */
  onPosition?(position: Position): Promise<void>;
  /** Handle order fills */
  onFill?(fill: OrderFill): Promise<void>;
  /** Cleanup strategy resources */
  cleanup?(): Promise<void>;
}

/**
 * Strategy configuration
 */
export interface StrategyConfig {
  /** Strategy parameters */
  parameters: Record<string, unknown>;
  /** Assets to trade */
  symbols: string[];
  /** Data timeframe */
  timeframe: string;
  /** Risk management settings */
  riskManagement: RiskManagementConfig;
  /** Execution settings */
  execution: ExecutionConfig;
}

/**
 * Risk management configuration
 */
export interface RiskManagementConfig {
  /** Maximum position size as percentage of portfolio */
  maxPositionSize: number;
  /** Stop loss percentage */
  stopLossPercent?: number;
  /** Take profit percentage */
  takeProfitPercent?: number;
  /** Maximum daily loss */
  maxDailyLoss?: number;
  /** Maximum drawdown */
  maxDrawdown?: number;
  /** Position sizing method */
  positionSizing: 'fixed' | 'percent' | 'volatility' | 'kelly';
  /** Risk per trade */
  riskPerTrade: number;
}

/**
 * Execution configuration
 */
export interface ExecutionConfig {
  /** Order type */
  orderType: 'market' | 'limit' | 'stop' | 'stop_limit';
  /** Time in force */
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  /** Slippage tolerance */
  slippageTolerance: number;
  /** Minimum order size */
  minOrderSize: number;
  /** Maximum order size */
  maxOrderSize?: number;
  /** Order execution delay */
  executionDelay?: number;
}

/**
 * Position information
 */
export interface Position {
  /** Position ID */
  id: string;
  /** Asset symbol */
  symbol: string;
  /** Position side */
  side: 'long' | 'short';
  /** Position size */
  size: number;
  /** Average entry price */
  avgPrice: number;
  /** Current market price */
  marketPrice: number;
  /** Unrealized P&L */
  unrealizedPnL: number;
  /** Realized P&L */
  realizedPnL: number;
  /** Position timestamp */
  timestamp: Date;
  /** Stop loss price */
  stopLoss?: number;
  /** Take profit price */
  takeProfit?: number;
  /** Position metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Order information
 */
export interface Order {
  /** Order ID */
  id: string;
  /** Asset symbol */
  symbol: string;
  /** Order side */
  side: 'buy' | 'sell';
  /** Order type */
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  /** Order quantity */
  quantity: number;
  /** Order price (for limit orders) */
  price?: number;
  /** Stop price (for stop orders) */
  stopPrice?: number;
  /** Time in force */
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  /** Order status */
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected';
  /** Filled quantity */
  filledQuantity: number;
  /** Average fill price */
  avgFillPrice: number;
  /** Order timestamp */
  timestamp: Date;
  /** Strategy that created the order */
  strategy?: string;
}

/**
 * Order fill information
 */
export interface OrderFill {
  /** Fill ID */
  id: string;
  /** Order ID */
  orderId: string;
  /** Asset symbol */
  symbol: string;
  /** Fill side */
  side: 'buy' | 'sell';
  /** Fill quantity */
  quantity: number;
  /** Fill price */
  price: number;
  /** Fill timestamp */
  timestamp: Date;
  /** Commission paid */
  commission: number;
  /** Fill metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Strategy performance metrics
 */
export interface StrategyPerformance {
  /** Total return */
  totalReturn: number;
  /** Annualized return */
  annualizedReturn: number;
  /** Volatility */
  volatility: number;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Sortino ratio */
  sortinoRatio: number;
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Win rate */
  winRate: number;
  /** Profit factor */
  profitFactor: number;
  /** Average win */
  avgWin: number;
  /** Average loss */
  avgLoss: number;
  /** Total trades */
  totalTrades: number;
  /** Winning trades */
  winningTrades: number;
  /** Losing trades */
  losingTrades: number;
}

/**
 * Strategy state
 */
export interface StrategyState {
  /** Strategy name */
  name: string;
  /** Current positions */
  positions: Position[];
  /** Open orders */
  orders: Order[];
  /** Account balance */
  balance: number;
  /** Equity */
  equity: number;
  /** Available margin */
  availableMargin: number;
  /** Performance metrics */
  performance: StrategyPerformance;
  /** Strategy status */
  status: 'running' | 'stopped' | 'paused' | 'error';
  /** Last update timestamp */
  lastUpdate: Date;
}

/**
 * Market condition detection
 */
export interface MarketCondition {
  /** Condition type */
  type: 'trending' | 'ranging' | 'volatile' | 'calm' | 'bullish' | 'bearish';
  /** Condition strength (0-1) */
  strength: number;
  /** Condition confidence (0-1) */
  confidence: number;
  /** Detection timestamp */
  timestamp: Date;
  /** Supporting indicators */
  indicators: Record<string, number>;
}

/**
 * Strategy optimization configuration
 */
export interface StrategyOptimizationConfig {
  /** Parameters to optimize */
  parameters: ParameterRange[];
  /** Optimization objective */
  objective: 'return' | 'sharpe' | 'sortino' | 'calmar' | 'custom';
  /** Optimization method */
  method: 'grid' | 'random' | 'genetic' | 'bayesian';
  /** Number of iterations */
  iterations: number;
  /** Cross-validation folds */
  cvFolds: number;
  /** Walk-forward analysis */
  walkForward?: {
    trainingPeriod: number;
    testingPeriod: number;
    step: number;
  };
}

export interface ParameterRange {
  /** Parameter name */
  name: string;
  /** Parameter type */
  type: 'int' | 'float' | 'categorical';
  /** Minimum value (for numeric parameters) */
  min?: number;
  /** Maximum value (for numeric parameters) */
  max?: number;
  /** Step size (for numeric parameters) */
  step?: number;
  /** Possible values (for categorical parameters) */
  values?: unknown[];
}

/**
 * Strategy optimization result
 */
export interface StrategyOptimizationResult {
  /** Best parameters found */
  bestParameters: Record<string, unknown>;
  /** Best objective value */
  bestScore: number;
  /** All tested parameter combinations */
  results: Array<{
    parameters: Record<string, unknown>;
    score: number;
    metrics: StrategyPerformance;
  }>;
  /** Optimization statistics */
  statistics: {
    totalCombinations: number;
    completedCombinations: number;
    averageScore: number;
    standardDeviation: number;
  };
}

/**
 * Multi-strategy portfolio
 */
export interface MultiStrategyPortfolio {
  /** Individual strategies */
  strategies: StrategyAllocation[];
  /** Portfolio-level risk management */
  riskManagement: PortfolioRiskManagement;
  /** Correlation matrix between strategies */
  correlations: number[][];
  /** Portfolio performance */
  performance: StrategyPerformance;
}

export interface StrategyAllocation {
  /** Strategy instance */
  strategy: IStrategy;
  /** Capital allocation (percentage) */
  allocation: number;
  /** Strategy weight in portfolio */
  weight: number;
  /** Strategy performance */
  performance: StrategyPerformance;
}

export interface PortfolioRiskManagement {
  /** Maximum portfolio drawdown */
  maxDrawdown: number;
  /** Maximum correlation between strategies */
  maxCorrelation: number;
  /** Minimum number of active strategies */
  minActiveStrategies: number;
  /** Strategy allocation limits */
  allocationLimits: {
    min: number;
    max: number;
  };
}