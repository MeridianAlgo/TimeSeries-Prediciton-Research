/**
 * Portfolio Management Types
 * 
 * Type definitions for portfolio optimization, risk management, and asset allocation.
 */

/**
 * Portfolio optimizer configuration options
 */
export interface OptimizerOptions {
  /** Optimization objective */
  objective: 'sharpe' | 'return' | 'risk' | 'sortino' | 'calmar' | 'custom';
  /** Portfolio constraints */
  constraints: PortfolioConstraints;
  /** Risk model to use */
  riskModel: 'historical' | 'factor' | 'garch' | 'shrinkage';
  /** Optimization method */
  optimizationMethod: 'quadratic' | 'genetic' | 'gradient' | 'montecarlo';
  /** Rebalancing frequency */
  rebalanceFrequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annually';
  /** Transaction cost model */
  transactionCosts?: TransactionCostModel;
}

/**
 * Portfolio constraints
 */
export interface PortfolioConstraints {
  /** Minimum weight for any asset */
  minWeight: number;
  /** Maximum weight for any asset */
  maxWeight: number;
  /** Minimum total portfolio weight (should be close to 1.0) */
  minTotalWeight: number;
  /** Maximum total portfolio weight (should be close to 1.0) */
  maxTotalWeight: number;
  /** Maximum portfolio volatility */
  maxVolatility?: number;
  /** Minimum expected return */
  minReturn?: number;
  /** Maximum drawdown */
  maxDrawdown?: number;
  /** Sector/category constraints */
  sectorConstraints?: Record<string, { min: number; max: number }>;
  /** Long-only constraint */
  longOnly: boolean;
  /** Turnover constraint */
  maxTurnover?: number;
}

/**
 * Asset information
 */
export interface Asset {
  /** Asset symbol */
  symbol: string;
  /** Asset name */
  name: string;
  /** Asset category/sector */
  category: string;
  /** Current price */
  price: number;
  /** Expected return */
  expectedReturn: number;
  /** Volatility */
  volatility: number;
  /** Beta (market sensitivity) */
  beta?: number;
  /** Market capitalization */
  marketCap?: number;
  /** Liquidity score */
  liquidity?: number;
}

/**
 * Portfolio composition
 */
export interface Portfolio {
  /** Portfolio assets */
  assets: Asset[];
  /** Asset weights (sum should equal 1.0) */
  weights: number[];
  /** Total portfolio value */
  totalValue: number;
  /** Last rebalancing date */
  lastRebalance: Date;
  /** Portfolio performance metrics */
  performance: PerformanceMetrics;
  /** Portfolio metadata */
  metadata?: {
    /** Portfolio name */
    name?: string;
    /** Portfolio strategy */
    strategy?: string;
    /** Creation date */
    createdAt?: Date;
    /** Portfolio manager */
    manager?: string;
  };
}

/**
 * Optimal portfolio result
 */
export interface OptimalPortfolio {
  /** Optimal asset weights */
  weights: number[];
  /** Expected portfolio return */
  expectedReturn: number;
  /** Expected portfolio risk (volatility) */
  expectedRisk: number;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Optimization status */
  status: 'optimal' | 'suboptimal' | 'failed';
  /** Optimization iterations */
  iterations: number;
  /** Objective function value */
  objectiveValue: number;
}

/**
 * Efficient frontier point
 */
export interface EfficientFrontierPoint {
  /** Expected return */
  return: number;
  /** Risk (volatility) */
  risk: number;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Asset weights */
  weights: number[];
}

/**
 * Efficient frontier
 */
export interface EfficientFrontier {
  /** Frontier points */
  points: EfficientFrontierPoint[];
  /** Minimum variance portfolio */
  minVariancePortfolio: EfficientFrontierPoint;
  /** Maximum Sharpe ratio portfolio */
  maxSharpePortfolio: EfficientFrontierPoint;
  /** Risk-free rate used */
  riskFreeRate: number;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  /** Total return */
  totalReturn: number;
  /** Annualized return */
  annualizedReturn: number;
  /** Volatility (annualized) */
  volatility: number;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Sortino ratio */
  sortinoRatio: number;
  /** Calmar ratio */
  calmarRatio: number;
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Value at Risk (95%) */
  var95: number;
  /** Expected Shortfall (95%) */
  expectedShortfall95: number;
  /** Beta */
  beta: number;
  /** Alpha */
  alpha: number;
  /** Information ratio */
  informationRatio: number;
  /** Tracking error */
  trackingError: number;
  /** Win rate */
  winRate: number;
  /** Average win */
  averageWin: number;
  /** Average loss */
  averageLoss: number;
}

/**
 * Risk metrics
 */
export interface RiskMetrics {
  /** Value at Risk at different confidence levels */
  var: Record<string, number>;
  /** Expected Shortfall at different confidence levels */
  expectedShortfall: Record<string, number>;
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Drawdown duration */
  drawdownDuration: number;
  /** Volatility */
  volatility: number;
  /** Downside deviation */
  downsideDeviation: number;
  /** Skewness */
  skewness: number;
  /** Kurtosis */
  kurtosis: number;
  /** Tail ratio */
  tailRatio: number;
}

/**
 * Risk assessment result
 */
export interface RiskAssessment {
  /** Overall risk score (0-100) */
  riskScore: number;
  /** Risk level */
  riskLevel: 'low' | 'medium' | 'high' | 'extreme';
  /** Risk metrics */
  metrics: RiskMetrics;
  /** Risk factors */
  factors: RiskFactor[];
  /** Recommendations */
  recommendations: string[];
}

export interface RiskFactor {
  /** Factor name */
  name: string;
  /** Factor impact (0-1) */
  impact: number;
  /** Factor description */
  description: string;
  /** Mitigation strategies */
  mitigation: string[];
}

/**
 * Position sizing parameters
 */
export interface PositionSizeParams {
  /** Account balance */
  accountBalance: number;
  /** Risk per trade (as percentage of account) */
  riskPerTrade: number;
  /** Entry price */
  entryPrice: number;
  /** Stop loss price */
  stopLossPrice?: number;
  /** Volatility-based sizing */
  volatility?: number;
  /** Kelly criterion parameters */
  kelly?: {
    winRate: number;
    avgWin: number;
    avgLoss: number;
  };
}

/**
 * Rebalancing configuration
 */
export interface RebalanceConfig {
  /** Rebalancing frequency */
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'threshold';
  /** Threshold for threshold-based rebalancing */
  threshold?: number;
  /** Minimum trade size */
  minTradeSize?: number;
  /** Transaction cost consideration */
  considerTransactionCosts: boolean;
  /** Rebalancing method */
  method: 'full' | 'partial' | 'drift';
}

/**
 * Transaction cost model
 */
export interface TransactionCostModel {
  /** Fixed cost per transaction */
  fixedCost: number;
  /** Variable cost as percentage of trade value */
  variableCost: number;
  /** Market impact model */
  marketImpact?: {
    /** Linear impact coefficient */
    linear: number;
    /** Square root impact coefficient */
    sqrt: number;
  };
  /** Bid-ask spread cost */
  bidAskSpread?: number;
}

/**
 * Backtesting configuration for portfolios
 */
export interface PortfolioBacktestConfig {
  /** Start date */
  startDate: Date;
  /** End date */
  endDate: Date;
  /** Initial capital */
  initialCapital: number;
  /** Rebalancing configuration */
  rebalancing: RebalanceConfig;
  /** Benchmark for comparison */
  benchmark?: string;
  /** Transaction costs */
  transactionCosts?: TransactionCostModel;
}

/**
 * Portfolio backtest results
 */
export interface PortfolioBacktestResults {
  /** Portfolio performance */
  portfolio: PerformanceMetrics;
  /** Benchmark performance (if provided) */
  benchmark?: PerformanceMetrics;
  /** Equity curve */
  equityCurve: Array<{
    date: Date;
    value: number;
    drawdown: number;
  }>;
  /** Trade history */
  trades: Trade[];
  /** Rebalancing history */
  rebalances: Array<{
    date: Date;
    oldWeights: number[];
    newWeights: number[];
    turnover: number;
    cost: number;
  }>;
}

/**
 * Trade information
 */
export interface Trade {
  /** Trade ID */
  id: string;
  /** Asset symbol */
  symbol: string;
  /** Trade side */
  side: 'buy' | 'sell';
  /** Quantity */
  quantity: number;
  /** Price */
  price: number;
  /** Timestamp */
  timestamp: Date;
  /** Transaction cost */
  cost: number;
  /** Trade reason */
  reason: 'rebalance' | 'signal' | 'risk_management';
}

/**
 * Risk parity configuration
 */
export interface RiskParityConfig {
  /** Risk budgets for each asset (should sum to 1.0) */
  riskBudgets?: number[];
  /** Risk measure to use */
  riskMeasure: 'volatility' | 'var' | 'expected_shortfall';
  /** Lookback period for risk estimation */
  lookbackPeriod: number;
  /** Optimization tolerance */
  tolerance: number;
  /** Maximum iterations */
  maxIterations: number;
}

/**
 * Multi-asset portfolio configuration
 */
export interface MultiAssetConfig {
  /** Asset universe */
  universe: Asset[];
  /** Asset allocation strategy */
  strategy: 'equal_weight' | 'market_cap' | 'risk_parity' | 'momentum' | 'mean_reversion';
  /** Rebalancing configuration */
  rebalancing: RebalanceConfig;
  /** Risk management rules */
  riskManagement: {
    /** Maximum position size */
    maxPosition: number;
    /** Stop loss percentage */
    stopLoss?: number;
    /** Volatility target */
    volatilityTarget?: number;
  };
}