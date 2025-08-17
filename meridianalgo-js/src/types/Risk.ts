/**
 * Risk Management Types
 * 
 * Type definitions for risk analysis, management, and monitoring.
 */

/**
 * Risk metrics calculation result
 */
export interface RiskMetrics {
  /** Value at Risk at different confidence levels */
  var: {
    var95: number;
    var99: number;
    var999: number;
  };
  /** Expected Shortfall (Conditional VaR) */
  expectedShortfall: {
    es95: number;
    es99: number;
    es999: number;
  };
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Current drawdown */
  currentDrawdown: number;
  /** Drawdown duration (in periods) */
  drawdownDuration: number;
  /** Volatility (annualized) */
  volatility: number;
  /** Downside deviation */
  downsideDeviation: number;
  /** Skewness */
  skewness: number;
  /** Excess kurtosis */
  kurtosis: number;
  /** Tail ratio (95th percentile / 5th percentile) */
  tailRatio: number;
  /** Gain-to-pain ratio */
  gainToPainRatio: number;
}

/**
 * Risk assessment configuration
 */
export interface RiskAssessmentConfig {
  /** Confidence levels for VaR calculation */
  confidenceLevels: number[];
  /** Lookback period for risk calculation */
  lookbackPeriod: number;
  /** Risk-free rate for Sharpe ratio calculation */
  riskFreeRate: number;
  /** Benchmark for beta calculation */
  benchmark?: number[];
  /** Monte Carlo simulation parameters */
  monteCarlo?: {
    simulations: number;
    horizon: number;
  };
}

/**
 * Position risk analysis
 */
export interface PositionRisk {
  /** Position identifier */
  positionId: string;
  /** Asset symbol */
  symbol: string;
  /** Position size */
  size: number;
  /** Market value */
  marketValue: number;
  /** Portfolio weight */
  portfolioWeight: number;
  /** Position VaR */
  var95: number;
  /** Position beta */
  beta: number;
  /** Concentration risk score */
  concentrationRisk: number;
  /** Liquidity risk score */
  liquidityRisk: number;
  /** Currency risk (for foreign assets) */
  currencyRisk?: number;
}

/**
 * Portfolio risk analysis
 */
export interface PortfolioRisk {
  /** Overall portfolio risk metrics */
  metrics: RiskMetrics;
  /** Individual position risks */
  positions: PositionRisk[];
  /** Risk decomposition by asset */
  assetRiskContribution: Array<{
    symbol: string;
    contribution: number;
    percentage: number;
  }>;
  /** Risk decomposition by sector */
  sectorRiskContribution?: Array<{
    sector: string;
    contribution: number;
    percentage: number;
  }>;
  /** Correlation matrix */
  correlationMatrix: number[][];
  /** Risk budget utilization */
  riskBudgetUtilization: number;
}

/**
 * Risk limit configuration
 */
export interface RiskLimits {
  /** Maximum portfolio VaR */
  maxPortfolioVar: number;
  /** Maximum position size (as % of portfolio) */
  maxPositionSize: number;
  /** Maximum sector concentration */
  maxSectorConcentration: number;
  /** Maximum single asset concentration */
  maxAssetConcentration: number;
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Maximum daily loss */
  maxDailyLoss: number;
  /** Maximum leverage */
  maxLeverage: number;
  /** Minimum liquidity ratio */
  minLiquidityRatio: number;
}

/**
 * Risk alert
 */
export interface RiskAlert {
  /** Alert ID */
  id: string;
  /** Alert type */
  type: 'limit_breach' | 'concentration' | 'correlation' | 'volatility' | 'drawdown';
  /** Alert severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Alert message */
  message: string;
  /** Affected asset/position */
  asset?: string;
  /** Current value */
  currentValue: number;
  /** Limit/threshold */
  threshold: number;
  /** Alert timestamp */
  timestamp: Date;
  /** Recommended actions */
  recommendations: string[];
}

/**
 * Risk monitoring configuration
 */
export interface RiskMonitoringConfig {
  /** Risk limits */
  limits: RiskLimits;
  /** Monitoring frequency */
  frequency: 'realtime' | 'minute' | 'hourly' | 'daily';
  /** Alert thresholds */
  alertThresholds: {
    warning: number; // percentage of limit
    critical: number; // percentage of limit
  };
  /** Notification settings */
  notifications: {
    email?: string[];
    webhook?: string;
    sms?: string[];
  };
}

/**
 * Stress testing configuration
 */
export interface StressTestConfig {
  /** Stress test scenarios */
  scenarios: StressScenario[];
  /** Monte Carlo parameters */
  monteCarlo?: {
    simulations: number;
    horizon: number;
    confidenceLevel: number;
  };
  /** Historical scenarios */
  historical?: {
    events: string[];
    lookbackPeriod: number;
  };
}

export interface StressScenario {
  /** Scenario name */
  name: string;
  /** Scenario description */
  description: string;
  /** Market shocks */
  shocks: Array<{
    asset: string;
    shock: number; // percentage change
  }>;
  /** Correlation changes */
  correlationShocks?: Array<{
    asset1: string;
    asset2: string;
    newCorrelation: number;
  }>;
  /** Volatility shocks */
  volatilityShocks?: Array<{
    asset: string;
    multiplier: number;
  }>;
}

/**
 * Stress test results
 */
export interface StressTestResults {
  /** Scenario results */
  scenarios: Array<{
    scenario: string;
    portfolioReturn: number;
    portfolioValue: number;
    maxDrawdown: number;
    var95: number;
    worstAsset: {
      symbol: string;
      return: number;
    };
  }>;
  /** Monte Carlo results */
  monteCarlo?: {
    meanReturn: number;
    stdReturn: number;
    var95: number;
    var99: number;
    expectedShortfall: number;
    probabilityOfLoss: number;
  };
  /** Summary statistics */
  summary: {
    worstScenario: string;
    worstReturn: number;
    averageReturn: number;
    probabilityOfLoss: number;
  };
}

/**
 * Liquidity risk assessment
 */
export interface LiquidityRisk {
  /** Asset liquidity scores */
  assetLiquidity: Array<{
    symbol: string;
    liquidityScore: number; // 0-1 scale
    avgDailyVolume: number;
    bidAskSpread: number;
    marketImpact: number;
  }>;
  /** Portfolio liquidity metrics */
  portfolioLiquidity: {
    weightedLiquidityScore: number;
    liquidationTime: number; // days to liquidate
    liquidationCost: number; // percentage cost
  };
  /** Liquidity stress scenarios */
  stressScenarios: Array<{
    scenario: string;
    liquidationTime: number;
    liquidationCost: number;
  }>;
}

/**
 * Credit risk assessment (for fixed income)
 */
export interface CreditRisk {
  /** Credit ratings */
  ratings: Array<{
    asset: string;
    rating: string;
    probability: number; // default probability
  }>;
  /** Credit spread risk */
  spreadRisk: Array<{
    asset: string;
    spread: number;
    duration: number;
    spreadVar: number;
  }>;
  /** Portfolio credit metrics */
  portfolioMetrics: {
    averageRating: string;
    creditVar: number;
    expectedLoss: number;
    unexpectedLoss: number;
  };
}

/**
 * Market risk factors
 */
export interface MarketRiskFactors {
  /** Equity risk factors */
  equity?: {
    marketReturn: number;
    size: number; // SMB factor
    value: number; // HML factor
    momentum: number;
    quality: number;
  };
  /** Fixed income risk factors */
  fixedIncome?: {
    interestRate: number;
    creditSpread: number;
    termStructure: number[];
  };
  /** Currency risk factors */
  currency?: Array<{
    pair: string;
    return: number;
    volatility: number;
  }>;
  /** Commodity risk factors */
  commodity?: Array<{
    commodity: string;
    return: number;
    volatility: number;
  }>;
}

/**
 * Risk attribution analysis
 */
export interface RiskAttribution {
  /** Factor contributions to portfolio risk */
  factorContributions: Array<{
    factor: string;
    contribution: number;
    percentage: number;
  }>;
  /** Asset contributions to portfolio risk */
  assetContributions: Array<{
    asset: string;
    contribution: number;
    percentage: number;
  }>;
  /** Residual (idiosyncratic) risk */
  residualRisk: number;
  /** Total portfolio risk */
  totalRisk: number;
}

/**
 * Dynamic risk management
 */
export interface DynamicRiskManagement {
  /** Risk regime detection */
  riskRegime: 'low' | 'medium' | 'high' | 'extreme';
  /** Adaptive position sizing */
  adaptivePositionSizing: boolean;
  /** Dynamic hedging */
  dynamicHedging: {
    enabled: boolean;
    hedgeRatio: number;
    hedgeInstruments: string[];
  };
  /** Volatility targeting */
  volatilityTargeting: {
    enabled: boolean;
    targetVolatility: number;
    currentVolatility: number;
    adjustment: number;
  };
}