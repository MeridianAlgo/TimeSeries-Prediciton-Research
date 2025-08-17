/**
 * Market Data Types
 * 
 * Core data structures for market data representation and processing.
 */

/**
 * Basic market data structure representing OHLCV data
 */
export interface MarketData {
  /** Timestamp of the data point */
  timestamp: Date;
  /** Trading symbol (e.g., 'AAPL', 'BTC-USD') */
  symbol: string;
  /** Opening price */
  open: number;
  /** Highest price during the period */
  high: number;
  /** Lowest price during the period */
  low: number;
  /** Closing price */
  close: number;
  /** Trading volume */
  volume: number;
  /** Volume-weighted average price (optional) */
  vwap?: number;
  /** Number of trades (optional) */
  trades?: number;
}

/**
 * Extended market data with additional fields for advanced analysis
 */
export interface ExtendedMarketData extends MarketData {
  /** Bid price */
  bid?: number;
  /** Ask price */
  ask?: number;
  /** Bid size */
  bidSize?: number;
  /** Ask size */
  askSize?: number;
  /** Open interest (for derivatives) */
  openInterest?: number;
  /** Funding rate (for perpetual contracts) */
  fundingRate?: number;
}



/**
 * Tick data for high-frequency analysis
 */
export interface TickData {
  /** Timestamp of the tick */
  timestamp: Date;
  /** Trading symbol */
  symbol: string;
  /** Price of the trade */
  price: number;
  /** Size of the trade */
  size: number;
  /** Side of the trade ('buy' or 'sell') */
  side: 'buy' | 'sell';
  /** Trade ID */
  tradeId?: string;
}

/**
 * Order book data structure
 */
export interface OrderBookLevel {
  /** Price level */
  price: number;
  /** Size at this level */
  size: number;
  /** Number of orders at this level */
  count?: number;
}

export interface OrderBook {
  /** Timestamp of the order book snapshot */
  timestamp: Date;
  /** Trading symbol */
  symbol: string;
  /** Bid levels (sorted by price descending) */
  bids: OrderBookLevel[];
  /** Ask levels (sorted by price ascending) */
  asks: OrderBookLevel[];
  /** Sequence number for ordering */
  sequence?: number;
}

/**
 * Data quality metrics
 */
export interface DataQuality {
  /** Completeness score (0-1) */
  completeness: number;
  /** Accuracy score (0-1) */
  accuracy: number;
  /** Consistency score (0-1) */
  consistency: number;
  /** Timeliness score (0-1) */
  timeliness: number;
  /** Overall quality score (0-1) */
  overall: number;
  /** Issues found during validation */
  issues: string[];
}

/**
 * Data source configuration
 */
export interface DataSource {
  /** Name of the data source */
  name: string;
  /** Type of data source */
  type: 'rest' | 'websocket' | 'file' | 'database';
  /** Connection URL or path */
  url: string;
  /** Authentication credentials */
  credentials?: {
    apiKey?: string;
    secret?: string;
    token?: string;
  };
  /** Rate limiting configuration */
  rateLimit?: {
    requests: number;
    period: number; // in milliseconds
  };
  /** Retry configuration */
  retry?: {
    attempts: number;
    delay: number; // in milliseconds
    backoff: 'linear' | 'exponential';
  };
}

/**
 * Data validation result
 */
export interface ValidationResult {
  /** Whether the data is valid */
  isValid: boolean;
  /** Validation errors */
  errors: ValidationError[];
  /** Validation warnings */
  warnings: ValidationWarning[];
  /** Data quality metrics */
  quality?: DataQuality;
}

export interface ValidationError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Field that caused the error */
  field?: string;
  /** Value that caused the error */
  value?: unknown;
  /** Severity level */
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface ValidationWarning {
  /** Warning code */
  code: string;
  /** Warning message */
  message: string;
  /** Field that caused the warning */
  field?: string;
  /** Value that caused the warning */
  value?: unknown;
}

/**
 * Time series data structure
 */
export interface TimeSeries<T = number> {
  /** Timestamps */
  timestamps: Date[];
  /** Values */
  values: T[];
  /** Metadata */
  metadata?: {
    symbol?: string;
    interval?: string;
    source?: string;
  };
}

/**
 * Multi-asset data structure
 */
export interface MultiAssetData {
  /** Asset symbols */
  symbols: string[];
  /** Market data for each asset */
  data: Map<string, MarketData[]>;
  /** Correlation matrix */
  correlations?: number[][];
  /** Covariance matrix */
  covariances?: number[][];
}

/**
 * Data aggregation configuration
 */
export interface AggregationConfig {
  /** Aggregation method */
  method: 'ohlc' | 'vwap' | 'mean' | 'median' | 'sum';
  /** Time interval for aggregation */
  interval: string; // e.g., '1m', '5m', '1h', '1d'
  /** Timezone for aggregation */
  timezone?: string;
  /** Whether to include incomplete periods */
  includeIncomplete?: boolean;
}