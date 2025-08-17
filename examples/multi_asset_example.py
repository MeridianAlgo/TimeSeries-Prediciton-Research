"""
Example usage of MultiAssetDataCoordinator for batch processing multiple assets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_timeseries.multi_asset.data_coordinator import MultiAssetDataCoordinator


def create_sample_data(symbol: str, **kwargs) -> pd.DataFrame:
    """
    Create sample market data for demonstration.
    
    Args:
        symbol: Asset symbol
        **kwargs: Additional parameters
        
    Returns:
        Sample market data DataFrame
    """
    # Create different characteristics for different symbols
    np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
    
    start_date = kwargs.get('start_date', '2020-01-01')
    end_date = kwargs.get('end_date', '2023-12-31')
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Base price varies by symbol
    base_price = 50 + (hash(symbol) % 200)
    
    # Generate price data with different volatilities
    if 'TECH' in symbol or symbol in ['AAPL', 'GOOGL', 'MSFT']:
        volatility = 0.025  # Higher volatility for tech stocks
        trend = 0.0008  # Positive trend
    elif 'UTIL' in symbol or symbol in ['XEL', 'NEE']:
        volatility = 0.015  # Lower volatility for utilities
        trend = 0.0003  # Modest trend
    else:
        volatility = 0.02  # Medium volatility
        trend = 0.0005  # Medium trend
    
    # Generate returns
    returns = np.random.normal(trend, volatility, n_days)
    
    # Add some regime changes
    if n_days > 500:
        # Bear market period
        bear_start = n_days // 3
        bear_end = bear_start + 100
        returns[bear_start:bear_end] = np.random.normal(-0.002, volatility * 1.5, bear_end - bear_start)
        
        # High volatility period
        vol_start = 2 * n_days // 3
        vol_end = vol_start + 80
        returns[vol_start:vol_end] = np.random.normal(trend, volatility * 2, vol_end - vol_start)
    
    # Generate prices
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    
    # Generate Open, High, Low based on Close
    daily_range = np.random.uniform(0.005, 0.03, n_days)  # Daily range as % of close
    
    data['Open'] = data['Close'] * (1 + np.random.uniform(-0.01, 0.01, n_days))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + daily_range * np.random.uniform(0.3, 1.0, n_days))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - daily_range * np.random.uniform(0.3, 1.0, n_days))
    
    # Generate volume (higher volume on high volatility days)
    base_volume = 1000000 + (hash(symbol) % 5000000)
    volume_multiplier = 1 + 2 * np.abs(returns)  # Higher volume on big moves
    data['Volume'] = (base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, n_days)).astype(int)
    
    return data


def main():
    """Demonstrate multi-asset data coordinator functionality."""
    print("Multi-Asset Data Coordinator Example")
    print("=" * 50)
    
    # Initialize coordinator
    coordinator = MultiAssetDataCoordinator(
        storage_path="temp_data/multi_asset_demo",
        max_assets=20,
        alignment_method='inner',
        max_workers=4,
        batch_size=5
    )
    
    # Define asset universe
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    financial_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
    utility_stocks = ['XEL', 'NEE', 'DUK']
    
    all_symbols = tech_stocks + financial_stocks + utility_stocks
    
    print(f"Loading data for {len(all_symbols)} assets...")
    
    # Add assets to coordinator
    results = coordinator.add_assets(
        all_symbols, 
        create_sample_data,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Report loading results
    successful_loads = sum(results.values())
    print(f"Successfully loaded {successful_loads}/{len(all_symbols)} assets")
    
    for symbol, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {symbol}")
    
    print("\nAsset Statistics:")
    print("-" * 30)
    
    # Get asset statistics
    stats = coordinator.get_asset_statistics()
    
    for symbol in sorted(stats.keys()):
        asset_stats = stats[symbol]
        print(f"{symbol}:")
        print(f"  Data points: {asset_stats['data_points']:,}")
        print(f"  Date range: {asset_stats['date_range']['start'][:10]} to {asset_stats['date_range']['end'][:10]}")
        print(f"  Data quality: {asset_stats['data_quality_score']:.3f}")
        print(f"  Missing data: {asset_stats['missing_data_pct']:.1%}")
    
    print("\nSynchronizing assets...")
    
    # Synchronize all assets
    sync_status = coordinator.synchronize_all_assets(fill_method='forward')
    
    print(f"Synchronization Results:")
    print(f"  Total assets: {sync_status.total_assets}")
    print(f"  Synchronized: {sync_status.synchronized_assets}")
    print(f"  Success rate: {sync_status.sync_percentage:.1f}%")
    print(f"  Common date range: {sync_status.common_start_date.date()} to {sync_status.common_end_date.date()}")
    print(f"  Alignment method: {sync_status.alignment_method}")
    
    print("\nCreating combined dataset...")
    
    # Get synchronized DataFrame
    combined_df = coordinator.get_synchronized_dataframe(['Close', 'Volume'])
    
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns: {len(combined_df.columns)} (symbols × metrics)")
    print(f"Date range: {combined_df.index[0].date()} to {combined_df.index[-1].date()}")
    
    # Calculate cross-asset correlations
    print("\nCalculating cross-asset correlations...")
    
    # Extract close prices for correlation analysis
    close_prices = combined_df.xs('Close', level=1, axis=1)
    correlation_matrix = close_prices.corr()
    
    print("Top 5 most correlated pairs:")
    
    # Find top correlations (excluding self-correlations)
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            symbol1 = correlation_matrix.columns[i]
            symbol2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]
            corr_pairs.append((symbol1, symbol2, corr))
    
    # Sort by correlation and show top 5
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i, (sym1, sym2, corr) in enumerate(corr_pairs[:5]):
        print(f"  {i+1}. {sym1} - {sym2}: {corr:.3f}")
    
    # Memory usage analysis
    print("\nMemory Usage Analysis:")
    print("-" * 25)
    
    memory_stats = coordinator.get_memory_usage()
    print(f"Total memory usage: {memory_stats['total_memory_mb']:.1f} MB")
    print(f"Average per asset: {memory_stats['avg_memory_per_asset_mb']:.1f} MB")
    print(f"Number of assets: {memory_stats['asset_count']}")
    
    print("\nTop 5 memory consumers:")
    memory_breakdown = memory_stats['asset_memory_breakdown']
    sorted_memory = sorted(memory_breakdown.items(), key=lambda x: x[1], reverse=True)
    
    for i, (symbol, memory_mb) in enumerate(sorted_memory[:5]):
        print(f"  {i+1}. {symbol}: {memory_mb:.1f} MB")
    
    # Export coordinator state
    print("\nExporting coordinator state...")
    
    export_path = "temp_data/coordinator_state.json"
    coordinator.export_coordinator_state(export_path)
    print(f"State exported to: {export_path}")
    
    # Demonstrate sector analysis
    print("\nSector Analysis:")
    print("-" * 20)
    
    sectors = {
        'Technology': tech_stocks,
        'Financial': financial_stocks,
        'Utilities': utility_stocks
    }
    
    for sector_name, sector_symbols in sectors.items():
        # Get sector data
        sector_data = []
        for symbol in sector_symbols:
            if symbol in close_prices.columns:
                sector_data.append(close_prices[symbol])
        
        if sector_data:
            sector_df = pd.DataFrame(sector_data).T
            sector_returns = sector_df.pct_change().dropna()
            
            avg_return = sector_returns.mean().mean() * 252  # Annualized
            avg_volatility = sector_returns.std().mean() * np.sqrt(252)  # Annualized
            
            print(f"{sector_name}:")
            print(f"  Assets: {len(sector_data)}")
            print(f"  Avg Annual Return: {avg_return:.1%}")
            print(f"  Avg Annual Volatility: {avg_volatility:.1%}")
            
            # Sector correlation
            if len(sector_data) > 1:
                sector_corr = sector_returns.corr().values
                avg_intra_corr = np.mean(sector_corr[np.triu_indices_from(sector_corr, k=1)])
                print(f"  Avg Intra-Sector Correlation: {avg_intra_corr:.3f}")
    
    print("\nExample completed successfully!")
    print("The multi-asset coordinator has demonstrated:")
    print("  ✓ Batch loading of multiple assets")
    print("  ✓ Data synchronization and alignment")
    print("  ✓ Memory-efficient storage and retrieval")
    print("  ✓ Cross-asset correlation analysis")
    print("  ✓ Sector-based analysis")
    print("  ✓ Performance monitoring and statistics")


if __name__ == "__main__":
    main()