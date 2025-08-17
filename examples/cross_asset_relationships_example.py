"""
Example usage of CrossAssetRelationshipModeler for analyzing relationships between assets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_timeseries.multi_asset.relationship_modeling import CrossAssetRelationshipModeler


def create_realistic_market_data(symbols: list, start_date: str = '2020-01-01', 
                               end_date: str = '2023-12-31') -> pd.DataFrame:
    """
    Create realistic market data with sector-based correlations and factor exposures.
    
    Args:
        symbols: List of asset symbols
        start_date: Start date for data generation
        end_date: End date for data generation
        
    Returns:
        DataFrame with realistic return data
    """
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Define market factors
    market_factor = np.random.normal(0.0005, 0.02, n_days)  # Broad market
    tech_factor = np.random.normal(0.0008, 0.03, n_days)    # Technology sector
    fin_factor = np.random.normal(0.0003, 0.025, n_days)    # Financial sector
    energy_factor = np.random.normal(0.0002, 0.035, n_days) # Energy sector
    util_factor = np.random.normal(0.0001, 0.015, n_days)   # Utilities sector
    
    # Define sector classifications and factor exposures
    sector_config = {
        # Technology stocks
        'AAPL': {'market': 0.8, 'tech': 0.9, 'idiosyncratic': 0.015},
        'GOOGL': {'market': 0.7, 'tech': 0.85, 'idiosyncratic': 0.018},
        'MSFT': {'market': 0.75, 'tech': 0.8, 'idiosyncratic': 0.016},
        'NVDA': {'market': 0.9, 'tech': 1.2, 'idiosyncratic': 0.025},
        'META': {'market': 0.8, 'tech': 0.95, 'idiosyncratic': 0.022},
        
        # Financial stocks
        'JPM': {'market': 0.6, 'fin': 0.9, 'idiosyncratic': 0.02},
        'BAC': {'market': 0.5, 'fin': 0.85, 'idiosyncratic': 0.022},
        'WFC': {'market': 0.55, 'fin': 0.8, 'idiosyncratic': 0.021},
        'GS': {'market': 0.7, 'fin': 1.0, 'idiosyncratic': 0.025},
        
        # Energy stocks
        'XOM': {'market': 0.4, 'energy': 0.8, 'idiosyncratic': 0.03},
        'CVX': {'market': 0.35, 'energy': 0.75, 'idiosyncratic': 0.028},
        'COP': {'market': 0.45, 'energy': 0.85, 'idiosyncratic': 0.032},
        
        # Utilities
        'NEE': {'market': 0.3, 'util': 0.7, 'idiosyncratic': 0.012},
        'DUK': {'market': 0.25, 'util': 0.65, 'idiosyncratic': 0.011},
        'SO': {'market': 0.28, 'util': 0.6, 'idiosyncratic': 0.01},
        
        # Consumer/Mixed
        'JNJ': {'market': 0.4, 'idiosyncratic': 0.014},  # Defensive
        'PG': {'market': 0.35, 'idiosyncratic': 0.013},   # Defensive
        'KO': {'market': 0.3, 'idiosyncratic': 0.012},    # Defensive
    }
    
    # Generate returns for each symbol
    returns_data = pd.DataFrame(index=dates)
    
    for symbol in symbols:
        if symbol not in sector_config:
            # Default configuration for unknown symbols
            config = {'market': 0.5, 'idiosyncratic': 0.02}
        else:
            config = sector_config[symbol]
        
        # Calculate return as weighted sum of factors
        returns = (
            config.get('market', 0) * market_factor +
            config.get('tech', 0) * tech_factor +
            config.get('fin', 0) * fin_factor +
            config.get('energy', 0) * energy_factor +
            config.get('util', 0) * util_factor +
            np.random.normal(0, config.get('idiosyncratic', 0.02), n_days)
        )
        
        # Add some regime changes
        if n_days > 500:
            # Market crash period (COVID-like)
            crash_start = n_days // 4
            crash_end = crash_start + 30
            returns[crash_start:crash_end] *= 2.5  # Increase volatility
            returns[crash_start:crash_start+5] -= 0.05  # Sharp decline
            
            # Recovery period
            recovery_start = crash_end
            recovery_end = recovery_start + 60
            returns[recovery_start:recovery_end] += np.linspace(0.01, 0.002, recovery_end - recovery_start)
        
        returns_data[symbol] = returns
    
    return returns_data


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str = "Asset Correlation Matrix"):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_sector_analysis(sector_assignments: dict, returns_data: pd.DataFrame):
    """Plot sector analysis results."""
    # Group assets by sector
    sectors = {}
    for asset, sector_info in sector_assignments.items():
        sector_name = sector_info['sector_name']
        if sector_name not in sectors:
            sectors[sector_name] = []
        sectors[sector_name].append(asset)
    
    # Calculate sector returns
    sector_returns = {}
    for sector_name, assets in sectors.items():
        if len(assets) > 0:
            sector_return = returns_data[assets].mean(axis=1)
            sector_returns[sector_name] = sector_return
    
    if not sector_returns:
        print("No sector data to plot")
        return
    
    # Plot cumulative sector performance
    plt.figure(figsize=(14, 8))
    
    for sector_name, returns in sector_returns.items():
        cumulative_returns = (1 + returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, 
                label=f"{sector_name} ({len(sectors[sector_name])} assets)", linewidth=2)
    
    plt.title('Cumulative Sector Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot sector characteristics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sector_names = list(sectors.keys())
    sector_stats = {}
    
    for sector_name in sector_names:
        assets = sectors[sector_name]
        if len(assets) > 0:
            sector_data = returns_data[assets]
            sector_stats[sector_name] = {
                'avg_return': sector_data.mean().mean() * 252,  # Annualized
                'volatility': sector_data.std().mean() * np.sqrt(252),  # Annualized
                'sharpe': (sector_data.mean().mean() / sector_data.std().mean()) * np.sqrt(252),
                'max_drawdown': ((sector_data.cumsum() - sector_data.cumsum().expanding().max()).min()).mean()
            }
    
    if sector_stats:
        # Average Return
        returns_vals = [stats['avg_return'] for stats in sector_stats.values()]
        axes[0, 0].bar(sector_names, returns_vals, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Annual Return by Sector')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Volatility
        vol_vals = [stats['volatility'] for stats in sector_stats.values()]
        axes[0, 1].bar(sector_names, vol_vals, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Annual Volatility by Sector')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sharpe Ratio
        sharpe_vals = [stats['sharpe'] for stats in sector_stats.values()]
        axes[1, 0].bar(sector_names, sharpe_vals, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Sharpe Ratio by Sector')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Max Drawdown
        dd_vals = [stats['max_drawdown'] for stats in sector_stats.values()]
        axes[1, 1].bar(sector_names, dd_vals, color='orange', alpha=0.7)
        axes[1, 1].set_title('Max Drawdown by Sector')
        axes[1, 1].set_ylabel('Max Drawdown')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def main():
    """Demonstrate cross-asset relationship modeling."""
    print("Cross-Asset Relationship Modeling Example")
    print("=" * 50)
    
    # Define asset universe
    symbols = [
        # Technology
        'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META',
        # Financial
        'JPM', 'BAC', 'WFC', 'GS',
        # Energy
        'XOM', 'CVX', 'COP',
        # Utilities
        'NEE', 'DUK', 'SO',
        # Consumer/Defensive
        'JNJ', 'PG', 'KO'
    ]
    
    print(f"Analyzing relationships for {len(symbols)} assets...")
    
    # Generate realistic market data
    returns_data = create_realistic_market_data(symbols, '2020-01-01', '2023-12-31')
    
    print(f"Generated {len(returns_data)} days of return data")
    print(f"Date range: {returns_data.index[0].date()} to {returns_data.index[-1].date()}")
    
    # Initialize relationship modeler
    modeler = CrossAssetRelationshipModeler(
        correlation_window=63,  # ~3 months
        min_correlation_periods=30,
        n_factors=5,
        max_sectors=8
    )
    
    print("\nPerforming comprehensive relationship analysis...")
    
    # Analyze cross-asset relationships
    results = modeler.analyze_cross_asset_relationships(returns_data)
    
    print("\nAnalysis Results Summary:")
    print("-" * 30)
    
    # Display correlation analysis
    corr_analysis = results.get('correlation_analysis', {})
    if 'error' not in corr_analysis:
        avg_corr = corr_analysis.get('avg_correlation', 0)
        n_pairs = corr_analysis.get('n_correlation_pairs', 0)
        print(f"Average correlation: {avg_corr:.3f}")
        print(f"Correlation pairs analyzed: {n_pairs}")
        
        # Plot correlation matrix
        corr_matrix = pd.DataFrame(corr_analysis.get('correlation_matrix', {}))
        if not corr_matrix.empty:
            plot_correlation_heatmap(corr_matrix, "Cross-Asset Correlation Matrix")
    
    # Display sector analysis
    sector_analysis = results.get('sector_analysis', {})
    if 'error' not in sector_analysis:
        n_sectors = sector_analysis.get('n_sectors', 0)
        sector_assignments = sector_analysis.get('sector_assignments', {})
        
        print(f"Sectors identified: {n_sectors}")
        
        # Show sector assignments
        if sector_assignments:
            print("\nSector Assignments:")
            sectors_summary = {}
            for asset, info in sector_assignments.items():
                sector_name = info['sector_name']
                confidence = info['confidence']
                if sector_name not in sectors_summary:
                    sectors_summary[sector_name] = []
                sectors_summary[sector_name].append((asset, confidence))
            
            for sector_name, assets in sectors_summary.items():
                asset_list = [f"{asset} ({conf:.2f})" for asset, conf in assets]
                print(f"  {sector_name}: {', '.join(asset_list)}")
            
            # Plot sector analysis
            plot_sector_analysis(sector_assignments, returns_data)
    
    # Display factor analysis
    factor_analysis = results.get('factor_analysis', {})
    if 'error' not in factor_analysis:
        avg_explained_var = factor_analysis.get('avg_explained_variance', 0)
        avg_stability = factor_analysis.get('avg_factor_stability', 0)
        n_factors = factor_analysis.get('n_factors', 0)
        
        print(f"\nFactor Analysis:")
        print(f"Number of factors: {n_factors}")
        print(f"Average explained variance: {avg_explained_var:.1%}")
        print(f"Average factor stability: {avg_stability:.3f}")
        
        # Show top factor exposures
        factor_exposures = factor_analysis.get('factor_exposures', {})
        if factor_exposures:
            print("\nTop Factor Exposures (Factor 1):")
            factor1_exposures = []
            for asset, exposure in factor_exposures.items():
                factor1_loading = exposure.get('factor_loadings', {}).get('factor_1', 0)
                factor1_exposures.append((asset, abs(factor1_loading)))
            
            factor1_exposures.sort(key=lambda x: x[1], reverse=True)
            for i, (asset, loading) in enumerate(factor1_exposures[:10]):
                print(f"  {i+1}. {asset}: {loading:.3f}")
    
    # Display adaptive weighting
    adaptive_analysis = results.get('adaptive_weighting', {})
    if 'error' not in adaptive_analysis:
        n_regime_changes = adaptive_analysis.get('n_regime_changes', 0)
        adaptive_weights = adaptive_analysis.get('adaptive_weights', {})
        
        print(f"\nAdaptive Weighting:")
        print(f"Correlation regime changes detected: {n_regime_changes}")
        
        if adaptive_weights:
            print(f"Adaptive weights calculated for {len(adaptive_weights)} target assets")
            
            # Show example adaptive weights for first asset
            first_asset = list(adaptive_weights.keys())[0]
            weights = adaptive_weights[first_asset]
            
            print(f"\nExample adaptive weights for {first_asset}:")
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for asset, weight in sorted_weights[:5]:
                print(f"  {asset}: {weight:.3f}")
    
    # Relationship summary
    print("\nRelationship Summary:")
    print("-" * 25)
    
    summary = modeler.get_relationship_summary()
    for key, value in summary.items():
        if key != 'last_update':
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Export results
    export_path = "temp_data/cross_asset_relationships.json"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    modeler.export_relationship_analysis(export_path)
    print(f"\nDetailed results exported to: {export_path}")
    
    # Advanced Analysis: Time-varying correlations
    print("\nAdvanced Analysis: Time-Varying Correlations")
    print("-" * 45)
    
    # Calculate rolling correlations for key pairs
    tech_pairs = [('AAPL', 'GOOGL'), ('AAPL', 'MSFT'), ('GOOGL', 'MSFT')]
    cross_sector_pairs = [('AAPL', 'JPM'), ('GOOGL', 'XOM'), ('MSFT', 'NEE')]
    
    print("Technology sector correlations (rolling 63-day):")
    for asset1, asset2 in tech_pairs:
        if asset1 in returns_data.columns and asset2 in returns_data.columns:
            rolling_corr = returns_data[asset1].rolling(63).corr(returns_data[asset2])
            avg_corr = rolling_corr.mean()
            corr_std = rolling_corr.std()
            print(f"  {asset1}-{asset2}: {avg_corr:.3f} ± {corr_std:.3f}")
    
    print("\nCross-sector correlations (rolling 63-day):")
    for asset1, asset2 in cross_sector_pairs:
        if asset1 in returns_data.columns and asset2 in returns_data.columns:
            rolling_corr = returns_data[asset1].rolling(63).corr(returns_data[asset2])
            avg_corr = rolling_corr.mean()
            corr_std = rolling_corr.std()
            print(f"  {asset1}-{asset2}: {avg_corr:.3f} ± {corr_std:.3f}")
    
    # Market stress analysis
    print("\nMarket Stress Analysis:")
    print("-" * 25)
    
    # Identify high volatility periods
    market_vol = returns_data.std(axis=1).rolling(21).mean()
    high_vol_threshold = market_vol.quantile(0.9)
    stress_periods = market_vol[market_vol > high_vol_threshold]
    
    print(f"High volatility periods identified: {len(stress_periods)}")
    if len(stress_periods) > 0:
        print(f"Average market volatility during stress: {stress_periods.mean():.4f}")
        print(f"Normal market volatility: {market_vol[market_vol <= high_vol_threshold].mean():.4f}")
        
        # Correlation during stress vs normal periods
        stress_dates = stress_periods.index
        normal_dates = market_vol[market_vol <= market_vol.quantile(0.5)].index
        
        if len(stress_dates) > 30 and len(normal_dates) > 30:
            stress_corr = returns_data.loc[stress_dates].corr().values
            normal_corr = returns_data.loc[normal_dates].corr().values
            
            # Average correlation (excluding diagonal)
            stress_avg = stress_corr[np.triu_indices_from(stress_corr, k=1)].mean()
            normal_avg = normal_corr[np.triu_indices_from(normal_corr, k=1)].mean()
            
            print(f"Average correlation during stress: {stress_avg:.3f}")
            print(f"Average correlation during normal periods: {normal_avg:.3f}")
            print(f"Correlation increase during stress: {stress_avg - normal_avg:.3f}")
    
    print("\nCross-asset relationship analysis completed!")
    print("\nKey Insights:")
    print("  ✓ Comprehensive correlation analysis across all asset pairs")
    print("  ✓ Sector classification based on return patterns")
    print("  ✓ Factor decomposition revealing market drivers")
    print("  ✓ Adaptive weighting system for dynamic relationships")
    print("  ✓ Regime change detection for correlation shifts")
    print("  ✓ Market stress analysis and correlation dynamics")


if __name__ == "__main__":
    main()