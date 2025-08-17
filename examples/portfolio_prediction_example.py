"""
Example usage of PortfolioPredictionSystem for portfolio optimization and asset ranking.
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

from enhanced_timeseries.multi_asset.portfolio_prediction import (
    AssetPrediction, PortfolioPredictionSystem
)


def create_sample_asset_predictions() -> dict:
    """Create sample asset predictions for demonstration."""
    
    # Define asset universe with different characteristics
    assets_config = {
        # Technology stocks - high growth, high volatility
        'AAPL': {'return': 0.15, 'std': 0.25, 'confidence': 0.85, 'sector': 'Technology'},
        'GOOGL': {'return': 0.18, 'std': 0.28, 'confidence': 0.80, 'sector': 'Technology'},
        'MSFT': {'return': 0.14, 'std': 0.22, 'confidence': 0.90, 'sector': 'Technology'},
        'NVDA': {'return': 0.25, 'std': 0.40, 'confidence': 0.70, 'sector': 'Technology'},
        
        # Financial stocks - moderate growth, moderate volatility
        'JPM': {'return': 0.10, 'std': 0.20, 'confidence': 0.85, 'sector': 'Financial'},
        'BAC': {'return': 0.08, 'std': 0.22, 'confidence': 0.80, 'sector': 'Financial'},
        'WFC': {'return': 0.07, 'std': 0.24, 'confidence': 0.75, 'sector': 'Financial'},
        
        # Healthcare - defensive, lower volatility
        'JNJ': {'return': 0.08, 'std': 0.15, 'confidence': 0.95, 'sector': 'Healthcare'},
        'PFE': {'return': 0.06, 'std': 0.18, 'confidence': 0.90, 'sector': 'Healthcare'},
        
        # Energy - cyclical, high volatility
        'XOM': {'return': 0.05, 'std': 0.30, 'confidence': 0.65, 'sector': 'Energy'},
        'CVX': {'return': 0.06, 'std': 0.28, 'confidence': 0.70, 'sector': 'Energy'},
        
        # Utilities - defensive, low volatility
        'NEE': {'return': 0.07, 'std': 0.12, 'confidence': 0.88, 'sector': 'Utilities'},
        'DUK': {'return': 0.06, 'std': 0.14, 'confidence': 0.85, 'sector': 'Utilities'},
        
        # Consumer goods
        'PG': {'return': 0.08, 'std': 0.16, 'confidence': 0.92, 'sector': 'Consumer'},
        'KO': {'return': 0.06, 'std': 0.15, 'confidence': 0.90, 'sector': 'Consumer'},
    }
    
    asset_predictions = {}
    
    for symbol, config in assets_config.items():
        # Create factor exposures based on sector
        if config['sector'] == 'Technology':
            factor_exposure = {'market': 0.9, 'tech': 1.2, 'growth': 0.8}
        elif config['sector'] == 'Financial':
            factor_exposure = {'market': 1.1, 'financial': 1.0, 'value': 0.6}
        elif config['sector'] == 'Healthcare':
            factor_exposure = {'market': 0.7, 'healthcare': 0.8, 'quality': 0.9}
        elif config['sector'] == 'Energy':
            factor_exposure = {'market': 0.8, 'energy': 1.3, 'value': 1.0}
        elif config['sector'] == 'Utilities':
            factor_exposure = {'market': 0.5, 'utilities': 0.7, 'quality': 0.8}
        else:  # Consumer
            factor_exposure = {'market': 0.6, 'consumer': 0.8, 'quality': 0.7}
        
        # Add some noise to make it more realistic
        np.random.seed(hash(symbol) % 2**32)
        return_noise = np.random.normal(0, 0.02)
        confidence_noise = np.random.normal(0, 0.05)
        
        asset_predictions[symbol] = AssetPrediction(
            symbol=symbol,
            expected_return=config['return'] + return_noise,
            return_std=config['std'],
            confidence_score=max(0.5, min(0.99, config['confidence'] + confidence_noise)),
            prediction_horizon=252,  # 1 year
            model_ensemble_agreement=config['confidence'] * 0.9,  # Slightly lower than confidence
            factor_exposure=factor_exposure,
            sector=config['sector']
        )
    
    return asset_predictions


def create_historical_data(asset_predictions: dict, years: int = 3) -> pd.DataFrame:
    """Create realistic historical data based on asset predictions."""
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    historical_data = pd.DataFrame(index=dates)
    
    # Generate market factors
    np.random.seed(42)
    n_days = len(dates)
    
    market_factor = np.random.normal(0.0005, 0.015, n_days)
    tech_factor = np.random.normal(0.0008, 0.025, n_days)
    financial_factor = np.random.normal(0.0003, 0.020, n_days)
    
    for symbol, prediction in asset_predictions.items():
        # Generate returns based on factor exposures and prediction characteristics
        sector = prediction.sector
        
        # Base return from prediction
        base_return = prediction.expected_return / 252  # Daily return
        
        # Factor contributions
        if sector == 'Technology':
            factor_return = (0.8 * market_factor + 0.6 * tech_factor)
        elif sector == 'Financial':
            factor_return = (0.9 * market_factor + 0.7 * financial_factor)
        else:
            factor_return = 0.6 * market_factor
        
        # Idiosyncratic returns
        idiosyncratic_vol = prediction.return_std / np.sqrt(252) * 0.7  # Daily vol
        idiosyncratic_returns = np.random.normal(0, idiosyncratic_vol, n_days)
        
        # Combine all return sources
        total_returns = base_return + factor_return + idiosyncratic_returns
        
        # Add some regime changes and market events
        if n_days > 500:
            # Market crash (like COVID)
            crash_start = n_days // 3
            crash_end = crash_start + 20
            total_returns[crash_start:crash_end] *= 3  # Increase volatility
            total_returns[crash_start:crash_start+5] -= 0.05  # Sharp decline
            
            # Recovery period
            recovery_start = crash_end
            recovery_end = recovery_start + 40
            total_returns[recovery_start:recovery_end] += np.linspace(0.01, 0.002, recovery_end - recovery_start)
        
        # Generate prices from returns
        prices = 100 * np.cumprod(1 + total_returns)
        historical_data[symbol] = prices
    
    return historical_data


def plot_asset_rankings(rankings: list):
    """Plot asset ranking results."""
    
    # Extract data for plotting
    symbols = [r['symbol'] for r in rankings]
    composite_scores = [r['composite_score'] for r in rankings]
    return_scores = [r['ranking_factors']['return_score'] for r in rankings]
    risk_adj_scores = [r['ranking_factors']['risk_adjusted_score'] for r in rankings]
    confidence_scores = [r['ranking_factors']['confidence_score'] for r in rankings]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Composite scores
    axes[0, 0].barh(symbols, composite_scores, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Composite Asset Rankings', fontweight='bold')
    axes[0, 0].set_xlabel('Composite Score')
    
    # Return scores
    axes[0, 1].barh(symbols, return_scores, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Expected Return Scores', fontweight='bold')
    axes[0, 1].set_xlabel('Return Score')
    
    # Risk-adjusted scores
    axes[1, 0].barh(symbols, risk_adj_scores, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Risk-Adjusted Return Scores', fontweight='bold')
    axes[1, 0].set_xlabel('Risk-Adjusted Score')
    
    # Confidence scores
    axes[1, 1].barh(symbols, confidence_scores, color='orange', alpha=0.7)
    axes[1, 1].set_title('Prediction Confidence Scores', fontweight='bold')
    axes[1, 1].set_xlabel('Confidence Score')
    
    plt.tight_layout()
    plt.show()


def plot_portfolio_composition(weights: dict, asset_predictions: dict):
    """Plot portfolio composition by sector and individual assets."""
    
    # Calculate sector allocations
    sector_weights = {}
    for symbol, weight in weights.items():
        if symbol in asset_predictions:
            sector = asset_predictions[symbol].sector
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sector allocation pie chart
    if sector_weights:
        ax1.pie(sector_weights.values(), labels=sector_weights.keys(), autopct='%1.1f%%', 
                startangle=90, colors=plt.cm.Set3.colors)
        ax1.set_title('Portfolio Allocation by Sector', fontweight='bold')
    
    # Individual asset weights
    symbols = list(weights.keys())
    asset_weights = list(weights.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(symbols)))
    bars = ax2.bar(range(len(symbols)), asset_weights, color=colors, alpha=0.7)
    ax2.set_title('Individual Asset Weights', fontweight='bold')
    ax2.set_xlabel('Assets')
    ax2.set_ylabel('Weight')
    ax2.set_xticks(range(len(symbols)))
    ax2.set_xticklabels(symbols, rotation=45)
    
    # Add weight labels on bars
    for bar, weight in zip(bars, asset_weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_efficient_frontier_simulation(system: PortfolioPredictionSystem, 
                                     asset_predictions: dict, 
                                     historical_data: pd.DataFrame):
    """Simulate and plot efficient frontier."""
    
    returns = []
    volatilities = []
    sharpe_ratios = []
    
    # Test different risk aversion levels
    risk_aversions = np.linspace(0.1, 5.0, 20)
    
    for risk_aversion in risk_aversions:
        # Temporarily change risk aversion
        original_risk_aversion = system.portfolio_optimizer.risk_aversion
        system.portfolio_optimizer.risk_aversion = risk_aversion
        
        # Generate portfolio prediction
        result = system.generate_portfolio_prediction(
            asset_predictions=asset_predictions,
            historical_data=historical_data
        )
        
        if 'portfolio_optimization' in result:
            opt_result = result['portfolio_optimization']
            returns.append(opt_result['expected_return'])
            volatilities.append(opt_result['expected_volatility'])
            sharpe_ratios.append(opt_result['sharpe_ratio'])
        
        # Restore original risk aversion
        system.portfolio_optimizer.risk_aversion = original_risk_aversion
    
    if returns and volatilities:
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, 
                            cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Simulated Efficient Frontier', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Highlight maximum Sharpe ratio portfolio
        if sharpe_ratios:
            max_sharpe_idx = np.argmax(sharpe_ratios)
            plt.scatter(volatilities[max_sharpe_idx], returns[max_sharpe_idx], 
                       color='red', s=100, marker='*', label='Max Sharpe')
            plt.legend()
        
        plt.show()


def main():
    """Demonstrate portfolio prediction and asset ranking system."""
    print("Portfolio Prediction and Asset Ranking Example")
    print("=" * 55)
    
    # Create sample asset predictions
    print("Creating sample asset predictions...")
    asset_predictions = create_sample_asset_predictions()
    
    print(f"Generated predictions for {len(asset_predictions)} assets:")
    for symbol, pred in list(asset_predictions.items())[:5]:  # Show first 5
        print(f"  {symbol}: Expected Return = {pred.expected_return:.1%}, "
              f"Volatility = {pred.return_std:.1%}, "
              f"Confidence = {pred.confidence_score:.2f}")
    print("  ...")
    
    # Create historical data
    print("\nGenerating historical market data...")
    historical_data = create_historical_data(asset_predictions, years=3)
    
    print(f"Generated {len(historical_data)} days of historical data")
    print(f"Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
    
    # Initialize portfolio prediction system
    print("\nInitializing portfolio prediction system...")
    system = PortfolioPredictionSystem(
        covariance_method='ledoit_wolf',
        optimization_objective='max_sharpe',
        max_position_size=0.20,  # Max 20% per asset
        min_position_size=0.02,  # Min 2% per asset
        rebalancing_threshold=0.05
    )
    
    # Generate comprehensive portfolio prediction
    print("\nGenerating portfolio prediction...")
    result = system.generate_portfolio_prediction(
        asset_predictions=asset_predictions,
        historical_data=historical_data
    )
    
    # Display results
    print("\nPortfolio Prediction Results:")
    print("-" * 35)
    
    # Asset Rankings
    if 'asset_rankings' in result:
        rankings = result['asset_rankings']
        print(f"\nAsset Rankings (Top 10):")
        print("Rank | Symbol | Composite Score | Sector")
        print("-" * 45)
        
        for i, ranking in enumerate(rankings[:10]):
            symbol = ranking['symbol']
            score = ranking['composite_score']
            sector = asset_predictions[symbol].sector if symbol in asset_predictions else 'Unknown'
            print(f"{i+1:4d} | {symbol:6s} | {score:13.3f} | {sector}")
        
        # Plot rankings
        plot_asset_rankings(rankings)
    
    # Portfolio Optimization
    if 'portfolio_optimization' in result:
        opt_result = result['portfolio_optimization']
        
        print(f"\nOptimal Portfolio:")
        print(f"Expected Annual Return: {opt_result['expected_return']:.1%}")
        print(f"Expected Annual Volatility: {opt_result['expected_volatility']:.1%}")
        print(f"Sharpe Ratio: {opt_result['sharpe_ratio']:.2f}")
        
        weights = opt_result['optimal_weights']
        print(f"\nTop 10 Holdings:")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for symbol, weight in sorted_weights[:10]:
            sector = asset_predictions[symbol].sector if symbol in asset_predictions else 'Unknown'
            print(f"  {symbol}: {weight:.1%} ({sector})")
        
        # Plot portfolio composition
        plot_portfolio_composition(weights, asset_predictions)
    
    # Portfolio Prediction Metrics
    if 'portfolio_prediction' in result:
        portfolio_pred = result['portfolio_prediction']
        
        print(f"\nPortfolio Risk Metrics:")
        print(f"Maximum Drawdown Estimate: {portfolio_pred['max_drawdown_estimate']:.1%}")
        print(f"Value at Risk (95%): {portfolio_pred['var_95']:.1%}")
        print(f"Conditional VaR (95%): {portfolio_pred['cvar_95']:.1%}")
        print(f"Portfolio Beta: {portfolio_pred['portfolio_beta']:.2f}")
        print(f"Diversification Ratio: {portfolio_pred['diversification_ratio']:.2f}")
        print(f"Concentration Risk: {portfolio_pred['concentration_risk']:.3f}")
        print(f"Prediction Confidence: {portfolio_pred['prediction_confidence']:.2f}")
    
    # Risk Analysis
    if 'risk_analysis' in result:
        risk_analysis = result['risk_analysis']
        
        print(f"\nRisk Analysis:")
        
        # Sector exposure
        if 'sector_exposure' in risk_analysis:
            print("Sector Exposure:")
            for sector, exposure in risk_analysis['sector_exposure'].items():
                print(f"  {sector}: {exposure:.1%}")
        
        # Risk contributions
        if 'risk_contributions' in risk_analysis:
            print("\nTop Risk Contributors:")
            risk_contribs = risk_analysis['risk_contributions']
            sorted_contribs = sorted(risk_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for symbol, contrib in sorted_contribs[:5]:
                print(f"  {symbol}: {contrib:.1%}")
        
        # Tail risk metrics
        tail_metrics = ['skewness', 'kurtosis', 'worst_day', 'best_day', 'downside_deviation']
        print("\nTail Risk Metrics:")
        for metric in tail_metrics:
            if metric in risk_analysis:
                value = risk_analysis[metric]
                print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
    
    # Position Sizing Recommendations
    if 'position_sizing' in result:
        pos_sizing = result['position_sizing']
        
        print(f"\nPosition Sizing Adjustments:")
        
        if 'correlation_adjustments' in pos_sizing:
            print("Correlation-based adjustments:")
            corr_adj = pos_sizing['correlation_adjustments']
            for symbol, adj in list(corr_adj.items())[:5]:
                print(f"  {symbol}: {adj:.2f}x")
        
        if 'confidence_adjustments' in pos_sizing:
            print("Confidence-based adjustments:")
            conf_adj = pos_sizing['confidence_adjustments']
            for symbol, adj in list(conf_adj.items())[:5]:
                print(f"  {symbol}: {adj:.2f}x")
    
    # Demonstrate rebalancing recommendations
    print(f"\nRebalancing Analysis:")
    
    # Simulate current portfolio (slightly different from optimal)
    if 'portfolio_optimization' in result:
        optimal_weights = pd.Series(result['portfolio_optimization']['optimal_weights'])
        
        # Create a "current" portfolio with some drift
        np.random.seed(123)
        current_weights = optimal_weights * (1 + np.random.normal(0, 0.1, len(optimal_weights)))
        current_weights = current_weights / current_weights.sum()  # Renormalize
        
        rebalancing_recs = system.get_rebalancing_recommendations(current_weights, optimal_weights)
        
        print(f"Rebalancing needed: {rebalancing_recs['rebalancing_needed']}")
        print(f"Total turnover: {rebalancing_recs['total_turnover']:.1%}")
        
        if rebalancing_recs['assets_to_buy']:
            print("Assets to buy:")
            for symbol, weight_diff in list(rebalancing_recs['assets_to_buy'].items())[:3]:
                print(f"  {symbol}: +{weight_diff:.1%}")
        
        if rebalancing_recs['assets_to_sell']:
            print("Assets to sell:")
            for symbol, weight_diff in list(rebalancing_recs['assets_to_sell'].items())[:3]:
                print(f"  {symbol}: {weight_diff:.1%}")
    
    # Advanced Analysis: Efficient Frontier
    print(f"\nGenerating efficient frontier simulation...")
    plot_efficient_frontier_simulation(system, asset_predictions, historical_data)
    
    # Export results
    export_path = "temp_data/portfolio_prediction_results.json"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    system.export_portfolio_analysis(export_path)
    print(f"\nDetailed results exported to: {export_path}")
    
    # Performance comparison by sector
    print(f"\nSector Performance Analysis:")
    print("-" * 30)
    
    sector_performance = {}
    for symbol, prediction in asset_predictions.items():
        sector = prediction.sector
        if sector not in sector_performance:
            sector_performance[sector] = {
                'returns': [],
                'volatilities': [],
                'confidences': [],
                'count': 0
            }
        
        sector_performance[sector]['returns'].append(prediction.expected_return)
        sector_performance[sector]['volatilities'].append(prediction.return_std)
        sector_performance[sector]['confidences'].append(prediction.confidence_score)
        sector_performance[sector]['count'] += 1
    
    for sector, metrics in sector_performance.items():
        avg_return = np.mean(metrics['returns'])
        avg_vol = np.mean(metrics['volatilities'])
        avg_confidence = np.mean(metrics['confidences'])
        sharpe = avg_return / avg_vol if avg_vol > 0 else 0
        
        print(f"{sector}:")
        print(f"  Assets: {metrics['count']}")
        print(f"  Avg Expected Return: {avg_return:.1%}")
        print(f"  Avg Volatility: {avg_vol:.1%}")
        print(f"  Avg Confidence: {avg_confidence:.2f}")
        print(f"  Sector Sharpe: {sharpe:.2f}")
    
    print("\nPortfolio prediction and optimization completed!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Multi-factor asset ranking system")
    print("  ✓ Portfolio optimization with risk constraints")
    print("  ✓ Comprehensive risk analysis and metrics")
    print("  ✓ Correlation-aware position sizing")
    print("  ✓ Rebalancing recommendations")
    print("  ✓ Sector-based analysis and diversification")
    print("  ✓ Efficient frontier simulation")
    print("  ✓ Advanced portfolio risk metrics (VaR, CVaR, etc.)")


if __name__ == "__main__":
    main()