"""
Comprehensive performance reporting and statistical analysis for backtesting results.
Implements detailed analytics, visualizations, and statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest
import warnings
from .regime_analysis import RegimePerformanceTracker, MarketRegime
from ..utils.math_utils import StatisticalUtils, PerformanceUtils
warnings.filterwarnings('ignore')


@dataclass
class StatisticalTest:
    """Results from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    is_significant: bool
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    summary_metrics: Dict[str, float]
    regime_analysis: Dict[str, Any]
    statistical_tests: List[StatisticalTest]
    risk_metrics: Dict[str, float]
    consistency_metrics: Dict[str, float]
    benchmark_comparison: Optional[Dict[str, float]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'summary_metrics': self.summary_metrics,
            'regime_analysis': self.regime_analysis,
            'statistical_tests': [test.to_dict() for test in self.statistical_tests],
            'risk_metrics': self.risk_metrics,
            'consistency_metrics': self.consistency_metrics,
            'benchmark_comparison': self.benchmark_comparison,
            'recommendations': self.recommendations
        }


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and reporting system.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def analyze_backtest_results(self, results_df: pd.DataFrame,
                                regime_tracker: Optional[RegimePerformanceTracker] = None,
                                benchmark_returns: Optional[pd.Series] = None) -> PerformanceReport:
        """
        Comprehensive analysis of backtest results.
        
        Args:
            results_df: DataFrame with backtest results
            regime_tracker: Optional regime performance tracker
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Comprehensive performance report
        """
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(results_df)
        
        # Regime analysis
        regime_analysis = {}
        if regime_tracker:
            regime_analysis = regime_tracker.get_performance_summary()
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(results_df)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(results_df)
        
        # Consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(results_df)
        
        # Benchmark comparison
        benchmark_comparison = None
        if benchmark_returns is not None:
            benchmark_comparison = self._compare_to_benchmark(results_df, benchmark_returns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            summary_metrics, risk_metrics, consistency_metrics, statistical_tests
        )
        
        return PerformanceReport(
            summary_metrics=summary_metrics,
            regime_analysis=regime_analysis,
            statistical_tests=statistical_tests,
            risk_metrics=risk_metrics,
            consistency_metrics=consistency_metrics,
            benchmark_comparison=benchmark_comparison,
            recommendations=recommendations
        )
    
    def _calculate_summary_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive summary metrics."""   
        if results_df.empty:
            return {}
        
        predictions = results_df['prediction'].values
        actuals = results_df['actual'].values
        errors = results_df['error'].values
        
        # Basic accuracy metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(results_df['squared_error'].values))
        mape = np.mean(np.abs(errors / (np.abs(actuals) + 1e-8))) * 100
        
        # Directional accuracy
        directional_accuracy = np.mean(results_df['directional_correct'].values) * 100
        
        # Correlation metrics
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        # Financial metrics (treating predictions as returns)
        returns = predictions
        
        # Return statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        annualized_return = mean_return * 252
        annualized_volatility = std_return * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        max_drawdown = StatisticalUtils.calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win/loss statistics
        win_rate = np.mean(returns > 0)
        loss_rate = np.mean(returns < 0)
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        es_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
        
        return {
            # Accuracy metrics
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            
            # Return statistics
            'mean_return': mean_return,
            'annualized_return': annualized_return,
            'volatility': std_return,
            'annualized_volatility': annualized_volatility,
            
            # Risk-adjusted returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Drawdown metrics
            'max_drawdown': max_drawdown,
            
            # Win/loss metrics
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'profit_factor': profit_factor,
            
            # Distribution metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            
            # Risk metrics
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            
            # Sample statistics
            'n_predictions': len(predictions),
            'n_periods': len(results_df['period_id'].unique()) if 'period_id' in results_df.columns else 1
        }
    
    def _perform_statistical_tests(self, results_df: pd.DataFrame) -> List[StatisticalTest]:
        """Perform comprehensive statistical tests."""
        if results_df.empty:
            return []
        
        tests = []
        predictions = results_df['prediction'].values
        actuals = results_df['actual'].values
        errors = results_df['error'].values
        
        # 1. Normality test for errors (Jarque-Bera)
        if len(errors) > 7:  # Minimum sample size for JB test
            try:
                jb_stat, jb_p = jarque_bera(errors)
                tests.append(StatisticalTest(
                    test_name="Jarque-Bera Normality Test (Errors)",
                    statistic=jb_stat,
                    p_value=jb_p,
                    critical_value=None,
                    is_significant=jb_p < self.significance_level,
                    interpretation="Errors are NOT normally distributed" if jb_p < self.significance_level 
                                 else "Errors appear normally distributed"
                ))
            except:
                pass
        
        # 2. Shapiro-Wilk test for small samples
        if 3 <= len(errors) <= 5000:
            try:
                sw_stat, sw_p = shapiro(errors)
                tests.append(StatisticalTest(
                    test_name="Shapiro-Wilk Normality Test (Errors)",
                    statistic=sw_stat,
                    p_value=sw_p,
                    critical_value=None,
                    is_significant=sw_p < self.significance_level,
                    interpretation="Errors are NOT normally distributed" if sw_p < self.significance_level 
                                 else "Errors appear normally distributed"
                ))
            except:
                pass
        
        # 3. One-sample t-test for zero mean errors
        if len(errors) > 1:
            try:
                t_stat, t_p = stats.ttest_1samp(errors, 0)
                tests.append(StatisticalTest(
                    test_name="One-Sample t-test (Zero Mean Errors)",
                    statistic=t_stat,
                    p_value=t_p,
                    critical_value=None,
                    is_significant=t_p < self.significance_level,
                    interpretation="Errors have significant bias" if t_p < self.significance_level 
                                 else "Errors are unbiased"
                ))
            except:
                pass
        
        # 4. Ljung-Box test for autocorrelation in errors
        if len(errors) > 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(errors, lags=min(10, len(errors)//4), return_df=True)
                lb_stat = lb_result['lb_stat'].iloc[-1]
                lb_p = lb_result['lb_pvalue'].iloc[-1]
                
                tests.append(StatisticalTest(
                    test_name="Ljung-Box Test (Error Autocorrelation)",
                    statistic=lb_stat,
                    p_value=lb_p,
                    critical_value=None,
                    is_significant=lb_p < self.significance_level,
                    interpretation="Errors show significant autocorrelation" if lb_p < self.significance_level 
                                 else "Errors appear independent"
                ))
            except:
                pass
        
        # 5. Kolmogorov-Smirnov test for uniform distribution of p-values
        if len(predictions) > 1 and len(actuals) > 1:
            try:
                ks_stat, ks_p = kstest(predictions, actuals)
                tests.append(StatisticalTest(
                    test_name="Kolmogorov-Smirnov Test (Predictions vs Actuals)",
                    statistic=ks_stat,
                    p_value=ks_p,
                    critical_value=None,
                    is_significant=ks_p < self.significance_level,
                    interpretation="Predictions and actuals have different distributions" if ks_p < self.significance_level 
                                 else "Predictions and actuals have similar distributions"
                ))
            except:
                pass
        
        # 6. Runs test for randomness
        if len(errors) > 10:
            try:
                median_error = np.median(errors)
                runs, n1, n2 = self._runs_test(errors > median_error)
                expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
                z_stat = (runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0
                runs_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                tests.append(StatisticalTest(
                    test_name="Runs Test (Error Randomness)",
                    statistic=z_stat,
                    p_value=runs_p,
                    critical_value=1.96,  # 95% confidence
                    is_significant=runs_p < self.significance_level,
                    interpretation="Errors show non-random patterns" if runs_p < self.significance_level 
                                 else "Errors appear random"
                ))
            except:
                pass
        
        return tests
    
    def _runs_test(self, binary_sequence: np.ndarray) -> Tuple[int, int, int]:
        """Perform runs test on binary sequence."""
        runs = 1
        n1 = np.sum(binary_sequence)
        n2 = len(binary_sequence) - n1
        
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        return runs, n1, n2    
    
    def _calculate_risk_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if results_df.empty:
            return {}
        
        predictions = results_df['prediction'].values
        returns = predictions  # Treating predictions as returns
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Tail risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Ulcer Index (alternative to standard deviation)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        ulcer_index = np.sqrt(np.mean(drawdowns**2))
        
        # Pain Index (average drawdown)
        pain_index = np.mean(abs(drawdowns))
        
        # Sterling Ratio
        avg_drawdown = np.mean(abs(drawdowns))
        sterling_ratio = np.mean(returns) * 252 / avg_drawdown if avg_drawdown > 0 else 0
        
        # Burke Ratio
        burke_ratio = np.mean(returns) * 252 / np.sqrt(np.sum(drawdowns**2)) if np.sum(drawdowns**2) > 0 else 0
        
        return {
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_consecutive_losses': max_consecutive_losses,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            'sterling_ratio': sterling_ratio,
            'burke_ratio': burke_ratio
        }
    
    def _calculate_consistency_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate consistency and stability metrics."""
        if results_df.empty or 'period_id' not in results_df.columns:
            return {}
        
        # Period-wise performance
        period_performance = results_df.groupby('period_id').agg({
            'error': 'mean',
            'directional_correct': 'mean',
            'prediction': ['mean', 'std']
        }).reset_index()
        
        period_maes = period_performance[('error', 'mean')].values
        period_accuracies = period_performance[('directional_correct', 'mean')].values * 100
        
        # Consistency metrics
        mae_consistency = 1 - (np.std(period_maes) / np.mean(period_maes)) if np.mean(period_maes) > 0 else 0
        accuracy_consistency = 1 - (np.std(period_accuracies) / np.mean(period_accuracies)) if np.mean(period_accuracies) > 0 else 0
        
        # Stability over time
        periods = len(period_maes)
        if periods > 2:
            # Linear trend in performance
            x = np.arange(periods)
            mae_slope, _, mae_r, _, _ = stats.linregress(x, period_maes)
            accuracy_slope, _, accuracy_r, _, _ = stats.linregress(x, period_accuracies)
        else:
            mae_slope = mae_r = accuracy_slope = accuracy_r = 0
        
        # Rolling window consistency
        window_size = min(5, periods // 2)
        if window_size >= 2:
            rolling_mae_std = pd.Series(period_maes).rolling(window_size).std().mean()
            rolling_accuracy_std = pd.Series(period_accuracies).rolling(window_size).std().mean()
        else:
            rolling_mae_std = rolling_accuracy_std = 0
        
        # Percentage of periods above median performance
        median_mae = np.median(period_maes)
        median_accuracy = np.median(period_accuracies)
        
        periods_above_median_mae = np.mean(period_maes <= median_mae) * 100  # Lower MAE is better
        periods_above_median_accuracy = np.mean(period_accuracies >= median_accuracy) * 100
        
        return {
            'mae_consistency': mae_consistency,
            'accuracy_consistency': accuracy_consistency,
            'mae_trend_slope': mae_slope,
            'mae_trend_r_squared': mae_r**2,
            'accuracy_trend_slope': accuracy_slope,
            'accuracy_trend_r_squared': accuracy_r**2,
            'rolling_mae_stability': 1 - rolling_mae_std if rolling_mae_std > 0 else 1,
            'rolling_accuracy_stability': 1 - rolling_accuracy_std if rolling_accuracy_std > 0 else 1,
            'periods_above_median_mae': periods_above_median_mae,
            'periods_above_median_accuracy': periods_above_median_accuracy,
            'n_periods': periods
        }
    
    def _compare_to_benchmark(self, results_df: pd.DataFrame, 
                            benchmark_returns: pd.Series) -> Dict[str, float]:
        """Compare performance to benchmark."""
        if results_df.empty:
            return {}
        
        predictions = results_df['prediction'].values
        
        # Align benchmark with predictions
        if 'timestamp' in results_df.columns:
            timestamps = pd.to_datetime(results_df['timestamp'])
            aligned_benchmark = []
            
            for ts in timestamps:
                # Find closest benchmark return
                closest_idx = benchmark_returns.index.get_indexer([ts], method='nearest')[0]
                if closest_idx >= 0 and closest_idx < len(benchmark_returns):
                    aligned_benchmark.append(benchmark_returns.iloc[closest_idx])
                else:
                    aligned_benchmark.append(0)
            
            benchmark_aligned = np.array(aligned_benchmark)
        else:
            # Use first N benchmark returns
            n_predictions = len(predictions)
            benchmark_aligned = benchmark_returns.values[:n_predictions]
        
        if len(benchmark_aligned) != len(predictions):
            return {}
        
        # Performance comparison
        strategy_return = np.mean(predictions) * 252
        benchmark_return = np.mean(benchmark_aligned) * 252
        
        strategy_vol = np.std(predictions) * np.sqrt(252)
        benchmark_vol = np.std(benchmark_aligned) * np.sqrt(252)
        
        strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0
        benchmark_sharpe = benchmark_return / benchmark_vol if benchmark_vol > 0 else 0
        
        # Excess returns
        excess_returns = predictions - benchmark_aligned
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        
        # Beta calculation
        if np.var(benchmark_aligned) > 0:
            beta = np.cov(predictions, benchmark_aligned)[0, 1] / np.var(benchmark_aligned)
        else:
            beta = 0
        
        # Alpha calculation (Jensen's alpha)
        alpha = strategy_return - beta * benchmark_return
        
        # Correlation
        correlation = np.corrcoef(predictions, benchmark_aligned)[0, 1] if len(predictions) > 1 else 0
        
        # Up/down capture ratios
        up_periods = benchmark_aligned > 0
        down_periods = benchmark_aligned < 0
        
        if np.any(up_periods) and np.mean(benchmark_aligned[up_periods]) != 0:
            up_capture = np.mean(predictions[up_periods]) / np.mean(benchmark_aligned[up_periods])
        else:
            up_capture = 0
        
        if np.any(down_periods) and np.mean(benchmark_aligned[down_periods]) != 0:
            down_capture = np.mean(predictions[down_periods]) / np.mean(benchmark_aligned[down_periods])
        else:
            down_capture = 0
        
        return {
            'strategy_return': strategy_return,
            'benchmark_return': benchmark_return,
            'excess_return': strategy_return - benchmark_return,
            'strategy_volatility': strategy_vol,
            'benchmark_volatility': benchmark_vol,
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture
        }
    
    def _generate_recommendations(self, summary_metrics: Dict[str, float],
                                risk_metrics: Dict[str, float],
                                consistency_metrics: Dict[str, float],
                                statistical_tests: List[StatisticalTest]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Accuracy recommendations
        if summary_metrics.get('mae', float('inf')) > 0.02:
            recommendations.append("HIGH ERROR: Consider improving feature engineering or model architecture")
        
        if summary_metrics.get('directional_accuracy', 0) < 55:
            recommendations.append("LOW DIRECTIONAL ACCURACY: Model predictions are barely better than random")
        
        # Risk recommendations
        if summary_metrics.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("LOW SHARPE RATIO: Risk-adjusted returns are poor, consider risk management")
        
        if summary_metrics.get('max_drawdown', 0) > 0.2:
            recommendations.append("HIGH DRAWDOWN: Implement better risk controls and position sizing")
        
        # Consistency recommendations
        if consistency_metrics.get('mae_consistency', 0) < 0.7:
            recommendations.append("INCONSISTENT PERFORMANCE: Model performance varies significantly across periods")
        
        if consistency_metrics.get('accuracy_trend_slope', 0) < -0.001:
            recommendations.append("DECLINING PERFORMANCE: Model accuracy is deteriorating over time")
        
        # Statistical test recommendations
        for test in statistical_tests:
            if test.test_name == "One-Sample t-test (Zero Mean Errors)" and test.is_significant:
                recommendations.append("BIASED PREDICTIONS: Model shows systematic bias, consider recalibration")
            
            if test.test_name == "Ljung-Box Test (Error Autocorrelation)" and test.is_significant:
                recommendations.append("AUTOCORRELATED ERRORS: Consider adding lagged features or AR terms")
            
            if test.test_name == "Runs Test (Error Randomness)" and test.is_significant:
                recommendations.append("NON-RANDOM ERRORS: Error patterns suggest missing features or model complexity")
        
        # Positive recommendations
        if summary_metrics.get('directional_accuracy', 0) > 65:
            recommendations.append("GOOD DIRECTIONAL ACCURACY: Model shows strong predictive power")
        
        if summary_metrics.get('sharpe_ratio', 0) > 2.0:
            recommendations.append("EXCELLENT RISK-ADJUSTED RETURNS: Model demonstrates strong performance")
        
        if consistency_metrics.get('mae_consistency', 0) > 0.8:
            recommendations.append("CONSISTENT PERFORMANCE: Model shows stable performance across periods")
        
        # Default recommendation if no issues found
        if not recommendations:
            recommendations.append("OVERALL GOOD PERFORMANCE: Model meets basic performance criteria")
        
        return recommendations


class ReportGenerator:
    """Generate formatted performance reports."""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
    
    def generate_html_report(self, performance_report: PerformanceReport, 
                           title: str = "Backtest Performance Report") -> str:
        """Generate HTML performance report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 5px 0; }}
                .section {{ margin: 20px 0; border: 1px solid #ccc; padding: 15px; }}
                .recommendation {{ background-color: #f0f8ff; padding: 10px; margin: 5px 0; }}
                .test-significant {{ color: red; }}
                .test-normal {{ color: green; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """
        
        # Summary metrics section
        html += "<div class='section'><h2>Summary Metrics</h2>"
        for metric, value in performance_report.summary_metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            html += f"<div class='metric'><strong>{metric.replace('_', ' ').title()}:</strong> {formatted_value}</div>"
        html += "</div>"
        
        # Statistical tests section
        html += "<div class='section'><h2>Statistical Tests</h2><table>"
        html += "<tr><th>Test</th><th>Statistic</th><th>P-Value</th><th>Significant</th><th>Interpretation</th></tr>"
        
        for test in performance_report.statistical_tests:
            significance_class = "test-significant" if test.is_significant else "test-normal"
            html += f"""
            <tr class='{significance_class}'>
                <td>{test.test_name}</td>
                <td>{test.statistic:.4f}</td>
                <td>{test.p_value:.4f}</td>
                <td>{'Yes' if test.is_significant else 'No'}</td>
                <td>{test.interpretation}</td>
            </tr>
            """
        html += "</table></div>"
        
        # Recommendations section
        html += "<div class='section'><h2>Recommendations</h2>"
        for rec in performance_report.recommendations:
            html += f"<div class='recommendation'>{rec}</div>"
        html += "</div>"
        
        html += "</body></html>"
        return html
    
    def generate_text_report(self, performance_report: PerformanceReport,
                           title: str = "Backtest Performance Report") -> str:
        """Generate text performance report."""
        report = f"{title}\n{'='*len(title)}\n\n"
        
        # Summary metrics
        report += "SUMMARY METRICS\n" + "-"*50 + "\n"
        for metric, value in performance_report.summary_metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            report += f"{metric.replace('_', ' ').title():<30}: {formatted_value}\n"
        
        # Statistical tests
        report += f"\n\nSTATISTICAL TESTS\n" + "-"*50 + "\n"
        for test in performance_report.statistical_tests:
            significance = "SIGNIFICANT" if test.is_significant else "NOT SIGNIFICANT"
            report += f"\n{test.test_name}:\n"
            report += f"  Statistic: {test.statistic:.4f}\n"
            report += f"  P-Value: {test.p_value:.4f}\n"
            report += f"  Result: {significance}\n"
            report += f"  Interpretation: {test.interpretation}\n"
        
        # Recommendations
        report += f"\n\nRECOMMENDATIONS\n" + "-"*50 + "\n"
        for i, rec in enumerate(performance_report.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report