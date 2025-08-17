#!/usr/bin/env python3
"""Real-time accuracy monitoring system with live dashboard updates."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from maximum_accuracy_predictor import MaximumAccuracyPredictor
from stock_predictor.visualization.dashboard_builder import DashboardBuilder
from stock_predictor.visualization.theme_manager import ThemeManager
from stock_predictor.visualization.chart_factory import ChartFactory


class RealTimeAccuracyMonitor:
    """Real-time accuracy monitoring with live dashboard updates."""
    
    def __init__(self):
        self.predictor = MaximumAccuracyPredictor()
        
        # Initialize visualization components
        theme_config = {'default': 'dark_modern'}
        self.theme_manager = ThemeManager(theme_config)
        self.chart_factory = ChartFactory(self.theme_manager)
        self.dashboard_builder = DashboardBuilder(self.chart_factory, self.theme_manager)
        
        # Monitoring state
        self.accuracy_history = []
        self.error_history = []
        self.prediction_history = []
        self.model_performance_history = {}
        self.is_monitoring = False
        
        # Accuracy targets
        self.accuracy_targets = {
            'excellent': 90.0,  # ±1% accuracy
            'good': 80.0,       # ±2% accuracy
            'acceptable': 70.0   # ±3% accuracy
        }
        
        print("🎯 Real-Time Accuracy Monitor Initialized")
        print(f"📊 Accuracy Targets: Excellent ≥{self.accuracy_targets['excellent']}%, Good ≥{self.accuracy_targets['good']}%, Acceptable ≥{self.accuracy_targets['acceptable']}%")
    
    def train_predictor(self, data):
        """Train the maximum accuracy predictor."""
        print("🧠 Training maximum accuracy predictor...")
        
        # Create ultimate features
        enhanced_data = self.predictor.create_ultimate_features(data)
        
        # Prepare training data
        numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Handle NaN values
        enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Split data for training
        split_idx = int(len(enhanced_data) * 0.8)
        train_data = enhanced_data.iloc[:split_idx]
        
        X_train = train_data[feature_cols].values
        y_train = train_data['close'].values
        
        # Train models
        model_scores = self.predictor.train_maximum_accuracy_models(X_train, y_train)
        
        print("✅ Predictor training completed!")
        return model_scores, enhanced_data, feature_cols
    
    def make_prediction_with_accuracy(self, X, y_actual):
        """Make prediction and calculate real-time accuracy."""
        # Make prediction
        ensemble_pred, individual_preds = self.predictor.predict_maximum_accuracy(X)
        
        # Calculate accuracy metrics
        errors = np.abs((y_actual - ensemble_pred) / y_actual) * 100
        
        accuracy_metrics = {
            'rmse': np.sqrt(np.mean((y_actual - ensemble_pred) ** 2)),
            'mae': np.mean(np.abs(y_actual - ensemble_pred)),
            'mean_error_pct': np.mean(errors),
            'median_error_pct': np.median(errors),
            'accuracy_1pct': np.mean(errors <= 1.0) * 100,
            'accuracy_2pct': np.mean(errors <= 2.0) * 100,
            'accuracy_3pct': np.mean(errors <= 3.0) * 100,
            'max_error_pct': np.max(errors),
            'predictions_count': len(ensemble_pred)
        }
        
        # Calculate individual model accuracies
        individual_accuracies = {}
        for model_name, pred in individual_preds.items():
            model_errors = np.abs((y_actual - pred) / y_actual) * 100
            individual_accuracies[model_name] = {
                'rmse': np.sqrt(np.mean((y_actual - pred) ** 2)),
                'accuracy_2pct': np.mean(model_errors <= 2.0) * 100,
                'mean_error_pct': np.mean(model_errors)
            }
        
        return ensemble_pred, individual_preds, accuracy_metrics, individual_accuracies
    
    def update_accuracy_history(self, accuracy_metrics, timestamp=None):
        """Update accuracy history for monitoring."""
        if timestamp is None:
            timestamp = datetime.now()
        
        history_entry = {
            'timestamp': timestamp,
            **accuracy_metrics
        }
        
        self.accuracy_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]
    
    def get_accuracy_status(self, accuracy_2pct):
        """Get accuracy status based on targets."""
        if accuracy_2pct >= self.accuracy_targets['excellent']:
            return 'EXCELLENT', '🟢'
        elif accuracy_2pct >= self.accuracy_targets['good']:
            return 'GOOD', '🟡'
        elif accuracy_2pct >= self.accuracy_targets['acceptable']:
            return 'ACCEPTABLE', '🟠'
        else:
            return 'NEEDS IMPROVEMENT', '🔴'
    
    def create_live_dashboard(self, accuracy_metrics, individual_accuracies, predictions_data):
        """Create live accuracy monitoring dashboard."""
        print("📊 Creating live accuracy dashboard...")
        
        # Prepare data for dashboard
        model_metrics = {}
        for model_name, metrics in individual_accuracies.items():
            model_metrics[model_name] = {
                'rmse': metrics['rmse'],
                'accuracy_2pct': metrics['accuracy_2pct'],
                'mean_error_pct': metrics['mean_error_pct']
            }
        
        # Add ensemble metrics
        model_metrics['Ensemble'] = {
            'rmse': accuracy_metrics['rmse'],
            'accuracy_2pct': accuracy_metrics['accuracy_2pct'],
            'mean_error_pct': accuracy_metrics['mean_error_pct']
        }
        
        # Create accuracy monitoring dashboard
        dashboard = self.dashboard_builder.create_performance_dashboard(
            model_metrics=model_metrics,
            config={'title': 'Real-Time Accuracy Monitor'}
        )
        
        # Add real-time accuracy summary
        status, status_icon = self.get_accuracy_status(accuracy_metrics['accuracy_2pct'])
        
        accuracy_summary = {
            'Status': f"{status_icon} {status}",
            'Current Accuracy (±2%)': f"{accuracy_metrics['accuracy_2pct']:.1f}%",
            'Mean Error': f"{accuracy_metrics['mean_error_pct']:.2f}%",
            'RMSE': f"${accuracy_metrics['rmse']:.3f}",
            'Predictions Made': accuracy_metrics['predictions_count'],
            'Best Model': max(individual_accuracies.items(), key=lambda x: x[1]['accuracy_2pct'])[0],
            'Last Updated': datetime.now().strftime('%H:%M:%S')
        }
        
        dashboard.add_summary_panel(accuracy_summary, 'top')
        
        # Export live dashboard
        dashboard_path = Path('live_accuracy_dashboard.html')
        dashboard.export_dashboard('html', dashboard_path)
        
        return dashboard_path
    
    def run_continuous_monitoring(self, data, update_interval=30):
        """Run continuous accuracy monitoring with live updates."""
        print("🚀 Starting Continuous Accuracy Monitoring")
        print(f"⏱️ Update interval: {update_interval} seconds")
        print("=" * 60)
        
        # Train predictor
        model_scores, enhanced_data, feature_cols = self.train_predictor(data)
        
        # Prepare test data
        split_idx = int(len(enhanced_data) * 0.8)
        test_data = enhanced_data.iloc[split_idx:]
        
        X_test = test_data[feature_cols].values
        y_test = test_data['close'].values
        
        self.is_monitoring = True
        iteration = 0
        
        try:
            while self.is_monitoring:
                iteration += 1
                print(f"\n🔄 Monitoring Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Make predictions and calculate accuracy
                ensemble_pred, individual_preds, accuracy_metrics, individual_accuracies = \
                    self.make_prediction_with_accuracy(X_test, y_test)
                
                # Update history
                self.update_accuracy_history(accuracy_metrics)
                
                # Display current status
                status, status_icon = self.get_accuracy_status(accuracy_metrics['accuracy_2pct'])
                print(f"📊 Status: {status_icon} {status}")
                print(f"🎯 Accuracy (±2%): {accuracy_metrics['accuracy_2pct']:.1f}%")
                print(f"📉 Mean Error: {accuracy_metrics['mean_error_pct']:.2f}%")
                print(f"📊 RMSE: ${accuracy_metrics['rmse']:.3f}")
                
                # Show top performing models
                sorted_models = sorted(individual_accuracies.items(), 
                                     key=lambda x: x[1]['accuracy_2pct'], reverse=True)
                print(f"🏆 Top Models:")
                for i, (model_name, metrics) in enumerate(sorted_models[:3]):
                    print(f"  {i+1}. {model_name}: {metrics['accuracy_2pct']:.1f}% accuracy")
                
                # Create live dashboard
                dashboard_path = self.create_live_dashboard(
                    accuracy_metrics, individual_accuracies, 
                    {'ensemble': ensemble_pred, 'individual': individual_preds}
                )
                print(f"📊 Live dashboard updated: {dashboard_path}")
                
                # Check for accuracy alerts
                if accuracy_metrics['accuracy_2pct'] < self.accuracy_targets['acceptable']:
                    print(f"⚠️ ACCURACY ALERT: Current accuracy ({accuracy_metrics['accuracy_2pct']:.1f}%) below acceptable threshold ({self.accuracy_targets['acceptable']}%)")
                
                # Wait for next update
                if iteration < 5:  # Run 5 iterations for demo
                    print(f"⏳ Waiting {update_interval} seconds for next update...")
                    time.sleep(update_interval)
                else:
                    print("🛑 Demo completed after 5 iterations")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        self.is_monitoring = False
        print("\n✅ Continuous monitoring completed!")
    
    def generate_accuracy_report(self):
        """Generate comprehensive accuracy report."""
        if not self.accuracy_history:
            print("❌ No accuracy history available")
            return
        
        print("\n📋 Generating Accuracy Report...")
        
        # Calculate summary statistics
        recent_accuracies = [entry['accuracy_2pct'] for entry in self.accuracy_history[-10:]]
        recent_errors = [entry['mean_error_pct'] for entry in self.accuracy_history[-10:]]
        
        report = {
            'Total Monitoring Sessions': len(self.accuracy_history),
            'Average Accuracy (±2%)': f"{np.mean(recent_accuracies):.1f}%",
            'Best Accuracy': f"{np.max(recent_accuracies):.1f}%",
            'Worst Accuracy': f"{np.min(recent_accuracies):.1f}%",
            'Average Error': f"{np.mean(recent_errors):.2f}%",
            'Accuracy Trend': 'Improving' if len(recent_accuracies) > 1 and recent_accuracies[-1] > recent_accuracies[0] else 'Stable',
            'Target Achievement': f"{np.mean([acc >= self.accuracy_targets['good'] for acc in recent_accuracies]) * 100:.0f}% of sessions"
        }
        
        # Create report dashboard
        dashboard = self.dashboard_builder.create_custom_dashboard(
            title="Accuracy Monitoring Report",
            charts=[],
            summary_data=report
        )
        
        # Export report
        report_path = Path('accuracy_monitoring_report.html')
        dashboard.export_dashboard('html', report_path)
        
        print(f"📊 Accuracy report generated: {report_path}")
        
        # Display summary
        print("\n📊 ACCURACY MONITORING SUMMARY:")
        for key, value in report.items():
            print(f"  {key}: {value}")


def run_real_time_monitoring():
    """Run the real-time accuracy monitoring system."""
    print("🎯 REAL-TIME ACCURACY MONITORING SYSTEM")
    print("🚀 Targeting Maximum Prediction Accuracy!")
    print("=" * 70)
    
    try:
        # Load data
        print("📊 Loading stock data...")
        from stock_predictor.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=3.0)
        
        if raw_data is None or raw_data.empty:
            raise Exception("Could not fetch data")
        
        # Prepare data
        if 'date' in raw_data.columns:
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            raw_data = raw_data.set_index('date')
        
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in raw_data.columns:
                raw_data = raw_data.rename(columns={old_name: new_name})
        
        data = raw_data.sort_index()
        
        # Initialize monitor
        monitor = RealTimeAccuracyMonitor()
        
        # Run continuous monitoring
        monitor.run_continuous_monitoring(data, update_interval=10)  # 10 second intervals for demo
        
        # Generate final report
        monitor.generate_accuracy_report()
        
        print("\n🎉 Real-time monitoring completed successfully!")
        print("📊 Check the generated HTML files for live dashboards and reports!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_real_time_monitoring()