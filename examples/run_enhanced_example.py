"""Example script demonstrating the enhanced visualization system with real stock data."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockPredictorApp
from stock_predictor.visualization import VisualizationManager


def run_enhanced_example():
    """Run stock prediction with enhanced visualization."""
    print("Stock Price Prediction with Enhanced Visualization")
    print("=" * 60)
    
    try:
        # Initialize the main application
        app = StockPredictorApp()
        
        # Run prediction pipeline
        print("Running prediction pipeline for AAPL...")
        results = app.run_full_pipeline(symbol='AAPL', years_back=3)
        
        if not results:
            print("❌ Prediction pipeline failed")
            return
        
        print("✓ Prediction pipeline completed successfully")
        
        # Initialize enhanced visualization system
        print("\nInitializing enhanced visualization system...")
        
        viz_config = {
            'themes': {
                'default': 'light'
            },
            'charts': {
                'price_chart': {
                    'backend': 'matplotlib',
                    'enable_zoom': True,
                    'enable_hover': True,
                    'show_confidence_intervals': True
                },
                'performance_chart': {
                    'backend': 'matplotlib',
                    'show_individual_models': True,
                    'show_ensemble': True
                }
            },
            'dashboard': {
                'layout': 'grid',
                'responsive': True
            },
            'reports': {
                'default_template': 'comprehensive',
                'include_raw_data': False
            },
            'export': {
                'dpi': 300,
                'format_preferences': ['html', 'png']
            }
        }
        
        viz_manager = VisualizationManager(viz_config)
        print("✓ Enhanced visualization system initialized")
        
        # Create interactive price chart
        print("\nCreating interactive price chart...")
        try:
            # Prepare data for price chart
            historical_data = results.get('processed_data')
            model_predictions = {}
            
            # Extract predictions from results
            if 'model_predictions' in results:
                model_predictions = results['model_predictions']
            
            # Create ensemble data
            ensemble_data = None
            if 'ensemble_prediction' in results:
                ensemble_data = {
                    'prediction': results['ensemble_prediction'],
                    'confidence_intervals': results.get('confidence_intervals')
                }
            
            price_chart = viz_manager.create_interactive_price_chart(
                historical_data=historical_data,
                predictions=model_predictions,
                ensemble_data=ensemble_data
            )
            
            # Export price chart
            viz_manager.export_chart(price_chart, 'enhanced_price_chart.png', 'png')
            print("✓ Interactive price chart created and exported")
            
        except Exception as e:
            print(f"⚠ Price chart creation failed: {str(e)}")
        
        # Create performance dashboard
        print("\nCreating performance dashboard...")
        try:
            model_metrics = results.get('model_metrics', {})
            ensemble_weights = results.get('ensemble_weights', {})
            
            dashboard = viz_manager.create_performance_dashboard(
                model_metrics=model_metrics,
                ensemble_weights=ensemble_weights
            )
            
            print("✓ Performance dashboard created")
            print(f"  Dashboard contains {len(dashboard.charts)} charts")
            
        except Exception as e:
            print(f"⚠ Performance dashboard creation failed: {str(e)}")
        
        # Generate comprehensive report
        print("\nGenerating comprehensive HTML report...")
        try:
            report_path = viz_manager.generate_comprehensive_report(
                prediction_results=results,
                output_path='enhanced_stock_report.html'
            )
            
            print(f"✓ Comprehensive report generated: {report_path}")
            
        except Exception as e:
            print(f"⚠ Report generation failed: {str(e)}")
        
        # Test theme switching
        print("\nTesting theme capabilities...")
        try:
            available_themes = viz_manager.get_available_themes()
            print(f"Available themes: {available_themes}")
            
            # Switch to dark theme and create another chart
            if 'dark' in available_themes:
                viz_manager.set_theme('dark')
                print("✓ Switched to dark theme")
                
                # Create a dark-themed performance chart
                if model_metrics:
                    dark_chart = viz_manager.chart_factory.create_performance_chart(
                        model_metrics, 'comparison'
                    )
                    viz_manager.export_chart(dark_chart, 'dark_theme_chart.png', 'png')
                    print("✓ Dark theme chart created and exported")
            
        except Exception as e:
            print(f"⚠ Theme testing failed: {str(e)}")
        
        # Display results summary
        print("\n" + "=" * 60)
        print("ENHANCED VISUALIZATION RESULTS")
        print("=" * 60)
        
        if 'ensemble_metrics' in results:
            metrics = results['ensemble_metrics']
            print(f"Symbol: {results.get('symbol', 'Unknown')}")
            print(f"Ensemble RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%")
            print(f"Models Trained: {len(results.get('model_metrics', {}))}")
        
        print("\nGenerated Files:")
        generated_files = [
            'enhanced_price_chart.png',
            'enhanced_stock_report.html',
            'dark_theme_chart.png'
        ]
        
        for file in generated_files:
            if Path(file).exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ⚠ {file} (not generated)")
        
        print("\n🎉 Enhanced visualization example completed successfully!")
        print("Check the generated files to see the advanced visualization capabilities.")
        
    except Exception as e:
        print(f"❌ Enhanced visualization example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_enhanced_example()