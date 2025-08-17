"""Test runner for the stock price prediction system."""

import subprocess
import sys
import os


def run_tests():
    """Run all tests with coverage reporting."""
    
    print("Running Stock Price Prediction System Tests")
    print("=" * 50)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Test commands to run
    test_commands = [
        # Run all tests with coverage
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        
        # Run with coverage report
        ["python", "-m", "pytest", "tests/", "--cov=stock_predictor", "--cov-report=term-missing"],
        
        # Run integration tests specifically
        ["python", "-m", "pytest", "tests/test_integration.py", "-v", "-s"],
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{'-'*30}")
        print(f"Running Test Command {i}: {' '.join(cmd)}")
        print(f"{'-'*30}")
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Test command {i} failed with return code {result.returncode}")
            else:
                print(f"Test command {i} completed successfully")
        except Exception as e:
            print(f"Error running test command {i}: {str(e)}")
    
    print(f"\n{'='*50}")
    print("Test execution completed!")
    print("Check the output above for detailed results.")


def run_quick_tests():
    """Run a quick subset of tests for development."""
    
    print("Running Quick Tests")
    print("=" * 30)
    
    quick_test_files = [
        "tests/test_config.py",
        "tests/test_data_fetcher.py",
        "tests/test_data_preprocessor.py",
        "tests/test_feature_engineer.py",
        "tests/test_evaluator.py",
        "tests/test_ensemble_builder.py"
    ]
    
    for test_file in quick_test_files:
        if os.path.exists(test_file):
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_file, "-v"], 
                    capture_output=False, text=True
                )
                if result.returncode == 0:
                    print(f"✓ {test_file} passed")
                else:
                    print(f"✗ {test_file} failed")
            except Exception as e:
                print(f"Error running {test_file}: {str(e)}")
        else:
            print(f"⚠ {test_file} not found")


def check_dependencies():
    """Check if required dependencies are installed."""
    
    print("Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        "pandas", "numpy", "scikit-learn", "statsmodels", 
        "yfinance", "pytest", "pyyaml"
    ]
    
    optional_packages = [
        "tensorflow", "matplotlib", "seaborn", "plotly"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (REQUIRED)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    print("\nAll required dependencies are available!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_tests()
        elif sys.argv[1] == "deps":
            check_dependencies()
        elif sys.argv[1] == "full":
            if check_dependencies():
                run_tests()
        else:
            print("Usage: python run_tests.py [quick|deps|full]")
            print("  quick: Run quick subset of tests")
            print("  deps:  Check dependencies")
            print("  full:  Run all tests with coverage")
    else:
        print("Stock Price Prediction System - Test Runner")
        print("=" * 50)
        print("Available commands:")
        print("  python run_tests.py quick  - Run quick tests")
        print("  python run_tests.py deps   - Check dependencies")
        print("  python run_tests.py full   - Run all tests")
        print("\nRunning dependency check...")
        check_dependencies()