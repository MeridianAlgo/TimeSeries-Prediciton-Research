#!/usr/bin/env python3
"""
Comprehensive test runner for Ultra-Precision Predictor System.
Aims for <1% error rate validation.
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and capture its output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(command)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Completed in {duration:.2f} seconds")
        
        if result.returncode == 0:
            logger.info(f"[PASS] {description} PASSED")
            return True, result.stdout, result.stderr, duration
        else:
            logger.error(f"[FAIL] {description} FAILED")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr, duration
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} TIMED OUT")
        return False, "", "Timeout expired", 300
    except Exception as e:
        logger.error(f"✗ {description} ERROR: {str(e)}")
        return False, "", str(e), 0


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'scipy', 
        'pytest', 'psutil', 'logging'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"[FAIL] {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("[OK] All dependencies are available")
    return True


def validate_project_structure():
    """Validate that the project structure is correct."""
    logger.info("Validating project structure...")
    
    required_paths = [
        'ultra_precision_predictor/__init__.py',
        'ultra_precision_predictor/core/__init__.py',
        'ultra_precision_predictor/feature_engineering/__init__.py',
        'ultra_precision_predictor/predictor.py',
        'tests/__init__.py',
        'tests/test_feature_engineering.py',
        'tests/test_prediction_accuracy.py',
        'tests/test_integration.py'
    ]
    
    missing_paths = []
    
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
            logger.error(f"[FAIL] Missing: {path}")
        else:
            logger.info(f"[OK] Found: {path}")
    
    if missing_paths:
        logger.error(f"Missing files: {', '.join(missing_paths)}")
        return False
    
    logger.info("[OK] Project structure is valid")
    return True


def run_syntax_check():
    """Run syntax check on all Python files."""
    logger.info("Running syntax check...")
    
    python_files = []
    for pattern in ['**/*.py']:
        python_files.extend(Path('.').glob(pattern))
    
    failed_files = []
    
    for py_file in python_files:
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        success, stdout, stderr, duration = run_command(
            [sys.executable, '-m', 'py_compile', str(py_file)],
            f"Syntax check: {py_file}"
        )
        
        if not success:
            failed_files.append(str(py_file))
    
    if failed_files:
        logger.error(f"Syntax errors in: {', '.join(failed_files)}")
        return False
    
    logger.info("[OK] All Python files have valid syntax")
    return True


def run_unit_tests():
    """Run unit tests for feature engineering."""
    logger.info("Running unit tests...")
    
    success, stdout, stderr, duration = run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_feature_engineering.py', '-v', '--tb=short'],
        "Unit Tests (Feature Engineering)"
    )
    
    return success, stdout, stderr, duration


def run_accuracy_tests():
    """Run prediction accuracy tests."""
    logger.info("Running accuracy tests...")
    
    success, stdout, stderr, duration = run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_prediction_accuracy.py', '-v', '--tb=short'],
        "Accuracy Tests (Prediction)"
    )
    
    return success, stdout, stderr, duration


def run_integration_tests():
    """Run integration tests."""
    logger.info("Running integration tests...")
    
    success, stdout, stderr, duration = run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_integration.py', '-v', '--tb=short'],
        "Integration Tests"
    )
    
    return success, stdout, stderr, duration


def run_performance_benchmark():
    """Run performance benchmark."""
    logger.info("Running performance benchmark...")
    
    try:
        # Import here to avoid issues if modules aren't ready
        from ultra_precision_predictor.predictor import UltraPrecisionPredictor
        
        # Generate test data
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        returns = np.random.normal(0.0001, 0.02, n_samples)
        prices = 100 * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.5, n_samples)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        # Benchmark training
        predictor = UltraPrecisionPredictor()
        
        start_time = time.time()
        predictor.train(data.iloc[:800])
        training_time = time.time() - start_time
        
        # Benchmark prediction
        start_time = time.time()
        predictions = predictor.predict(data.iloc[800:])
        prediction_time = time.time() - start_time
        
        logger.info(f"✓ Training time: {training_time:.2f} seconds")
        logger.info(f"✓ Prediction time: {prediction_time:.2f} seconds")
        logger.info(f"✓ Training speed: {800 / training_time:.0f} samples/second")
        
        if predictions is not None:
            logger.info(f"✓ Prediction speed: {len(predictions) / prediction_time:.0f} predictions/second")
        
        return True, f"Training: {training_time:.2f}s, Prediction: {prediction_time:.2f}s", "", training_time + prediction_time
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {str(e)}")
        return False, "", str(e), 0


def generate_test_report(results):
    """Generate a comprehensive test report."""
    logger.info("Generating test report...")
    
    report_lines = [
        "=" * 80,
        "ULTRA-PRECISION PREDICTOR SYSTEM - TEST REPORT",
        "=" * 80,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY:",
        "--------"
    ]
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result['success'])
    failed_tests = total_tests - passed_tests
    
    report_lines.extend([
        f"Total Tests: {total_tests}",
        f"Passed: {passed_tests}",
        f"Failed: {failed_tests}",
        f"Success Rate: {(passed_tests / total_tests * 100):.1f}%",
        ""
    ])
    
    # Detailed results
    report_lines.extend([
        "DETAILED RESULTS:",
        "-" * 50
    ])
    
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        duration = result['duration']
        
        report_lines.extend([
            f"{test_name}: {status} ({duration:.2f}s)",
            f"  Output: {result['stdout'][:200]}..." if len(result['stdout']) > 200 else f"  Output: {result['stdout']}",
            ""
        ])
        
        if not result['success'] and result['stderr']:
            report_lines.extend([
                f"  Error: {result['stderr'][:500]}..." if len(result['stderr']) > 500 else f"  Error: {result['stderr']}",
                ""
            ])
    
    # Performance summary
    total_time = sum(result['duration'] for result in results.values())
    report_lines.extend([
        "PERFORMANCE SUMMARY:",
        "-" * 50,
        f"Total execution time: {total_time:.2f} seconds",
        ""
    ])
    
    # Recommendations
    report_lines.extend([
        "RECOMMENDATIONS:",
        "-" * 50
    ])
    
    if failed_tests == 0:
        report_lines.append("✓ All tests passed! System is ready for production.")
    else:
        report_lines.append(f"✗ {failed_tests} test(s) failed. Review errors above.")
        
    if passed_tests / total_tests >= 0.9:
        report_lines.append("✓ High success rate indicates robust system.")
    else:
        report_lines.append("⚠ Low success rate indicates system needs improvement.")
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    # Write report to file
    report_content = "\\n".join(report_lines)
    
    with open('test_report.txt', 'w') as f:
        f.write(report_content)
    
    # Also log to console
    logger.info("\\n" + report_content)
    
    return report_content


def main():
    """Main test runner function."""
    logger.info("Starting Ultra-Precision Predictor System Test Suite")
    logger.info("=" * 80)
    
    # Store all test results
    results = {}
    
    # 1. Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        return 1
    
    # 2. Validate project structure
    if not validate_project_structure():
        logger.error("Project structure validation failed. Exiting.")
        return 1
    
    # 3. Run syntax check
    success, stdout, stderr, duration = (run_syntax_check(), "", "", 0) if run_syntax_check() else (False, "", "Syntax errors found", 0)
    results['Syntax Check'] = {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'duration': duration
    }
    
    if not success:
        logger.error("Syntax check failed. Fix syntax errors before running tests.")
        generate_test_report(results)
        return 1
    
    # 4. Run unit tests
    success, stdout, stderr, duration = run_unit_tests()
    results['Unit Tests'] = {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'duration': duration
    }
    
    # 5. Run accuracy tests
    success, stdout, stderr, duration = run_accuracy_tests()
    results['Accuracy Tests'] = {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'duration': duration
    }
    
    # 6. Run integration tests
    success, stdout, stderr, duration = run_integration_tests()
    results['Integration Tests'] = {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'duration': duration
    }
    
    # 7. Run performance benchmark
    success, stdout, stderr, duration = run_performance_benchmark()
    results['Performance Benchmark'] = {
        'success': success,
        'stdout': stdout,
        'stderr': stderr,
        'duration': duration
    }
    
    # Generate comprehensive report
    generate_test_report(results)
    
    # Calculate overall success
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result['success'])
    success_rate = passed_tests / total_tests
    
    logger.info(f"\\nOverall Success Rate: {success_rate * 100:.1f}%")
    
    if success_rate >= 0.8:
        logger.info("✓ Test suite completed successfully!")
        return 0
    else:
        logger.error("✗ Test suite failed. Check test_report.txt for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)