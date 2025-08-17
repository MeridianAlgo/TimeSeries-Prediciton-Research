#!/usr/bin/env python3
"""
Unified Test Runner for Ultra-Precision Predictor System
=======================================================

This script runs all tests in the unified test suite with proper organization
and comprehensive reporting.
"""

import sys
import os
import time
import logging
from pathlib import Path
import subprocess
import argparse

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results.log')
    ]
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Unified test runner for the system."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        
    def run_test_category(self, category, verbose=False):
        """Run tests in a specific category."""
        category_dir = self.test_dir / category
        
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return False
            
        logger.info(f"Running {category} tests...")
        
        # Find all test files in the category
        test_files = list(category_dir.glob("test_*.py"))
        
        if not test_files:
            logger.warning(f"No test files found in {category}")
            return False
            
        success = True
        category_results = {}
        
        for test_file in test_files:
            logger.info(f"  Running {test_file.name}...")
            
            try:
                # Run pytest on the specific file
                cmd = [
                    sys.executable, "-m", "pytest", 
                    str(test_file),
                    "-v" if verbose else "-q",
                    "--tb=short"
                ]
                
                start_time = time.time()
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    cwd=str(project_root)
                )
                end_time = time.time()
                
                duration = end_time - start_time
                
                if result.returncode == 0:
                    logger.info(f"    ✓ {test_file.name} passed ({duration:.2f}s)")
                    category_results[test_file.name] = {
                        'status': 'PASSED',
                        'duration': duration,
                        'output': result.stdout
                    }
                else:
                    logger.error(f"    ✗ {test_file.name} failed ({duration:.2f}s)")
                    logger.error(f"    Error: {result.stderr}")
                    category_results[test_file.name] = {
                        'status': 'FAILED',
                        'duration': duration,
                        'output': result.stdout,
                        'error': result.stderr
                    }
                    success = False
                    
            except Exception as e:
                logger.error(f"    ✗ {test_file.name} error: {str(e)}")
                category_results[test_file.name] = {
                    'status': 'ERROR',
                    'duration': 0,
                    'error': str(e)
                }
                success = False
        
        self.results[category] = category_results
        return success
    
    def run_basic_functionality_test(self):
        """Run the basic functionality test."""
        logger.info("Running basic functionality test...")
        
        basic_test = self.test_dir / "integration" / "test_basic_functionality.py"
        
        if not basic_test.exists():
            logger.warning("Basic functionality test not found")
            return False
            
        try:
            cmd = [sys.executable, str(basic_test)]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root)
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✓ Basic functionality test passed ({duration:.2f}s)")
                return True
            else:
                logger.error(f"✗ Basic functionality test failed ({duration:.2f}s)")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Basic functionality test error: {str(e)}")
            return False
    
    def run_all_tests(self, categories=None, verbose=False):
        """Run all tests or specific categories."""
        logger.info("=" * 60)
        logger.info("Ultra-Precision Predictor System - Test Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Default categories
        if categories is None:
            categories = ['unit', 'integration', 'performance', 'examples']
        
        # Run basic functionality test first
        basic_success = self.run_basic_functionality_test()
        
        # Run category tests
        all_success = True
        for category in categories:
            if not self.run_test_category(category, verbose):
                all_success = False
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate summary report
        self.generate_summary_report(total_duration, basic_success and all_success)
        
        return basic_success and all_success
    
    def generate_summary_report(self, total_duration, overall_success):
        """Generate a summary report of test results."""
        logger.info("=" * 60)
        logger.info("TEST SUMMARY REPORT")
        logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        for category, tests in self.results.items():
            logger.info(f"\\n{category.upper()} TESTS:")
            
            for test_name, result in tests.items():
                status = result['status']
                duration = result.get('duration', 0)
                
                if status == 'PASSED':
                    logger.info(f"  ✓ {test_name} ({duration:.2f}s)")
                    passed_tests += 1
                elif status == 'FAILED':
                    logger.error(f"  ✗ {test_name} ({duration:.2f}s)")
                    failed_tests += 1
                else:  # ERROR
                    logger.error(f"  ⚠ {test_name} (ERROR)")
                    error_tests += 1
                
                total_tests += 1
        
        logger.info("=" * 60)
        logger.info(f"TOTAL TESTS: {total_tests}")
        logger.info(f"PASSED: {passed_tests}")
        logger.info(f"FAILED: {failed_tests}")
        logger.info(f"ERRORS: {error_tests}")
        logger.info(f"SUCCESS RATE: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
        logger.info(f"TOTAL DURATION: {total_duration:.2f} seconds")
        
        if overall_success:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error("❌ SOME TESTS FAILED")
        
        logger.info("=" * 60)
    
    def run_quick_test(self):
        """Run a quick subset of tests for fast validation."""
        logger.info("Running quick test suite...")
        
        # Run basic functionality test
        basic_success = self.run_basic_functionality_test()
        
        # Run a few key unit tests
        unit_dir = self.test_dir / "unit"
        if unit_dir.exists():
            key_tests = ["test_feature_engineering.py"]
            
            for test_name in key_tests:
                test_file = unit_dir / test_name
                if test_file.exists():
                    logger.info(f"Running {test_name}...")
                    try:
                        cmd = [
                            sys.executable, "-m", "pytest", 
                            str(test_file), "-q", "--tb=short"
                        ]
                        result = subprocess.run(cmd, cwd=str(project_root))
                        if result.returncode != 0:
                            basic_success = False
                    except Exception as e:
                        logger.error(f"Error running {test_name}: {str(e)}")
                        basic_success = False
        
        return basic_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ultra-Precision Predictor Test Runner")
    parser.add_argument(
        "--categories", 
        nargs="+", 
        choices=["unit", "integration", "performance", "examples"],
        help="Test categories to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick test suite only"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.quick:
            success = runner.run_quick_test()
        else:
            success = runner.run_all_tests(args.categories, args.verbose)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()