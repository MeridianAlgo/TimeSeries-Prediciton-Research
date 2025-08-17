#!/usr/bin/env python3
"""
Unified Test Runner - Main Entry Point
=====================================

This is the main entry point for running the unified test suite.
It replaces the old scattered test files with a clean, organized approach.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the unified test runner
from tests_unified.run_all_tests import main

if __name__ == "__main__":
    print("🚀 Ultra-Precision Predictor - Unified Test Suite")
    print("=" * 60)
    print("This replaces the old scattered test files with organized testing.")
    print("Tests are now organized in: tests_unified/")
    print("=" * 60)
    
    # Run the unified test suite
    main()