#!/usr/bin/env python3
"""Minimal test."""

import sys
import traceback

print("Testing minimal import...")

try:
    # Test the module import
    import ultra_precision_predictor.predictor
    print("Module imported successfully")
    print(f"Module contents: {dir(ultra_precision_predictor.predictor)}")
    
    # Try to access the class
    if hasattr(ultra_precision_predictor.predictor, 'UltraPrecisionPredictor'):
        print("Class found in module")
        cls = getattr(ultra_precision_predictor.predictor, 'UltraPrecisionPredictor')
        print(f"Class: {cls}")
    else:
        print("Class NOT found in module")
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# Try direct execution of the file
print("\nTrying direct execution...")
try:
    exec(open('ultra_precision_predictor/predictor.py').read())
    print("Direct execution successful")
    if 'UltraPrecisionPredictor' in locals():
        print("Class defined in locals")
    else:
        print("Class NOT in locals")
except Exception as e:
    print(f"Direct execution error: {e}")
    traceback.print_exc()