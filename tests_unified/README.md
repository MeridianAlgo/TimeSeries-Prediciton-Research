# Ultra-Precision Predictor - Unified Test Suite

This directory contains all tests for the Ultra-Precision Predictor system, organized by category for better maintainability and clarity.

## Directory Structure

```
tests_unified/
├── unit/                    # Unit tests for individual components
│   ├── test_feature_engineering.py
│   └── ...
├── integration/             # Integration tests for system workflows
│   ├── test_basic_functionality.py
│   ├── test_system_integration.py
│   └── ...
├── performance/             # Performance and accuracy tests
│   ├── test_prediction_accuracy.py
│   └── ...
├── examples/                # Example usage and demonstration tests
│   ├── test_minimal_import.py
│   ├── test_simple_usage.py
│   └── ...
├── run_all_tests.py         # Unified test runner
└── README.md               # This file
```

## Test Categories

### Unit Tests (`unit/`)
Tests for individual components and modules:
- Feature engineering components
- Data validation
- Core utilities
- Individual predictors

### Integration Tests (`integration/`)
Tests for complete system workflows:
- End-to-end prediction pipeline
- Feature engineering integration
- Model training and prediction
- Error handling and robustness

### Performance Tests (`performance/`)
Tests for system performance and accuracy:
- Prediction accuracy validation
- Performance benchmarks
- Memory usage tests
- Scalability tests

### Example Tests (`examples/`)
Simple tests and usage examples:
- Basic import tests
- Simple usage workflows
- Quick validation tests
- Documentation examples

## Running Tests

### Run All Tests
```bash
python tests_unified/run_all_tests.py
```

### Run Specific Categories
```bash
python tests_unified/run_all_tests.py --categories unit integration
```

### Run Quick Test Suite
```bash
python tests_unified/run_all_tests.py --quick
```

### Run with Verbose Output
```bash
python tests_unified/run_all_tests.py --verbose
```

### Run Individual Test Files
```bash
# Using pytest
pytest tests_unified/unit/test_feature_engineering.py -v

# Direct execution
python tests_unified/examples/test_simple_usage.py
```

## Test Requirements

### Dependencies
- pytest
- numpy
- pandas
- scikit-learn
- psutil (for memory tests)

### Data Requirements
Tests generate synthetic data automatically, so no external data files are required.

## Writing New Tests

### Unit Tests
- Test individual functions and classes
- Use mocking for external dependencies
- Focus on edge cases and error conditions
- Keep tests fast and isolated

### Integration Tests
- Test complete workflows
- Use realistic data
- Test error handling
- Validate end-to-end functionality

### Performance Tests
- Measure accuracy metrics
- Benchmark speed and memory
- Test with different data sizes
- Validate against targets

### Example Tests
- Keep simple and educational
- Show basic usage patterns
- Validate core functionality
- Serve as documentation

## Test Data

All tests use synthetic data generated within the test files. This ensures:
- Reproducible results (using fixed random seeds)
- No external dependencies
- Realistic market-like patterns
- Controlled test scenarios

## Continuous Integration

The unified test runner is designed to work with CI/CD systems:
- Returns proper exit codes (0 for success, 1 for failure)
- Generates detailed logs
- Supports different verbosity levels
- Can run subsets of tests for faster feedback

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in Python path
2. **Missing Dependencies**: Install required packages with `pip install -r requirements.txt`
3. **Memory Issues**: Run tests individually if system has limited memory
4. **Slow Tests**: Use `--quick` flag for faster validation

### Debug Mode
For debugging test failures:
```bash
python tests_unified/run_all_tests.py --verbose --categories unit
```

### Individual Test Debugging
```bash
pytest tests_unified/unit/test_feature_engineering.py::TestFeatureEngineering::test_extreme_feature_engineer -v -s
```

## Contributing

When adding new tests:
1. Choose the appropriate category
2. Follow existing naming conventions
3. Include docstrings and comments
4. Add both positive and negative test cases
5. Update this README if needed

## Test Metrics

The test suite aims for:
- **Coverage**: >90% code coverage
- **Performance**: <1 minute for quick tests, <10 minutes for full suite
- **Reliability**: All tests should pass consistently
- **Maintainability**: Clear, well-documented test code