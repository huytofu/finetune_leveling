#!/usr/bin/env python
"""
Script to run all tests for the PEFT implementation in explore_llm.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import os
import sys

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if there are test files in the directory
    test_files = [f for f in os.listdir(current_dir) if f.startswith('test_') and f.endswith('.py')]
    
    if not test_files:
        print("No test files found in the tests directory.")
        print("Make sure to create test files that start with 'test_' and end with '.py'.")
        return 0
    
    # Discover tests in the current directory
    test_suite = unittest.defaultTestLoader.discover(current_dir, pattern="test_*.py")
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)

if __name__ == "__main__":
    # Add parent directory to path to ensure imports work correctly
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Create empty __init__.py files for proper imports
    init_files = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py")
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# This file is intentionally left empty for Python package recognition\n")
    
    # Run the tests and get the result
    exit_code = run_all_tests()
    
    # Exit with an appropriate code
    sys.exit(exit_code) 