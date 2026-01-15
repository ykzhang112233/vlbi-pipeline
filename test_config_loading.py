#!/usr/bin/env python
"""
Test script to verify configuration loading works correctly
"""
import sys
import os

# Add vlbi-pipeline directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vlbi-pipeline'))

# Simulate command line argument
sys.argv = ['test_config_loading.py', '--config', 'configs/test_input.py']

# Set environment variable
os.environ['VLBI_CONFIG'] = 'configs/test_input.py'

print("=" * 60)
print("Testing Configuration Loading")
print("=" * 60)

try:
    # Import config module
    from config import AIPS_NUMBER, file_name, target, antname
    
    print("✓ Configuration loaded successfully!")
    print(f"  AIPS_NUMBER: {AIPS_NUMBER}")
    print(f"  antname: {antname}")
    print(f"  file_name: {file_name}")
    print(f"  target: {target}")
    print()
    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("=" * 60)
    print("Test FAILED!")
    print("=" * 60)
    sys.exit(1)
