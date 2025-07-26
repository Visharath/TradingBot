"""
Package initialization for tests module.

This module provides common test utilities and imports for the test suite.
"""

# Test configuration
import pytest

# Mark slow tests
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Common test utilities
from tests.conftest import (
    assert_dataframe_valid,
    assert_series_valid,
    assert_predictions_valid,
    generate_ohlcv_data,
    time_function,
    memory_usage_test,
    TEST_SYMBOLS,
    VALID_SIGNALS,
)

__all__ = [
    "assert_dataframe_valid",
    "assert_series_valid", 
    "assert_predictions_valid",
    "generate_ohlcv_data",
    "time_function",
    "memory_usage_test",
    "TEST_SYMBOLS",
    "VALID_SIGNALS",
]