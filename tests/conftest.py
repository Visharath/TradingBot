"""
Test configuration for pytest.

This file contains fixtures and common test utilities used across
all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import tempfile
import os

# Import modules to test
from src.config import config, Config
from src.data_processor import DataProcessor
from src.model_builder import UnifiedModelBuilder
from src.unified_trading_bot import UnifiedTradingBot


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    
    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(1000) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(1000) * 0.5)
    open_prices = close_prices + np.random.randn(1000) * 0.3
    volume = np.random.randint(1000, 10000, 1000)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data


@pytest.fixture
def sample_processed_features():
    """Create sample processed features for testing."""
    n_samples = 500
    n_features = 150
    
    # Random features
    features = np.random.randn(n_samples, n_features)
    
    # Random targets (0=buy, 1=hold, 2=sell)
    targets = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return features, targets, feature_names


@pytest.fixture
def mock_ib_client():
    """Create a mock Interactive Brokers client."""
    mock_client = MagicMock()
    mock_client.isConnected.return_value = True
    mock_client.connectAsync.return_value = True
    
    # Mock trade execution
    mock_trade = MagicMock()
    mock_trade.order.orderId = 12345
    mock_client.placeOrder.return_value = mock_trade
    
    return mock_client


@pytest.fixture
def temp_model_file():
    """Create a temporary file for model testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def data_processor():
    """Create a DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def model_builder():
    """Create a UnifiedModelBuilder instance for testing."""
    return UnifiedModelBuilder()


@pytest.fixture
def trading_bot():
    """Create a UnifiedTradingBot instance for testing."""
    return UnifiedTradingBot()


@pytest.fixture
def test_config():
    """Create a test configuration."""
    test_config = Config()
    
    # Set test-specific values
    test_config.data.symbols = ["AAPL", "GOOGL"]
    test_config.model.sequence_length = 30  # Shorter for faster tests
    test_config.model.batch_size = 16
    test_config.model.epochs = 2  # Very few epochs for tests
    
    return test_config


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_libs():
    """Mock external libraries that might not be available in test environment."""
    with patch('talib.SMA') as mock_sma, \
         patch('talib.EMA') as mock_ema, \
         patch('talib.RSI') as mock_rsi:
        
        # Set up mock returns
        mock_sma.return_value = np.random.randn(100)
        mock_ema.return_value = np.random.randn(100)
        mock_rsi.return_value = np.random.randn(100) * 50 + 50
        
        yield


# Test data generators
def generate_price_series(length=1000, start_price=100, volatility=0.02):
    """Generate a realistic price series using geometric Brownian motion."""
    np.random.seed(42)
    returns = np.random.normal(0, volatility, length)
    prices = [start_price]
    
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    return np.array(prices[1:])


def generate_volume_series(length=1000, base_volume=10000):
    """Generate a realistic volume series."""
    np.random.seed(42)
    volume = np.random.lognormal(np.log(base_volume), 0.5, length)
    return volume.astype(int)


# Test utilities
class MockMarketDataProvider:
    """Mock market data provider for testing."""
    
    def __init__(self):
        self.data = {}
    
    def add_symbol_data(self, symbol, data):
        """Add data for a symbol."""
        self.data[symbol] = data
    
    def get_data(self, symbol, timeframe="1min"):
        """Get data for a symbol."""
        return self.data.get(symbol, pd.DataFrame())


# Performance test utilities
def measure_execution_time(func, *args, **kwargs):
    """Measure function execution time."""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# Configuration for different test types
pytest_plugins = []

# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require market data"
    )
    config.addinivalue_line(
        "markers", "requires_broker: marks tests that require broker connection"
    )