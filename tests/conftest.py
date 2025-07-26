"""
Test configuration and utilities for the Unified CNN-LSTM Trading Bot.

This module provides pytest fixtures and test utilities used across
all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.config import Config
from src.data_processor import DataProcessor
from src.model_builder import ModelBuilder
from src.unified_trading_bot import UnifiedTradingBot


@pytest.fixture
def config():
    """Provide a test configuration."""
    config = Config()
    
    # Override settings for testing
    config.model.sequence_length = 20  # Smaller for faster testing
    config.model.batch_size = 16
    config.model.epochs = 2  # Very few epochs for testing
    config.model.patience = 1
    
    config.data.symbols = ['AAPL', 'GOOGL']  # Fewer symbols for testing
    config.data.lookback_days = 30  # Less historical data
    config.data.min_data_points = 50  # Lower threshold
    
    config.trading.initial_capital = 10000  # Smaller capital for testing
    config.trading.max_positions = 2
    config.trading.paper_trading = True
    
    config.system.debug = True
    config.system.log_level = "DEBUG"
    
    return config


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Provide sample market data for testing."""
    np.random.seed(42)
    
    # Generate 100 days of synthetic market data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Start with a base price
    base_price = 100.0
    prices = []
    
    for i in range(len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    
    # Generate OHLC from close prices
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(data)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(data)))
    
    # Generate volume
    data['volume'] = np.random.lognormal(13, 0.5, len(data)).astype(int)
    
    # Forward fill the first NaN in open
    data['open'].fillna(method='bfill', inplace=True)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


@pytest.fixture
def sample_features(sample_data):
    """Provide sample feature data for testing."""
    # Simple features for testing
    features = pd.DataFrame(index=sample_data.index)
    
    # Price-based features
    features['price_change'] = sample_data['close'].pct_change()
    features['high_low_ratio'] = sample_data['high'] / sample_data['low']
    features['volume_ma'] = sample_data['volume'].rolling(5).mean()
    
    # Simple technical indicators
    features['sma_5'] = sample_data['close'].rolling(5).mean()
    features['sma_10'] = sample_data['close'].rolling(10).mean()
    features['rsi_14'] = calculate_simple_rsi(sample_data['close'], 14)
    
    # Time-based features
    features['hour'] = 10  # Market hour
    features['day_of_week'] = sample_data.index.dayofweek
    features['month'] = sample_data.index.month
    
    # Fill NaN values
    features.fillna(0, inplace=True)
    
    return features


@pytest.fixture
def sample_targets(sample_features):
    """Provide sample target labels for testing."""
    # Generate random but realistic targets
    np.random.seed(42)
    
    # Use price change to generate somewhat realistic targets
    price_change = sample_features['price_change']
    
    targets = pd.Series(1, index=sample_features.index)  # Default to hold
    targets[price_change > 0.02] = 2  # Buy signal for >2% expected gain
    targets[price_change < -0.02] = 0  # Sell signal for >2% expected loss
    
    return targets


@pytest.fixture
def data_processor(config):
    """Provide a DataProcessor instance."""
    return DataProcessor(config)


@pytest.fixture
def model_builder(config):
    """Provide a ModelBuilder instance."""
    return ModelBuilder(config)


@pytest.fixture
def trading_bot(config):
    """Provide a UnifiedTradingBot instance."""
    return UnifiedTradingBot()


def calculate_simple_rsi(prices, period=14):
    """Calculate a simple RSI for testing purposes."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class MockDataFetcher:
    """Mock data fetcher for testing without external API calls."""
    
    def __init__(self, sample_data):
        self.sample_data = sample_data
    
    def fetch_data(self, symbol, period='1y', interval='1d'):
        """Return sample data instead of fetching from external source."""
        data = self.sample_data.copy()
        data['symbol'] = symbol
        return data


@pytest.fixture
def mock_data_fetcher(sample_data, monkeypatch):
    """Mock the data fetching functionality."""
    fetcher = MockDataFetcher(sample_data)
    
    # Patch the fetch_data method in DataProcessor
    def mock_fetch_data(self, symbol, period='1y', interval='1d', force_refresh=False):
        return fetcher.fetch_data(symbol, period, interval)
    
    monkeypatch.setattr(DataProcessor, 'fetch_data', mock_fetch_data)
    return fetcher


# Test utilities
def assert_dataframe_valid(df, required_columns=None, min_rows=1):
    """Assert that a DataFrame is valid for testing."""
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert len(df) >= min_rows, f"DataFrame must have at least {min_rows} rows"
    assert not df.empty, "DataFrame cannot be empty"
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        assert not missing_columns, f"Missing required columns: {missing_columns}"


def assert_series_valid(series, min_length=1):
    """Assert that a Series is valid for testing."""
    assert isinstance(series, pd.Series), "Input must be a pandas Series"
    assert len(series) >= min_length, f"Series must have at least {min_length} elements"


def assert_predictions_valid(predictions, expected_classes=None):
    """Assert that predictions are valid."""
    assert isinstance(predictions, (np.ndarray, list)), "Predictions must be array-like"
    
    if expected_classes:
        unique_predictions = set(np.unique(predictions))
        assert unique_predictions.issubset(expected_classes), \
            f"Predictions contain invalid classes: {unique_predictions - expected_classes}"


def assert_model_trained(model):
    """Assert that a model has been trained."""
    assert model is not None, "Model cannot be None"
    # Add more specific checks based on your model structure


def create_test_data_file(file_path, data):
    """Create a test data file for testing file I/O operations."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path)
    return file_path


# Performance test utilities
def time_function(func, *args, **kwargs):
    """Time a function execution for performance testing."""
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    return result, execution_time


def memory_usage_test(func, *args, **kwargs):
    """Measure memory usage of a function."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Get initial memory
    initial_memory = process.memory_info().rss
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Get final memory
    final_memory = process.memory_info().rss
    
    memory_diff = final_memory - initial_memory
    return result, memory_diff


# Test data generators
def generate_price_series(length=100, start_price=100, volatility=0.02, trend=0.001):
    """Generate a synthetic price series for testing."""
    np.random.seed(42)
    
    prices = [start_price]
    for _ in range(length - 1):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    return np.array(prices)


def generate_ohlcv_data(length=100, start_price=100):
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    close_prices = generate_price_series(length, start_price)
    
    data = pd.DataFrame(index=dates)
    data['close'] = close_prices
    
    # Generate OHLC from close
    data['open'] = data['close'].shift(1).fillna(start_price)
    
    # Add some randomness to high and low
    daily_range = abs(data['close'] - data['open']) + np.random.uniform(0, start_price * 0.01, length)
    data['high'] = np.maximum(data['open'], data['close']) + daily_range * 0.5
    data['low'] = np.minimum(data['open'], data['close']) - daily_range * 0.5
    
    # Ensure low is positive
    data['low'] = np.maximum(data['low'], data['close'] * 0.5)
    
    # Generate volume
    data['volume'] = np.random.lognormal(10, 0.5, length).astype(int)
    
    return data


# Constants for testing
TEST_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT']
TEST_DATE_RANGE = ('2023-01-01', '2023-12-31')
VALID_SIGNALS = [0, 1, 2]  # Sell, Hold, Buy
VALID_INTERVALS = ['1m', '5m', '15m', '1h', '1d']

# Test configuration presets
FAST_CONFIG = {
    'model': {
        'sequence_length': 10,
        'batch_size': 8,
        'epochs': 1,
        'patience': 1
    },
    'data': {
        'symbols': ['AAPL'],
        'lookback_days': 30,
        'min_data_points': 20
    }
}

MINIMAL_CONFIG = {
    'model': {
        'sequence_length': 5,
        'batch_size': 4,
        'epochs': 1
    },
    'data': {
        'symbols': ['AAPL'],
        'lookback_days': 10,
        'min_data_points': 10
    }
}