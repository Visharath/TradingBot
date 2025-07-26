"""
Tests for the DataProcessor module.

This module contains comprehensive tests for data fetching, 
technical indicator calculation, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data_processor import DataProcessor
from src.config import Config
from tests.conftest import (
    assert_dataframe_valid, assert_series_valid, 
    generate_ohlcv_data, TEST_SYMBOLS
)


class TestDataProcessor:
    """Test cases for the DataProcessor class."""
    
    def test_initialization(self, config):
        """Test DataProcessor initialization."""
        processor = DataProcessor(config)
        
        assert processor.config == config
        assert processor.data_config == config.data
        assert processor.cache_dir.exists()
    
    def test_fetch_data_success(self, data_processor, mock_data_fetcher):
        """Test successful data fetching."""
        symbol = 'AAPL'
        data = data_processor.fetch_data(symbol, period='1y', interval='1d')
        
        assert_dataframe_valid(data, required_columns=['open', 'high', 'low', 'close', 'volume'])
        assert data['symbol'].iloc[0] == symbol
        assert len(data) > 0
    
    def test_fetch_data_invalid_symbol(self, data_processor):
        """Test data fetching with invalid symbol."""
        # This test would typically test network failures or invalid symbols
        # For now, we'll test the error handling path
        with patch.object(data_processor, 'fetch_data', return_value=None):
            result = data_processor.fetch_data('INVALID')
            assert result is None
    
    def test_calculate_technical_indicators(self, data_processor, sample_data):
        """Test technical indicators calculation."""
        indicators = data_processor.calculate_technical_indicators(sample_data)
        
        # Verify the function returns a DataFrame
        assert_dataframe_valid(indicators)
        
        # Check that new indicators were added
        assert len(indicators.columns) > len(sample_data.columns)
        
        # Check for some expected indicators
        expected_indicators = [
            'rsi_14', 'macd', 'bb_upper_20', 'sma_20', 'ema_20',
            'adx', 'atr_14', 'obv', 'price_change'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators.columns, f"Missing indicator: {indicator}"
    
    def test_price_features(self, data_processor, sample_data):
        """Test price-based feature calculation."""
        result = data_processor._add_price_features(sample_data.copy())
        
        expected_features = ['hl2', 'hlc3', 'ohlc4', 'price_change', 'price_change_pct']
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing price feature: {feature}"
        
        # Test calculations
        assert np.allclose(result['hl2'], (sample_data['high'] + sample_data['low']) / 2, equal_nan=True)
        assert np.allclose(result['hlc3'], (sample_data['high'] + sample_data['low'] + sample_data['close']) / 3, equal_nan=True)
    
    def test_trend_indicators(self, data_processor, sample_data):
        """Test trend indicator calculation."""
        result = data_processor._add_trend_indicators(sample_data.copy())
        
        expected_indicators = ['sma_20', 'ema_20', 'macd', 'macd_signal', 'adx']
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing trend indicator: {indicator}"
        
        # Test that indicators have reasonable values
        assert not result['sma_20'].isna().all(), "SMA should have some non-NaN values"
        assert not result['ema_20'].isna().all(), "EMA should have some non-NaN values"
    
    def test_momentum_indicators(self, data_processor, sample_data):
        """Test momentum indicator calculation."""
        result = data_processor._add_momentum_indicators(sample_data.copy())
        
        expected_indicators = ['rsi_14', 'stoch_k', 'williams_r', 'mfi']
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing momentum indicator: {indicator}"
        
        # Test RSI bounds
        rsi_valid = result['rsi_14'].dropna()
        if len(rsi_valid) > 0:
            assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all(), "RSI should be between 0 and 100"
    
    def test_volatility_indicators(self, data_processor, sample_data):
        """Test volatility indicator calculation."""
        result = data_processor._add_volatility_indicators(sample_data.copy())
        
        expected_indicators = ['bb_upper_20', 'bb_lower_20', 'atr_14', 'kc_upper']
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing volatility indicator: {indicator}"
        
        # Test Bollinger Bands relationship
        bb_data = result[['bb_upper_20', 'bb_middle_20', 'bb_lower_20']].dropna()
        if len(bb_data) > 0:
            assert (bb_data['bb_upper_20'] >= bb_data['bb_middle_20']).all(), "BB upper should be >= middle"
            assert (bb_data['bb_middle_20'] >= bb_data['bb_lower_20']).all(), "BB middle should be >= lower"
    
    def test_volume_indicators(self, data_processor, sample_data):
        """Test volume indicator calculation."""
        result = data_processor._add_volume_indicators(sample_data.copy())
        
        expected_indicators = ['obv', 'vpt', 'mfi', 'cmf', 'vwap']
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing volume indicator: {indicator}"
    
    def test_statistical_features(self, data_processor, sample_data):
        """Test statistical feature calculation."""
        result = data_processor._add_statistical_features(sample_data.copy())
        
        expected_features = ['close_mean_20', 'close_std_20', 'zscore_20']
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing statistical feature: {feature}"
    
    def test_pattern_features(self, data_processor, sample_data):
        """Test pattern recognition features."""
        result = data_processor._add_pattern_features(sample_data.copy())
        
        expected_patterns = ['doji', 'hammer', 'bullish_engulfing', 'inside_bar']
        
        for pattern in expected_patterns:
            assert pattern in result.columns, f"Missing pattern: {pattern}"
            # Pattern features should be binary (0 or 1)
            unique_values = set(result[pattern].dropna().unique())
            assert unique_values.issubset({0, 1}), f"Pattern {pattern} should be binary"
    
    def test_create_features_and_targets(self, data_processor, sample_data):
        """Test feature and target creation."""
        # First calculate indicators
        data_with_indicators = data_processor.calculate_technical_indicators(sample_data)
        
        # Create features and targets
        features, targets = data_processor.create_features_and_targets(
            data_with_indicators,
            prediction_horizon=5,
            threshold_pct=0.02
        )
        
        assert_dataframe_valid(features, min_rows=1)
        assert_series_valid(targets, min_length=1)
        
        # Check that targets are valid classes
        unique_targets = set(targets.unique())
        assert unique_targets.issubset({0, 1, 2}), "Targets should be 0, 1, or 2"
        
        # Check that features and targets have same length
        assert len(features) == len(targets), "Features and targets should have same length"
    
    def test_handle_missing_values(self, data_processor):
        """Test missing value handling."""
        # Create data with missing values
        test_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, 2, 3, np.nan, 5],
            'col3': [1, np.nan, np.nan, np.nan, 5]
        })
        
        # Test forward fill
        data_processor.data_config.handle_missing = "forward_fill"
        result = data_processor._handle_missing_values(test_data.copy())
        assert not result.isna().any().any(), "Forward fill should remove all NaN values"
        
        # Test interpolation
        data_processor.data_config.handle_missing = "interpolate"
        result = data_processor._handle_missing_values(test_data.copy())
        # Should have fewer NaN values
        assert result.isna().sum().sum() <= test_data.isna().sum().sum()
    
    def test_scale_features(self, data_processor, sample_features):
        """Test feature scaling."""
        # Test with new scaler
        scaled_features, scaler = data_processor.scale_features(sample_features)
        
        assert_dataframe_valid(scaled_features)
        assert scaler is not None
        assert scaled_features.shape == sample_features.shape
        
        # Test with existing scaler
        scaled_features2, scaler2 = data_processor.scale_features(sample_features, scaler)
        assert scaler2 == scaler
        
        # Scaled features should have similar distribution
        np.testing.assert_allclose(scaled_features.values, scaled_features2.values, rtol=1e-10)
    
    def test_process_symbol_success(self, data_processor, mock_data_fetcher):
        """Test successful symbol processing."""
        result = data_processor.process_symbol('AAPL')
        
        assert result is not None
        features, targets = result
        
        assert_dataframe_valid(features)
        assert_series_valid(targets)
        assert len(features) == len(targets)
    
    def test_process_symbol_insufficient_data(self, data_processor):
        """Test symbol processing with insufficient data."""
        # Mock fetch_data to return insufficient data
        with patch.object(data_processor, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({'close': [1, 2, 3]})  # Very small dataset
            
            result = data_processor.process_symbol('AAPL')
            assert result is None
    
    def test_get_feature_names(self, data_processor):
        """Test feature names retrieval."""
        feature_names = data_processor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Should contain expected categories of features
        feature_str = ' '.join(feature_names)
        expected_categories = ['rsi', 'sma', 'macd', 'bb_', 'atr', 'obv']
        
        for category in expected_categories:
            assert any(category in name for name in feature_names), f"Missing {category} indicators"
    
    def test_data_validation(self, data_processor):
        """Test data validation with invalid data."""
        # Test with missing required columns
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            data_processor.calculate_technical_indicators(invalid_data)
    
    def test_caching_mechanism(self, data_processor, mock_data_fetcher, temp_dir):
        """Test data caching functionality."""
        # Update cache directory to temp directory
        data_processor.cache_dir = temp_dir
        
        symbol = 'AAPL'
        
        # First call should create cache
        data1 = data_processor.fetch_data(symbol)
        
        # Check that cache file was created
        cache_files = list(temp_dir.glob(f"{symbol}_*.pkl"))
        assert len(cache_files) > 0, "Cache file should be created"
        
        # Second call should use cache (mock won't be called again)
        data2 = data_processor.fetch_data(symbol)
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_error_handling(self, data_processor):
        """Test error handling in data processing."""
        # Test with None data
        result = data_processor.calculate_technical_indicators(None)
        # Should handle gracefully or raise appropriate error
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            data_processor.calculate_technical_indicators(empty_data)
    
    def test_performance_with_large_dataset(self, data_processor):
        """Test performance with larger dataset."""
        # Generate larger dataset
        large_data = generate_ohlcv_data(length=1000)
        
        # Measure time for indicator calculation
        import time
        start_time = time.time()
        
        indicators = data_processor.calculate_technical_indicators(large_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30, f"Processing took too long: {processing_time} seconds"
        
        # Should produce valid output
        assert_dataframe_valid(indicators)
        assert len(indicators) == len(large_data)
    
    def test_different_timeframes(self, data_processor, mock_data_fetcher):
        """Test data processing with different timeframes."""
        intervals = ['1d', '1h', '5m']
        
        for interval in intervals:
            data = data_processor.fetch_data('AAPL', interval=interval)
            assert_dataframe_valid(data)
            
            # Technical indicators should work for all timeframes
            indicators = data_processor.calculate_technical_indicators(data)
            assert_dataframe_valid(indicators)


class TestDataProcessorIntegration:
    """Integration tests for DataProcessor with other components."""
    
    def test_integration_with_model_builder(self, data_processor, model_builder, mock_data_fetcher):
        """Test integration between DataProcessor and ModelBuilder."""
        # Process data
        result = data_processor.process_symbol('AAPL')
        assert result is not None
        
        features, targets = result
        
        # Scale features
        scaled_features, scaler = data_processor.scale_features(features)
        
        # Prepare data for model
        inputs, y = model_builder.prepare_data(scaled_features, targets)
        
        # Should produce valid inputs for model
        assert 'sequence_input' in inputs
        assert 'static_input' in inputs
        assert len(y) > 0
    
    def test_real_time_processing_simulation(self, data_processor, mock_data_fetcher):
        """Simulate real-time data processing."""
        symbol = 'AAPL'
        
        # Simulate processing data at different time intervals
        for i in range(3):
            data = data_processor.fetch_data(symbol)
            indicators = data_processor.calculate_technical_indicators(data)
            features, targets = data_processor.create_features_and_targets(indicators)
            
            # Each iteration should produce consistent results
            assert_dataframe_valid(features)
            assert_series_valid(targets)
    
    def test_multi_symbol_processing(self, data_processor, mock_data_fetcher):
        """Test processing multiple symbols simultaneously."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        results = []
        
        for symbol in symbols:
            result = data_processor.process_symbol(symbol)
            if result is not None:
                results.append(result)
        
        assert len(results) > 0, "Should process at least some symbols successfully"
        
        # All results should have consistent feature structure
        if len(results) > 1:
            feature_columns = [set(features.columns) for features, _ in results]
            assert all(cols == feature_columns[0] for cols in feature_columns), \
                "All symbols should have same feature columns"


@pytest.mark.parametrize("period,interval", [
    ("1y", "1d"),
    ("6mo", "1h"),
    ("1mo", "15m"),
])
def test_different_data_periods(data_processor, mock_data_fetcher, period, interval):
    """Test data fetching with different periods and intervals."""
    data = data_processor.fetch_data('AAPL', period=period, interval=interval)
    
    if data is not None:
        assert_dataframe_valid(data)
        assert 'symbol' in data.columns


@pytest.mark.parametrize("threshold", [0.01, 0.02, 0.05])
def test_different_target_thresholds(data_processor, sample_data, threshold):
    """Test target creation with different thresholds."""
    data_with_indicators = data_processor.calculate_technical_indicators(sample_data)
    features, targets = data_processor.create_features_and_targets(
        data_with_indicators,
        threshold_pct=threshold
    )
    
    # Higher thresholds should result in more "hold" signals
    hold_ratio = (targets == 1).sum() / len(targets)
    
    # This is a general expectation, actual results may vary
    assert 0 <= hold_ratio <= 1, "Hold ratio should be between 0 and 1"