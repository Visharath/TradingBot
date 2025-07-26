"""
Tests for the DataProcessor module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data_processor import DataProcessor
from src.config import config


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        
        assert processor.scaler is None
        assert processor.feature_selector is None
        assert processor.feature_names == []
        assert processor.processed_features == []
    
    def test_clean_data_basic(self, data_processor, sample_ohlcv_data):
        """Test basic data cleaning functionality."""
        # Add some invalid data
        bad_data = sample_ohlcv_data.copy()
        bad_data.loc[bad_data.index[0], 'high'] = bad_data.loc[bad_data.index[0], 'low'] - 1  # Invalid: high < low
        bad_data.loc[bad_data.index[1], 'volume'] = -100  # Invalid: negative volume
        
        cleaned_data = data_processor._clean_data(bad_data)
        
        # Should remove invalid rows
        assert len(cleaned_data) < len(bad_data)
        
        # Should have valid OHLCV relationships
        assert (cleaned_data['high'] >= cleaned_data['low']).all()
        assert (cleaned_data['high'] >= cleaned_data['open']).all()
        assert (cleaned_data['high'] >= cleaned_data['close']).all()
        assert (cleaned_data['volume'] >= 0).all()
    
    def test_clean_data_duplicates(self, data_processor, sample_ohlcv_data):
        """Test duplicate removal in data cleaning."""
        # Add duplicate rows
        data_with_dupes = pd.concat([sample_ohlcv_data, sample_ohlcv_data.iloc[:5]])
        
        cleaned_data = data_processor._clean_data(data_with_dupes)
        
        # Should remove duplicates
        assert len(cleaned_data) == len(sample_ohlcv_data)
        assert not cleaned_data.index.duplicated().any()
    
    @patch('src.data_processor.talib')
    def test_trend_indicators(self, mock_talib, data_processor, sample_ohlcv_data):
        """Test trend indicator calculation."""
        # Mock talib functions
        mock_talib.SMA.return_value = np.random.randn(len(sample_ohlcv_data))
        mock_talib.EMA.return_value = np.random.randn(len(sample_ohlcv_data))
        mock_talib.MACD.return_value = (
            np.random.randn(len(sample_ohlcv_data)),
            np.random.randn(len(sample_ohlcv_data)),
            np.random.randn(len(sample_ohlcv_data))
        )
        
        # Extract price arrays
        close_prices = sample_ohlcv_data['close'].values
        high_prices = sample_ohlcv_data['high'].values
        low_prices = sample_ohlcv_data['low'].values
        open_prices = sample_ohlcv_data['open'].values
        volume = sample_ohlcv_data['volume'].values
        
        indicators = data_processor._trend_indicators(
            open_prices, high_prices, low_prices, close_prices, volume
        )
        
        # Should return a DataFrame with trend indicators
        assert isinstance(indicators, pd.DataFrame)
        assert len(indicators) > 0
        
        # Should contain expected indicators
        expected_indicators = ['sma_5', 'sma_10', 'ema_12', 'macd', 'macd_signal']
        for indicator in expected_indicators:
            assert any(indicator in col for col in indicators.columns)
    
    def test_create_targets(self, data_processor, sample_ohlcv_data):
        """Test target creation."""
        targets = data_processor._create_targets(sample_ohlcv_data)
        
        # Should return a Series
        assert isinstance(targets, pd.Series)
        assert len(targets) == len(sample_ohlcv_data)
        
        # Should contain only valid target values (0, 1, 2)
        unique_targets = set(targets.dropna().unique())
        assert unique_targets.issubset({0, 1, 2})
        
        # Should have reasonable distribution (not all the same)
        assert len(unique_targets) > 1
    
    def test_handle_missing_values_forward_fill(self, data_processor):
        """Test forward fill missing value handling."""
        # Create data with missing values
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, np.nan, 50]
        })
        
        config.data.handle_missing_data = "forward_fill"
        result = data_processor._handle_missing_values(data)
        
        # Should not have any NaN values
        assert not result.isnull().any().any()
        
        # Should use forward fill logic
        assert result.loc[2, 'feature1'] == 2  # Forward filled
        assert result.loc[1, 'feature2'] == 10  # Forward filled
    
    def test_handle_missing_values_drop(self, data_processor):
        """Test drop missing value handling."""
        # Create data with missing values
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, np.nan, 50]
        })
        
        config.data.handle_missing_data = "drop"
        result = data_processor._handle_missing_values(data)
        
        # Should drop rows with NaN values
        assert len(result) < len(data)
        assert not result.isnull().any().any()
    
    def test_extract_static_features(self, data_processor, sample_ohlcv_data):
        """Test static feature extraction."""
        timestamp = sample_ohlcv_data.index[0]
        
        static_features = data_processor._extract_static_features(timestamp, sample_ohlcv_data)
        
        # Should return numpy array
        assert isinstance(static_features, np.ndarray)
        assert len(static_features) > 0
        
        # Should contain time-based features
        hour_feature = static_features[0]
        assert 0 <= hour_feature <= 23
        
        # Should contain cyclical encoding
        sin_features = static_features[4:6]  # sin encodings
        cos_features = static_features[6:8]  # cos encodings
        assert all(-1 <= f <= 1 for f in sin_features)
        assert all(-1 <= f <= 1 for f in cos_features)
    
    @patch('src.data_processor.talib')
    def test_process_data_full_pipeline(self, mock_talib, data_processor, sample_ohlcv_data):
        """Test the complete data processing pipeline."""
        # Mock all talib functions
        n_samples = len(sample_ohlcv_data)
        mock_talib.SMA.return_value = np.random.randn(n_samples)
        mock_talib.EMA.return_value = np.random.randn(n_samples)
        mock_talib.RSI.return_value = np.random.randn(n_samples) * 50 + 50
        mock_talib.MACD.return_value = (
            np.random.randn(n_samples),
            np.random.randn(n_samples), 
            np.random.randn(n_samples)
        )
        mock_talib.STOCH.return_value = (
            np.random.randn(n_samples) * 50 + 50,
            np.random.randn(n_samples) * 50 + 50
        )
        mock_talib.BBANDS.return_value = (
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples)
        )
        mock_talib.ATR.return_value = np.random.randn(n_samples)
        mock_talib.OBV.return_value = np.random.randn(n_samples)
        mock_talib.AD.return_value = np.random.randn(n_samples)
        
        # Set up additional mocks for other indicators
        for attr_name in dir(mock_talib):
            if attr_name.isupper() and not hasattr(getattr(mock_talib, attr_name), 'return_value'):
                getattr(mock_talib, attr_name).return_value = np.random.randn(n_samples)
        
        # Process data
        features, targets, feature_names = data_processor.process_data(
            sample_ohlcv_data, fit_transformers=True
        )
        
        # Should return proper types
        assert isinstance(features, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert isinstance(feature_names, list)
        
        # Should have reasonable dimensions
        assert features.ndim == 2
        assert len(targets) == features.shape[0]
        assert len(feature_names) == features.shape[1]
        
        # Features should be scaled (not all zeros)
        assert not np.allclose(features, 0)
        
        # Targets should be valid classes
        unique_targets = set(targets[~np.isnan(targets)])
        assert unique_targets.issubset({0, 1, 2})
    
    def test_feature_selection(self, data_processor, sample_processed_features):
        """Test feature selection functionality."""
        features, targets, feature_names = sample_processed_features
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Set configuration for feature selection
        config.data.max_features = 50
        config.data.feature_selection_method = "mutual_info"
        
        selected_features = data_processor._select_features(
            features_df, targets, fit_selector=True
        )
        
        # Should reduce number of features
        assert selected_features.shape[1] <= config.data.max_features
        assert selected_features.shape[0] == features_df.shape[0]
        
        # Should have feature selector fitted
        assert data_processor.feature_selector is not None
    
    def test_scale_features(self, data_processor, sample_processed_features):
        """Test feature scaling functionality."""
        features, _, feature_names = sample_processed_features
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Test different scaling methods
        for method in ["minmax", "standard", "robust"]:
            config.data.scaling_method = method
            
            scaled_features = data_processor._scale_features(features_df, fit_scaler=True)
            
            # Should return numpy array
            assert isinstance(scaled_features, np.ndarray)
            assert scaled_features.shape == features.shape
            
            # Should have scaler fitted
            assert data_processor.scaler is not None
            
            # Features should be scaled (different from original)
            assert not np.allclose(scaled_features, features)


@pytest.mark.unit
class TestDataProcessorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_insufficient_data(self, data_processor):
        """Test handling of insufficient data."""
        # Create minimal dataset
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        # Should handle gracefully
        try:
            features, targets, feature_names = data_processor.process_data(
                small_data, fit_transformers=True
            )
            # If it succeeds, should return reasonable results
            assert isinstance(features, np.ndarray)
            assert isinstance(targets, np.ndarray)
        except Exception as e:
            # If it fails, should be a reasonable exception
            assert "insufficient" in str(e).lower() or "minimum" in str(e).lower()
    
    def test_invalid_data_types(self, data_processor):
        """Test handling of invalid data types."""
        # Create data with string values
        invalid_data = pd.DataFrame({
            'open': ['100', '101', '102'],
            'high': ['102', '103', '104'],
            'low': ['99', '100', '101'],
            'close': ['101', '102', '103'],
            'volume': ['1000', '1100', '1200']
        })
        
        # Should either convert or raise appropriate error
        try:
            # Try to convert to numeric first
            for col in invalid_data.columns:
                invalid_data[col] = pd.to_numeric(invalid_data[col])
            
            features, targets, feature_names = data_processor.process_data(
                invalid_data, fit_transformers=True
            )
            assert isinstance(features, np.ndarray)
        except (ValueError, TypeError) as e:
            # Expected for invalid data
            assert True
    
    def test_empty_dataframe(self, data_processor):
        """Test handling of empty DataFrame."""
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError, KeyError)):
            data_processor.process_data(empty_data, fit_transformers=True)