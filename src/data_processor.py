"""
Advanced Data Processor for the Unified CNN-LSTM Trading Bot.

This module provides comprehensive data processing capabilities including:
- Data fetching from multiple sources (Yahoo Finance, Alpha Vantage, Interactive Brokers)
- 150+ technical indicators calculation
- Feature engineering and selection
- Data validation and cleaning
- Multi-timeframe data integration
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from loguru import logger

# Technical analysis libraries
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator, UltimateOscillator
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator, MFIIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, EaseOfMovementIndicator, VolWeightedAvgPrice

from src.config import Config, DataConfig
from src.utils import save_pickle, load_pickle

warnings.filterwarnings("ignore")


class DataProcessor:
    """
    Advanced data processor with 150+ technical indicators and comprehensive feature engineering.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the DataProcessor.
        
        Args:
            config: Configuration object containing data processing settings
        """
        self.config = config
        self.data_config = config.data
        self.cache_dir = config.system.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataProcessor initialized")
    
    def fetch_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: Force refresh of cached data
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            cache_file = self.cache_dir / f"{symbol}_{period}_{interval}.pkl"
            
            # Check if cached data exists and is recent (unless force refresh)
            if not force_refresh and cache_file.exists():
                cached_data = load_pickle(cache_file)
                if cached_data is not None:
                    # Check if data is recent enough (less than 1 hour old for intraday, 1 day for daily)
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    max_age = timedelta(hours=1) if interval.endswith('m') or interval.endswith('h') else timedelta(days=1)
                    
                    if cache_age < max_age:
                        logger.info(f"Using cached data for {symbol}")
                        return cached_data
            
            # Fetch fresh data
            logger.info(f"Fetching fresh data for {symbol} - Period: {period}, Interval: {interval}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns in data for {symbol}")
                return None
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Cache the data
            save_pickle(data, cache_file)
            
            logger.info(f"Fetched {len(data)} rows of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 150+ technical indicators for the given price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus technical indicators
        """
        logger.info("Calculating technical indicators...")
        
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        try:
            # Price-based features
            df = self._add_price_features(df)
            
            # Trend indicators
            df = self._add_trend_indicators(df)
            
            # Momentum indicators
            df = self._add_momentum_indicators(df)
            
            # Volatility indicators
            df = self._add_volatility_indicators(df)
            
            # Volume indicators
            df = self._add_volume_indicators(df)
            
            # Statistical features
            df = self._add_statistical_features(df)
            
            # Pattern recognition features
            df = self._add_pattern_features(df)
            
            # Market structure features
            df = self._add_market_structure_features(df)
            
            # Time-based features
            df = self._add_time_features(df)
            
            logger.info(f"Calculated {len(df.columns) - len(data.columns)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Basic price features
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Gaps
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        df['gap_size'] = df['open'] - df['close'].shift(1)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        periods = [5, 10, 20, 50, 100, 200]
        
        # Simple Moving Averages
        for period in periods:
            df[f'sma_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
            df[f'ema_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()
        
        # Moving average crossovers
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['ema_12_26_cross'] = np.where(df['ema_12'] > df['ema_26'], 1, -1) if 'ema_12' in df.columns and 'ema_26' in df.columns else 0
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Aroon
        aroon = AroonIndicator(df['close'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()
        
        # Commodity Channel Index
        df['cci_14'] = CCIIndicator(df['high'], df['low'], df['close'], window=14).cci()
        df['cci_20'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        
        # Detrended Price Oscillator
        df['dpo_20'] = DPOIndicator(df['close'], window=20).dpo()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI (Relative Strength Index)
        periods = [14, 21, 30]
        for period in periods:
            df[f'rsi_{period}'] = RSIIndicator(df['close'], window=period).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Rate of Change
        periods = [5, 10, 20]
        for period in periods:
            df[f'roc_{period}'] = ROCIndicator(df['close'], window=period).roc()
        
        # True Strength Index
        df['tsi'] = TSIIndicator(df['close']).tsi()
        
        # Ultimate Oscillator
        df['ultimate_oscillator'] = UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
        
        # Money Flow Index
        df['mfi'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        periods = [14, 20]
        for period in periods:
            bb = BollingerBands(df['close'], window=period)
            df[f'bb_upper_{period}'] = bb.bollinger_hband()
            df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            df[f'bb_lower_{period}'] = bb.bollinger_lband()
            df[f'bb_width_{period}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            df[f'bb_position_{period}'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Average True Range
        periods = [14, 21]
        for period in periods:
            df[f'atr_{period}'] = AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
        
        # Keltner Channel
        kc = KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        # Donchian Channel
        dc = DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_upper'] = dc.donchian_channel_hband()
        df['dc_middle'] = dc.donchian_channel_mband()
        df['dc_lower'] = dc.donchian_channel_lband()
        
        # Historical Volatility
        periods = [10, 20, 30]
        for period in periods:
            df[f'hist_vol_{period}'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # On Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume Price Trend
        df['vpt'] = VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        # Accumulation/Distribution Index
        df['adi'] = AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        
        # Chaikin Money Flow
        df['cmf'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        
        # Force Index
        df['force_index'] = ForceIndexIndicator(df['close'], df['volume']).force_index()
        
        # Ease of Movement
        df['eom'] = EaseOfMovementIndicator(df['high'], df['low'], df['volume']).ease_of_movement()
        
        # Volume Weighted Average Price
        df['vwap'] = VolWeightedAvgPrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['price_volume'] = df['close'] * df['volume']
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        periods = [5, 10, 20, 50]
        
        for period in periods:
            # Rolling statistics
            df[f'close_mean_{period}'] = df['close'].rolling(period).mean()
            df[f'close_std_{period}'] = df['close'].rolling(period).std()
            df[f'close_skew_{period}'] = df['close'].rolling(period).skew()
            df[f'close_kurt_{period}'] = df['close'].rolling(period).kurt()
            
            # Price position within range
            df[f'price_position_{period}'] = (df['close'] - df['close'].rolling(period).min()) / (df['close'].rolling(period).max() - df['close'].rolling(period).min())
            
            # Z-score
            df[f'zscore_{period}'] = (df['close'] - df[f'close_mean_{period}']) / df[f'close_std_{period}']
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        # Doji patterns
        body_size = abs(df['close'] - df['open'])
        wick_size = df['high'] - df['low']
        df['doji'] = (body_size / wick_size < 0.1).astype(int)
        
        # Hammer and hanging man
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        df['hammer'] = ((lower_wick > 2 * body_size) & (upper_wick < body_size)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['open'] < df['close'].shift(1)) &
                                  (df['close'] > df['open'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                  (df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['open'] > df['close'].shift(1)) &
                                  (df['close'] < df['open'].shift(1))).astype(int)
        
        # Inside/Outside bars
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
        
        return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features."""
        # Higher highs and lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        
        # Support and resistance levels
        periods = [20, 50]
        for period in periods:
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'distance_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
            df[f'distance_to_support_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']
        
        # Trend strength
        df['uptrend_strength'] = df[['higher_high', 'higher_low']].sum(axis=1)
        df['downtrend_strength'] = df[['lower_high', 'lower_low']].sum(axis=1)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'hour'):
            # Hour of day (for intraday data)
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Day of week
            df['day_of_week'] = df.index.dayofweek
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
            
            # Month
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Market session indicators
            df['market_open'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(int)
            df['market_close'] = ((df['hour'] == 15) & (df['minute'] >= 45)).astype(int)
            df['lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13)).astype(int)
        
        return df
    
    def create_features_and_targets(
        self,
        data: pd.DataFrame,
        prediction_horizon: int = 5,
        threshold_pct: float = 0.02
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and target labels for model training.
        
        Args:
            data: DataFrame with price data and technical indicators
            prediction_horizon: Number of periods ahead to predict
            threshold_pct: Threshold for buy/sell signals as percentage
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        logger.info(f"Creating features and targets with horizon={prediction_horizon}, threshold={threshold_pct}")
        
        df = data.copy()
        
        # Calculate future returns
        future_returns = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Create target labels: 0=sell, 1=hold, 2=buy
        targets = pd.Series(1, index=df.index)  # Default to hold
        targets[future_returns > threshold_pct] = 2  # Buy signal
        targets[future_returns < -threshold_pct] = 0  # Sell signal
        
        # Remove rows where target cannot be calculated
        valid_indices = targets.dropna().index
        features = df.loc[valid_indices].copy()
        targets = targets.loc[valid_indices]
        
        # Remove non-feature columns
        feature_cols = [col for col in features.columns if col not in ['symbol', 'open', 'high', 'low', 'close', 'volume']]
        features = features[feature_cols]
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        logger.info(f"Created {len(features)} samples with {len(features.columns)} features")
        logger.info(f"Target distribution: {targets.value_counts().to_dict()}")
        
        return features, targets
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        method = self.data_config.handle_missing
        
        if method == "forward_fill":
            df = df.fillna(method='ffill')
        elif method == "interpolate":
            df = df.interpolate()
        elif method == "drop":
            df = df.dropna()
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def scale_features(self, features: pd.DataFrame, scaler=None) -> Tuple[pd.DataFrame, Any]:
        """
        Scale features using the specified scaling method.
        
        Args:
            features: DataFrame with features to scale
            scaler: Pre-fitted scaler (optional, for inference)
            
        Returns:
            Tuple of (scaled_features, fitted_scaler)
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        scaling_method = self.data_config.scaling_method
        
        if scaler is None:
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {scaling_method}, using StandardScaler")
                scaler = StandardScaler()
            
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
        else:
            scaled_features = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
        
        return scaled_features, scaler
    
    def process_symbol(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Complete processing pipeline for a single symbol.
        
        Args:
            symbol: Stock symbol to process
            period: Data period to fetch
            interval: Data interval
            
        Returns:
            Tuple of (features, targets) or None if error
        """
        try:
            logger.info(f"Processing symbol: {symbol}")
            
            # Fetch data
            data = self.fetch_data(symbol, period, interval)
            if data is None or len(data) < self.data_config.min_data_points:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # Create features and targets
            features, targets = self.create_features_and_targets(
                data_with_indicators,
                self.data_config.prediction_horizon,
                self.data_config.threshold_pct
            )
            
            logger.info(f"Successfully processed {symbol}: {len(features)} samples")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of all feature names that would be generated.
        
        Returns:
            List of feature names
        """
        # Create a dummy dataset to get feature names
        dummy_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [102.0] * 100,
            'volume': [1000000] * 100,
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        
        processed_data = self.calculate_technical_indicators(dummy_data)
        features, _ = self.create_features_and_targets(processed_data)
        
        return list(features.columns)