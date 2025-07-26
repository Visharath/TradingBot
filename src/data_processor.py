"""
Advanced Data Processing for the Unified CNN-LSTM Trading Bot.

This module implements comprehensive data processing including:
- 150+ technical indicators
- Statistical features
- Pattern recognition
- Data validation and cleaning
- Feature selection and scaling
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .config import config
from .utils import logger, timing_decorator, validate_data_quality


class DataProcessor:
    """
    Advanced data processor with 150+ technical indicators and comprehensive
    feature engineering capabilities.
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.processed_features = []
        
    @timing_decorator
    def process_data(
        self,
        data: pd.DataFrame,
        target_column: str = "target",
        fit_transformers: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete data processing pipeline.
        
        Args:
            data: Raw market data DataFrame
            target_column: Name of target column
            fit_transformers: Whether to fit scalers and selectors
        
        Returns:
            Tuple of (features, targets, feature_names)
        """
        logger.info("Starting comprehensive data processing pipeline")
        
        # Validate data quality
        quality_report = validate_data_quality(
            data, 
            required_columns=["open", "high", "low", "close", "volume"]
        )
        logger.info(f"Data quality score: {quality_report['quality_score']:.2f}")
        
        # Clean data
        data_clean = self._clean_data(data)
        
        # Generate all features
        features_df = self._generate_all_features(data_clean)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Create targets if not provided
        if target_column not in features_df.columns:
            features_df[target_column] = self._create_targets(data_clean)
        
        # Split features and targets
        targets = features_df[target_column].values
        features_df = features_df.drop(columns=[target_column])
        
        # Feature selection
        if config.data.max_features and fit_transformers:
            features_df = self._select_features(features_df, targets, fit_transformers)
        elif not fit_transformers and self.feature_selector:
            features_df = features_df[self.feature_selector.get_feature_names_out()]
        
        # Scale features
        features_scaled = self._scale_features(features_df, fit_transformers)
        
        # Update feature names
        self.feature_names = list(features_df.columns)
        
        logger.info(f"Processed {features_scaled.shape[0]} samples with {features_scaled.shape[1]} features")
        
        return features_scaled, targets, self.feature_names
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate input data."""
        data_clean = data.copy()
        
        # Remove duplicates
        data_clean = data_clean.drop_duplicates()
        
        # Sort by datetime index
        if isinstance(data_clean.index, pd.DatetimeIndex):
            data_clean = data_clean.sort_index()
        
        # Basic OHLCV validation
        invalid_mask = (
            (data_clean['high'] < data_clean['low']) |
            (data_clean['high'] < data_clean['open']) |
            (data_clean['high'] < data_clean['close']) |
            (data_clean['low'] > data_clean['open']) |
            (data_clean['low'] > data_clean['close']) |
            (data_clean['volume'] < 0)
        )
        
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} invalid OHLCV rows, removing...")
            data_clean = data_clean[~invalid_mask]
        
        return data_clean
    
    def _generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all 150+ technical indicators and features."""
        features = pd.DataFrame(index=data.index)
        
        # Price and volume arrays
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volume = data['volume'].values
        
        # === TREND INDICATORS ===
        features.update(self._trend_indicators(open_prices, high_prices, low_prices, close_prices, volume))
        
        # === MOMENTUM INDICATORS ===
        features.update(self._momentum_indicators(open_prices, high_prices, low_prices, close_prices, volume))
        
        # === VOLATILITY INDICATORS ===
        features.update(self._volatility_indicators(open_prices, high_prices, low_prices, close_prices))
        
        # === VOLUME INDICATORS ===
        features.update(self._volume_indicators(high_prices, low_prices, close_prices, volume))
        
        # === CYCLE INDICATORS ===
        features.update(self._cycle_indicators(high_prices, low_prices, close_prices))
        
        # === PATTERN RECOGNITION ===
        if config.data.enable_pattern_recognition:
            features.update(self._pattern_recognition(open_prices, high_prices, low_prices, close_prices))
        
        # === STATISTICAL FEATURES ===
        if config.data.enable_statistical_features:
            features.update(self._statistical_features(close_prices))
        
        # === SUPPORT/RESISTANCE LEVELS ===
        features.update(self._support_resistance_levels(high_prices, low_prices, close_prices))
        
        # === MARKET REGIME FEATURES ===
        features.update(self._market_regime_features(close_prices, volume))
        
        # === TIME-BASED FEATURES ===
        features.update(self._time_based_features(data.index))
        
        return features
    
    def _trend_indicators(self, open_p, high_p, low_p, close_p, volume) -> pd.DataFrame:
        """Calculate trend-following indicators."""
        indicators = {}
        
        # Simple Moving Averages
        for period in config.data.sma_periods:
            indicators[f'sma_{period}'] = talib.SMA(close_p, timeperiod=period)
            indicators[f'sma_{period}_ratio'] = close_p / talib.SMA(close_p, timeperiod=period)
        
        # Exponential Moving Averages
        for period in config.data.ema_periods:
            indicators[f'ema_{period}'] = talib.EMA(close_p, timeperiod=period)
            indicators[f'ema_{period}_ratio'] = close_p / talib.EMA(close_p, timeperiod=period)
        
        # Weighted Moving Average
        indicators['wma_10'] = talib.WMA(close_p, timeperiod=10)
        indicators['wma_20'] = talib.WMA(close_p, timeperiod=20)
        
        # Double Exponential Moving Average
        indicators['dema_21'] = talib.DEMA(close_p, timeperiod=21)
        
        # Triple Exponential Moving Average
        indicators['tema_21'] = talib.TEMA(close_p, timeperiod=21)
        
        # Triangular Moving Average
        indicators['trima_21'] = talib.TRIMA(close_p, timeperiod=21)
        
        # Kaufman Adaptive Moving Average
        indicators['kama_30'] = talib.KAMA(close_p, timeperiod=30)
        
        # MESA Adaptive Moving Average
        indicators['mama'], indicators['fama'] = talib.MAMA(close_p)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_p, *config.data.macd_periods)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        indicators['macd_ratio'] = macd / macd_signal
        
        # Average Directional Index
        indicators['adx'] = talib.ADX(high_p, low_p, close_p, timeperiod=14)
        indicators['adxr'] = talib.ADXR(high_p, low_p, close_p, timeperiod=14)
        
        # Directional Movement Index
        indicators['plus_di'] = talib.PLUS_DI(high_p, low_p, close_p, timeperiod=14)
        indicators['minus_di'] = talib.MINUS_DI(high_p, low_p, close_p, timeperiod=14)
        indicators['dx'] = talib.DX(high_p, low_p, close_p, timeperiod=14)
        
        # Parabolic SAR
        indicators['sar'] = talib.SAR(high_p, low_p)
        indicators['sar_ext'] = talib.SAREXT(high_p, low_p)
        
        # Ichimoku Cloud components
        indicators['tenkan_sen'] = (pd.Series(high_p).rolling(9).max() + pd.Series(low_p).rolling(9).min()) / 2
        indicators['kijun_sen'] = (pd.Series(high_p).rolling(26).max() + pd.Series(low_p).rolling(26).min()) / 2
        indicators['senkou_span_a'] = (indicators['tenkan_sen'] + indicators['kijun_sen']) / 2
        indicators['chikou_span'] = pd.Series(close_p).shift(-26)
        
        return pd.DataFrame(indicators)
    
    def _momentum_indicators(self, open_p, high_p, low_p, close_p, volume) -> pd.DataFrame:
        """Calculate momentum indicators."""
        indicators = {}
        
        # Relative Strength Index
        indicators[f'rsi_{config.data.rsi_period}'] = talib.RSI(close_p, timeperiod=config.data.rsi_period)
        indicators['rsi_6'] = talib.RSI(close_p, timeperiod=6)
        indicators['rsi_21'] = talib.RSI(close_p, timeperiod=21)
        
        # Stochastic Oscillators
        indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high_p, low_p, close_p)
        indicators['stochf_k'], indicators['stochf_d'] = talib.STOCHF(high_p, low_p, close_p)
        indicators['stochrsi_k'], indicators['stochrsi_d'] = talib.STOCHRSI(close_p)
        
        # Williams %R
        indicators['willr'] = talib.WILLR(high_p, low_p, close_p, timeperiod=14)
        
        # Rate of Change
        indicators['roc_10'] = talib.ROC(close_p, timeperiod=10)
        indicators['roc_20'] = talib.ROC(close_p, timeperiod=20)
        indicators['rocp_10'] = talib.ROCP(close_p, timeperiod=10)
        indicators['rocr_10'] = talib.ROCR(close_p, timeperiod=10)
        
        # Money Flow Index
        indicators['mfi'] = talib.MFI(high_p, low_p, close_p, volume, timeperiod=14)
        
        # Commodity Channel Index
        indicators['cci'] = talib.CCI(high_p, low_p, close_p, timeperiod=14)
        
        # Aroon
        indicators['aroon_down'], indicators['aroon_up'] = talib.AROON(high_p, low_p, timeperiod=14)
        indicators['aroon_osc'] = talib.AROONOSC(high_p, low_p, timeperiod=14)
        
        # Balance of Power
        indicators['bop'] = talib.BOP(open_p, high_p, low_p, close_p)
        
        # Chande Momentum Oscillator
        indicators['cmo'] = talib.CMO(close_p, timeperiod=14)
        
        # Momentum
        indicators['mom_10'] = talib.MOM(close_p, timeperiod=10)
        indicators['mom_20'] = talib.MOM(close_p, timeperiod=20)
        
        # Plus/Minus Directional Movement
        indicators['plus_dm'] = talib.PLUS_DM(high_p, low_p, timeperiod=14)
        indicators['minus_dm'] = talib.MINUS_DM(high_p, low_p, timeperiod=14)
        
        # Percentage Price Oscillator
        indicators['ppo'] = talib.PPO(close_p)
        
        # Trix
        indicators['trix'] = talib.TRIX(close_p, timeperiod=14)
        
        # Ultimate Oscillator
        indicators['ultosc'] = talib.ULTOSC(high_p, low_p, close_p)
        
        return pd.DataFrame(indicators)
    
    def _volatility_indicators(self, open_p, high_p, low_p, close_p) -> pd.DataFrame:
        """Calculate volatility indicators."""
        indicators = {}
        
        # Average True Range
        indicators['atr'] = talib.ATR(high_p, low_p, close_p, timeperiod=14)
        indicators['natr'] = talib.NATR(high_p, low_p, close_p, timeperiod=14)
        indicators['trange'] = talib.TRANGE(high_p, low_p, close_p)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_p, 
            timeperiod=config.data.bollinger_period,
            nbdevup=config.data.bollinger_std,
            nbdevdn=config.data.bollinger_std
        )
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        indicators['bb_position'] = (close_p - bb_lower) / (bb_upper - bb_lower)
        
        # Keltner Channels (approximation using ATR)
        ema_20 = talib.EMA(close_p, timeperiod=20)
        atr_10 = talib.ATR(high_p, low_p, close_p, timeperiod=10)
        indicators['keltner_upper'] = ema_20 + (2 * atr_10)
        indicators['keltner_lower'] = ema_20 - (2 * atr_10)
        indicators['keltner_width'] = 4 * atr_10 / ema_20
        
        # Donchian Channels
        indicators['donchian_upper'] = pd.Series(high_p).rolling(20).max()
        indicators['donchian_lower'] = pd.Series(low_p).rolling(20).min()
        indicators['donchian_middle'] = (indicators['donchian_upper'] + indicators['donchian_lower']) / 2
        
        # Historical Volatility
        returns = pd.Series(close_p).pct_change()
        indicators['volatility_10'] = returns.rolling(10).std() * np.sqrt(252)
        indicators['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        indicators['volatility_30'] = returns.rolling(30).std() * np.sqrt(252)
        
        return pd.DataFrame(indicators)
    
    def _volume_indicators(self, high_p, low_p, close_p, volume) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        indicators = {}
        
        # On Balance Volume
        indicators['obv'] = talib.OBV(close_p, volume)
        
        # Accumulation/Distribution Line
        indicators['ad'] = talib.AD(high_p, low_p, close_p, volume)
        indicators['adosc'] = talib.ADOSC(high_p, low_p, close_p, volume)
        
        # Volume indicators
        indicators['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
        indicators['volume_ratio'] = volume / talib.SMA(volume, timeperiod=20)
        
        # VWAP approximation
        typical_price = (high_p + low_p + close_p) / 3
        indicators['vwap'] = pd.Series(typical_price * volume).rolling(20).sum() / pd.Series(volume).rolling(20).sum()
        
        # Chaikin Money Flow
        money_flow_multiplier = ((close_p - low_p) - (high_p - close_p)) / (high_p - low_p)
        money_flow_volume = money_flow_multiplier * volume
        indicators['cmf'] = pd.Series(money_flow_volume).rolling(20).sum() / pd.Series(volume).rolling(20).sum()
        
        # Volume Price Trend
        indicators['vpt'] = (pd.Series(close_p).pct_change() * volume).cumsum()
        
        # Ease of Movement
        distance_moved = (high_p + low_p) / 2 - (pd.Series(high_p).shift(1) + pd.Series(low_p).shift(1)) / 2
        box_height = volume / (high_p - low_p)
        indicators['eom'] = distance_moved / box_height
        indicators['eom_ma'] = pd.Series(indicators['eom']).rolling(14).mean()
        
        return pd.DataFrame(indicators)
    
    def _cycle_indicators(self, high_p, low_p, close_p) -> pd.DataFrame:
        """Calculate cycle indicators."""
        indicators = {}
        
        # Hilbert Transform - Dominant Cycle Period
        indicators['ht_dcperiod'] = talib.HT_DCPERIOD(close_p)
        indicators['ht_dcphase'] = talib.HT_DCPHASE(close_p)
        
        # Hilbert Transform - Sine Wave
        indicators['ht_sine'], indicators['ht_leadsine'] = talib.HT_SINE(close_p)
        
        # Hilbert Transform - Trend vs Cycle Mode
        indicators['ht_trendmode'] = talib.HT_TRENDMODE(close_p)
        
        # Hilbert Transform - Phasor Components
        indicators['ht_phasor_inphase'], indicators['ht_phasor_quadrature'] = talib.HT_PHASOR(close_p)
        
        return pd.DataFrame(indicators)
    
    def _pattern_recognition(self, open_p, high_p, low_p, close_p) -> pd.DataFrame:
        """Calculate candlestick pattern recognition indicators."""
        indicators = {}
        
        # Major candlestick patterns
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ]
        
        for pattern in patterns:
            try:
                indicators[pattern.lower()] = getattr(talib, pattern)(open_p, high_p, low_p, close_p)
            except AttributeError:
                continue
        
        return pd.DataFrame(indicators)
    
    def _statistical_features(self, close_p) -> pd.DataFrame:
        """Calculate statistical features."""
        indicators = {}
        price_series = pd.Series(close_p)
        
        # Rolling statistics
        windows = [5, 10, 20, 50]
        for window in windows:
            rolling = price_series.rolling(window)
            indicators[f'mean_{window}'] = rolling.mean()
            indicators[f'std_{window}'] = rolling.std()
            indicators[f'skew_{window}'] = rolling.skew()
            indicators[f'kurt_{window}'] = rolling.kurt()
            indicators[f'median_{window}'] = rolling.median()
            indicators[f'var_{window}'] = rolling.var()
            
            # Quantiles
            indicators[f'q25_{window}'] = rolling.quantile(0.25)
            indicators[f'q75_{window}'] = rolling.quantile(0.75)
            
            # Z-scores
            indicators[f'zscore_{window}'] = (close_p - rolling.mean()) / rolling.std()
        
        # Returns analysis
        returns = price_series.pct_change()
        for window in [5, 10, 20]:
            rolling_returns = returns.rolling(window)
            indicators[f'returns_mean_{window}'] = rolling_returns.mean()
            indicators[f'returns_std_{window}'] = rolling_returns.std()
            indicators[f'returns_skew_{window}'] = rolling_returns.skew()
            indicators[f'returns_kurt_{window}'] = rolling_returns.kurt()
        
        # Autocorrelation
        for lag in [1, 2, 3, 5]:
            indicators[f'autocorr_lag_{lag}'] = price_series.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        return pd.DataFrame(indicators)
    
    def _support_resistance_levels(self, high_p, low_p, close_p) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        indicators = {}
        
        # Pivot Points
        pivot = (pd.Series(high_p).shift(1) + pd.Series(low_p).shift(1) + pd.Series(close_p).shift(1)) / 3
        indicators['pivot_point'] = pivot
        indicators['resistance_1'] = 2 * pivot - pd.Series(low_p).shift(1)
        indicators['support_1'] = 2 * pivot - pd.Series(high_p).shift(1)
        indicators['resistance_2'] = pivot + (pd.Series(high_p).shift(1) - pd.Series(low_p).shift(1))
        indicators['support_2'] = pivot - (pd.Series(high_p).shift(1) - pd.Series(low_p).shift(1))
        
        # Fibonacci retracement levels
        window = 20
        high_roll = pd.Series(high_p).rolling(window).max()
        low_roll = pd.Series(low_p).rolling(window).min()
        diff = high_roll - low_roll
        
        indicators['fib_23.6'] = high_roll - 0.236 * diff
        indicators['fib_38.2'] = high_roll - 0.382 * diff
        indicators['fib_50.0'] = high_roll - 0.500 * diff
        indicators['fib_61.8'] = high_roll - 0.618 * diff
        indicators['fib_78.6'] = high_roll - 0.786 * diff
        
        return pd.DataFrame(indicators)
    
    def _market_regime_features(self, close_p, volume) -> pd.DataFrame:
        """Calculate market regime features."""
        indicators = {}
        price_series = pd.Series(close_p)
        volume_series = pd.Series(volume)
        
        # Trend strength
        sma_50 = price_series.rolling(50).mean()
        sma_200 = price_series.rolling(200).mean()
        indicators['trend_strength'] = (sma_50 - sma_200) / sma_200
        
        # Market volatility regime
        returns = price_series.pct_change()
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(50).mean()
        indicators['volatility_regime'] = volatility / vol_ma
        
        # Volume regime
        vol_ma = volume_series.rolling(20).mean()
        vol_ma_long = volume_series.rolling(50).mean()
        indicators['volume_regime'] = vol_ma / vol_ma_long
        
        # Price momentum regime
        momentum_10 = close_p / pd.Series(close_p).shift(10) - 1
        momentum_ma = pd.Series(momentum_10).rolling(20).mean()
        indicators['momentum_regime'] = momentum_10 / momentum_ma
        
        return pd.DataFrame(indicators)
    
    def _time_based_features(self, datetime_index) -> pd.DataFrame:
        """Calculate time-based features."""
        indicators = {}
        
        if isinstance(datetime_index, pd.DatetimeIndex):
            indicators['hour'] = datetime_index.hour
            indicators['day_of_week'] = datetime_index.dayofweek
            indicators['day_of_month'] = datetime_index.day
            indicators['month'] = datetime_index.month
            indicators['quarter'] = datetime_index.quarter
            indicators['is_month_end'] = datetime_index.is_month_end.astype(int)
            indicators['is_quarter_end'] = datetime_index.is_quarter_end.astype(int)
            
            # Cyclical encoding
            indicators['hour_sin'] = np.sin(2 * np.pi * datetime_index.hour / 24)
            indicators['hour_cos'] = np.cos(2 * np.pi * datetime_index.hour / 24)
            indicators['day_sin'] = np.sin(2 * np.pi * datetime_index.dayofweek / 7)
            indicators['day_cos'] = np.cos(2 * np.pi * datetime_index.dayofweek / 7)
            indicators['month_sin'] = np.sin(2 * np.pi * datetime_index.month / 12)
            indicators['month_cos'] = np.cos(2 * np.pi * datetime_index.month / 12)
        
        return pd.DataFrame(indicators, index=datetime_index)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if config.data.handle_missing_data == "forward_fill":
            return data.fillna(method='ffill').fillna(method='bfill')
        elif config.data.handle_missing_data == "interpolate":
            return data.interpolate(method='linear').fillna(method='bfill')
        elif config.data.handle_missing_data == "drop":
            return data.dropna()
        else:
            return data.fillna(0)
    
    def _create_targets(self, data: pd.DataFrame) -> pd.Series:
        """Create trading targets (Buy=0, Hold=1, Sell=2)."""
        returns = data['close'].pct_change(periods=config.model.prediction_horizon)
        
        # Define thresholds for classification
        buy_threshold = returns.quantile(0.7)
        sell_threshold = returns.quantile(0.3)
        
        targets = pd.Series(1, index=data.index)  # Default to Hold
        targets[returns > buy_threshold] = 0  # Buy
        targets[returns < sell_threshold] = 2  # Sell
        
        return targets
    
    def _select_features(
        self, 
        features: pd.DataFrame, 
        targets: np.ndarray, 
        fit_selector: bool = True
    ) -> pd.DataFrame:
        """Select most important features."""
        if fit_selector:
            # Choose selection method
            if config.data.feature_selection_method == "mutual_info":
                self.feature_selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=config.data.max_features
                )
            elif config.data.feature_selection_method == "chi2":
                self.feature_selector = SelectKBest(
                    score_func=chi2,
                    k=config.data.max_features
                )
            else:
                self.feature_selector = SelectKBest(
                    score_func=f_classif,
                    k=config.data.max_features
                )
            
            # Remove rows with NaN targets for fitting
            valid_mask = ~np.isnan(targets)
            features_valid = features[valid_mask]
            targets_valid = targets[valid_mask]
            
            self.feature_selector.fit(features_valid, targets_valid)
        
        if self.feature_selector is not None:
            selected_features = self.feature_selector.transform(features)
            feature_names = features.columns[self.feature_selector.get_support()]
            return pd.DataFrame(selected_features, columns=feature_names, index=features.index)
        
        return features
    
    def _scale_features(self, features: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Scale features using specified method."""
        if fit_scaler:
            if config.data.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif config.data.scaling_method == "standard":
                self.scaler = StandardScaler()
            else:
                self.scaler = RobustScaler()
            
            # Fit scaler on non-NaN values
            valid_data = features.dropna()
            if len(valid_data) > 0:
                self.scaler.fit(valid_data)
        
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features.fillna(0))
            return scaled_features
        
        return features.fillna(0).values


__all__ = ["DataProcessor"]