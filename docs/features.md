# Feature Engineering Documentation

This document provides comprehensive details about the 150+ technical indicators and feature engineering capabilities of the Unified CNN-LSTM Trading Bot.

## Overview

The bot implements one of the most comprehensive technical analysis libraries available, with over 150 technical indicators across multiple categories. These features are designed to capture various aspects of market behavior, from short-term price movements to long-term trends and market regimes.

## Feature Categories

### 1. Trend Following Indicators (25+ features)

#### Simple Moving Averages (SMA)
- **Periods**: 5, 10, 20, 50, 100, 200
- **Purpose**: Identify trend direction and strength
- **Usage**: Crossovers signal trend changes

```python
# SMA calculation example
sma_20 = talib.SMA(close_prices, timeperiod=20)
sma_ratio = close_prices / sma_20  # Price relative to trend
```

#### Exponential Moving Averages (EMA)
- **Periods**: 12, 26, 50, 100
- **Purpose**: React faster to price changes than SMA
- **Usage**: More responsive to recent price action

#### MACD Family
- **Components**: MACD line, Signal line, Histogram
- **Parameters**: (12, 26, 9) default periods
- **Purpose**: Trend momentum and crossover signals

#### Advanced Moving Averages
- **DEMA**: Double Exponential Moving Average
- **TEMA**: Triple Exponential Moving Average
- **TRIMA**: Triangular Moving Average
- **KAMA**: Kaufman Adaptive Moving Average
- **MAMA/FAMA**: MESA Adaptive Moving Averages

#### Directional Movement System
- **ADX**: Average Directional Index (trend strength)
- **+DI/-DI**: Directional Indicators
- **DX**: Directional Movement Index
- **ADXR**: ADX Rating

#### Parabolic SAR
- **Purpose**: Trailing stop and trend reversal
- **Variations**: Standard SAR and Extended SAR

#### Ichimoku Cloud Components
- **Tenkan-sen**: Conversion line (9-period)
- **Kijun-sen**: Base line (26-period)  
- **Senkou Span A**: Leading span A
- **Chikou Span**: Lagging span (26-period)

### 2. Momentum Oscillators (30+ features)

#### Relative Strength Index (RSI)
- **Periods**: 6, 14, 21
- **Range**: 0-100
- **Purpose**: Overbought/oversold conditions

#### Stochastic Oscillators
- **Fast Stochastic**: %K and %D
- **Slow Stochastic**: Smoothed version
- **Stochastic RSI**: RSI applied to Stochastic

#### Williams %R
- **Range**: -100 to 0
- **Purpose**: Overbought/oversold momentum
- **Period**: 14 (default)

#### Rate of Change (ROC) Family
- **ROC**: Standard rate of change
- **ROCP**: Percentage price change
- **ROCR**: Ratio price change
- **Periods**: 10, 20

#### Money Flow Index (MFI)
- **Range**: 0-100
- **Purpose**: Volume-weighted RSI
- **Period**: 14

#### Commodity Channel Index (CCI)
- **Purpose**: Cyclical trend identification
- **Typical values**: ±100 thresholds

#### Aroon Indicators
- **Aroon Up/Down**: Trend strength and direction
- **Aroon Oscillator**: Difference between Up and Down
- **Period**: 14

#### Additional Momentum Indicators
- **Balance of Power (BOP)**: Price vs. volume relationship
- **Chande Momentum Oscillator (CMO)**: Momentum with volume
- **Momentum (MOM)**: Simple price momentum
- **Plus/Minus Directional Movement**: Trend components
- **Percentage Price Oscillator (PPO)**: MACD percentage version
- **Trix**: Triple exponential smoothing
- **Ultimate Oscillator**: Multi-timeframe momentum

### 3. Volatility Indicators (20+ features)

#### Average True Range (ATR)
- **ATR**: Absolute volatility measure
- **NATR**: Normalized ATR (percentage)
- **TRANGE**: True Range
- **Period**: 14

#### Bollinger Bands
- **Components**: Upper, Middle, Lower bands
- **Parameters**: 20-period, 2 standard deviations
- **Derived features**: Band width, price position

```python
# Bollinger Bands calculation
bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, 20, 2, 2)
bb_width = (bb_upper - bb_lower) / bb_middle
bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower)
```

#### Keltner Channels
- **Based on**: EMA and ATR
- **Components**: Upper, Lower, Middle
- **Purpose**: Volatility-based support/resistance

#### Donchian Channels
- **Based on**: Highest high and lowest low
- **Period**: 20
- **Purpose**: Breakout identification

#### Historical Volatility
- **Periods**: 10, 20, 30 days
- **Calculation**: Rolling standard deviation of returns
- **Annualized**: Multiplied by √252

### 4. Volume Indicators (15+ features)

#### On Balance Volume (OBV)
- **Purpose**: Volume momentum
- **Calculation**: Cumulative volume based on price direction

#### Accumulation/Distribution Line
- **AD**: Basic accumulation/distribution
- **ADOSC**: AD Oscillator
- **Purpose**: Money flow analysis

#### Volume Ratios
- **Volume SMA**: Moving average of volume
- **Volume Ratio**: Current vs. average volume
- **Purpose**: Unusual volume detection

#### Volume-Weighted Average Price (VWAP)
- **Calculation**: Price weighted by volume
- **Period**: 20 (rolling)
- **Purpose**: Fair value estimation

#### Chaikin Money Flow (CMF)
- **Formula**: Money flow over volume
- **Period**: 20
- **Range**: -1 to +1

#### Additional Volume Indicators
- **Volume Price Trend (VPT)**: Cumulative volume-price
- **Ease of Movement (EOM)**: Volume per price movement
- **EOM Moving Average**: Smoothed EOM

### 5. Cycle and Market Analysis (10+ features)

#### Hilbert Transform Indicators
- **HT_DCPERIOD**: Dominant cycle period
- **HT_DCPHASE**: Dominant cycle phase
- **HT_SINE/HT_LEADSINE**: Sine wave analysis
- **HT_TRENDMODE**: Trend vs. cycle mode
- **HT_PHASOR**: Phasor components

### 6. Pattern Recognition (50+ features)

#### Candlestick Patterns
The bot recognizes all major candlestick patterns including:

**Reversal Patterns:**
- Hammer, Hanging Man, Inverted Hammer
- Doji variations (Dragonfly, Gravestone, Long-legged)
- Engulfing patterns (Bullish/Bearish)
- Evening Star, Morning Star
- Dark Cloud Cover, Piercing Pattern

**Continuation Patterns:**
- Three White Soldiers, Three Black Crows
- Rising/Falling Three Methods
- Separating Lines
- Thrusting Pattern

**Complex Patterns:**
- Abandoned Baby
- Belt Hold
- Breakaway
- Concealing Baby Swallow
- Counterattack
- Gap SideSide White
- Harami (regular and cross)
- Hikkake (regular and modified)
- And many more...

### 7. Statistical Features (30+ features)

#### Rolling Statistics
For multiple windows (5, 10, 20, 50 periods):
- **Mean**: Average price
- **Standard Deviation**: Price volatility
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution tail heaviness
- **Median**: Middle value
- **Variance**: Price dispersion

#### Quantile Features
- **25th Percentile (Q1)**: First quartile
- **75th Percentile (Q3)**: Third quartile
- **Interquartile Range**: Q3 - Q1

#### Z-Scores
- **Purpose**: Standardized price position
- **Calculation**: (price - mean) / std
- **Windows**: 5, 10, 20, 50 periods

#### Returns Analysis
For multiple windows:
- **Returns Mean**: Average return
- **Returns Std**: Return volatility
- **Returns Skewness**: Return distribution shape
- **Returns Kurtosis**: Return distribution tails

#### Autocorrelation
- **Lags**: 1, 2, 3, 5 periods
- **Purpose**: Price momentum persistence
- **Window**: 50 periods (rolling)

### 8. Support and Resistance (10+ features)

#### Pivot Points
- **Classic Pivot**: (H + L + C) / 3
- **Resistance Levels**: R1, R2 (multiple methods)
- **Support Levels**: S1, S2 (multiple methods)

#### Fibonacci Retracements
- **Levels**: 23.6%, 38.2%, 50.0%, 61.8%, 78.6%
- **Based on**: 20-period high/low range
- **Purpose**: Potential reversal levels

### 9. Market Regime Features (10+ features)

#### Trend Strength
- **Calculation**: (SMA50 - SMA200) / SMA200
- **Purpose**: Overall market direction

#### Volatility Regime
- **Current vs. Historical**: Rolling volatility comparison
- **Threshold**: Configurable volatility levels
- **Purpose**: Risk adjustment

#### Volume Regime
- **Current vs. Average**: Volume comparison
- **Purpose**: Market participation level

#### Momentum Regime
- **10-period Momentum**: Price momentum
- **Relative to Average**: Momentum comparison
- **Purpose**: Trend acceleration

### 10. Time-Based Features (12+ features)

#### Temporal Features
- **Hour**: Hour of day (0-23)
- **Day of Week**: Day index (0-6)
- **Day of Month**: Date (1-31)
- **Month**: Month index (1-12)
- **Quarter**: Quarter (1-4)

#### Market Calendar
- **Month End**: Boolean flag
- **Quarter End**: Boolean flag
- **Year End**: Boolean flag (if applicable)

#### Cyclical Encoding
To handle cyclical nature of time:
- **Hour Sin/Cos**: sin(2π × hour/24), cos(2π × hour/24)
- **Day Sin/Cos**: sin(2π × day/7), cos(2π × day/7)
- **Month Sin/Cos**: sin(2π × month/12), cos(2π × month/12)

## Feature Engineering Pipeline

### 1. Data Validation
```python
# Check for valid OHLCV relationships
assert (data['high'] >= data['low']).all()
assert (data['high'] >= data['open']).all()
assert (data['high'] >= data['close']).all()
assert (data['volume'] >= 0).all()
```

### 2. Missing Value Handling
- **Forward Fill**: Use last valid value
- **Interpolation**: Linear interpolation
- **Drop**: Remove rows with missing values

### 3. Outlier Detection
- **Z-Score Method**: Remove values > 3 standard deviations
- **IQR Method**: Remove values outside 1.5 × IQR
- **Configurable Thresholds**: Adjustable sensitivity

### 4. Feature Scaling
```python
# Multiple scaling options
if scaling_method == "minmax":
    scaler = MinMaxScaler()
elif scaling_method == "standard":
    scaler = StandardScaler()
else:  # robust
    scaler = RobustScaler()
```

### 5. Feature Selection
```python
# Automatic feature selection
if feature_selection_method == "mutual_info":
    selector = SelectKBest(mutual_info_classif, k=max_features)
elif feature_selection_method == "chi2":
    selector = SelectKBest(chi2, k=max_features)
else:  # f_classif
    selector = SelectKBest(f_classif, k=max_features)
```

## Custom Indicator Creation

### Adding New Indicators
```python
def custom_indicator(self, close_prices, high_prices, low_prices):
    """Add your custom indicator here."""
    # Example: Custom momentum indicator
    momentum_5 = close_prices / np.roll(close_prices, 5) - 1
    momentum_20 = close_prices / np.roll(close_prices, 20) - 1
    
    return {
        'custom_momentum_5': momentum_5,
        'custom_momentum_20': momentum_20,
        'momentum_ratio': momentum_5 / momentum_20
    }
```

### Indicator Combinations
```python
def indicator_combinations(self, indicators_df):
    """Create combinations of existing indicators."""
    combinations = {}
    
    # RSI and MACD combination
    combinations['rsi_macd_signal'] = (
        indicators_df['rsi_14'] > 50
    ) & (
        indicators_df['macd'] > indicators_df['macd_signal']
    )
    
    # Bollinger and Volume combination
    combinations['bb_volume_breakout'] = (
        indicators_df['bb_position'] > 0.8
    ) & (
        indicators_df['volume_ratio'] > 1.5
    )
    
    return combinations
```

## Performance Considerations

### Computational Efficiency
- **Vectorized Operations**: All indicators use NumPy/Pandas vectorization
- **Memory Management**: Efficient memory usage for large datasets
- **Parallel Processing**: Multi-core utilization where possible

### Caching Strategy
- **Indicator Caching**: Cache computed indicators for reuse
- **Incremental Updates**: Update only new data points
- **Memory Limits**: Configurable cache size limits

## Configuration Options

### Indicator Parameters
```python
# Customize indicator periods
config.data.sma_periods = [10, 20, 50, 100, 200]
config.data.ema_periods = [12, 26, 50]
config.data.rsi_period = 14
config.data.bollinger_period = 20
config.data.bollinger_std = 2.0
```

### Feature Selection
```python
# Feature selection configuration
config.data.feature_selection_method = "mutual_info"
config.data.max_features = 100
config.data.feature_importance_threshold = 0.01
```

### Data Processing
```python
# Data processing options
config.data.scaling_method = "robust"
config.data.handle_missing_data = "forward_fill"
config.data.outlier_threshold = 3.0
```

## Best Practices

### 1. Feature Selection
- Start with all features, then use selection algorithms
- Consider correlation analysis to remove redundant features
- Use domain knowledge to guide feature importance

### 2. Data Quality
- Always validate OHLCV relationships
- Handle missing data appropriately for your use case
- Monitor for data anomalies and outliers

### 3. Computational Efficiency
- Cache indicators when possible
- Use incremental updates for real-time systems
- Consider memory usage for large datasets

### 4. Model Performance
- Use feature importance to understand model behavior
- Regularly retrain with new data
- Monitor feature drift over time

## Troubleshooting

### Common Issues

#### Missing TA-Lib Installation
```bash
# Install TA-Lib system library first
sudo apt-get install libta-lib-dev  # Ubuntu
brew install ta-lib                 # macOS

# Then install Python wrapper
pip install TA-Lib
```

#### Memory Issues with Large Datasets
```python
# Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    processed_chunk = processor.process_data(chunk)
    # Save or use processed chunk
```

#### Performance Optimization
```python
# Use multiprocessing for multiple symbols
from multiprocessing import Pool

def process_symbol(symbol_data):
    return processor.process_data(symbol_data)

with Pool() as pool:
    results = pool.map(process_symbol, symbol_datasets)
```

## Future Enhancements

### Planned Features
1. **Alternative Data Integration**: News sentiment, social media indicators
2. **Cross-Asset Features**: Correlations between different instruments
3. **Regime-Specific Indicators**: Features that adapt to market conditions
4. **Machine Learning Features**: Auto-generated features using ML

### Research Areas
- **Feature Interaction**: Automatic discovery of feature combinations
- **Dynamic Features**: Features that adapt based on market volatility
- **Causal Features**: Indicators based on causal relationships
- **Multi-Timeframe Fusion**: Intelligent combination of different timeframes