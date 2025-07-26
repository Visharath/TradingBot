# Unified CNN-LSTM Trading Bot

A comprehensive, production-ready algorithmic trading bot that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for intelligent market prediction and automated trading.

## 🚀 Features

### Machine Learning Architecture
- **Hybrid CNN-LSTM Model**: Multi-scale convolutions combined with bidirectional LSTM and multi-head attention
- **150+ Technical Indicators**: Comprehensive feature engineering including trend, momentum, volatility, and volume indicators
- **Multi-Timeframe Analysis**: Supports 1min, 5min, and 15min timeframes
- **Advanced Feature Engineering**: Statistical features, pattern recognition, and market regime detection

### Trading Capabilities
- **Real-Time Trading**: Integration with Interactive Brokers for live market data and trade execution
- **Intelligent Decision Making**: Buy/Hold/Sell classification with confidence scoring
- **Risk Management**: Position sizing, stop-loss, take-profit, and maximum drawdown controls
- **Portfolio Management**: Multi-symbol trading with correlation analysis and diversification

### Production Features
- **Comprehensive Logging**: Structured logging with rotation and multiple levels
- **Performance Monitoring**: Real-time metrics including Sharpe ratio, maximum drawdown, and win rate
- **Error Handling**: Robust exception handling with retry mechanisms
- **Configuration Management**: Flexible configuration system with environment variable support

## 📊 Model Architecture

```
Input Layer (Sequence + Static)
         |
    ┌────────────┬────────────────┬──────────────┐
    │ CNN Branch │  LSTM Branch   │Static Branch │
    │            │                │              │
    │ Conv1D(3,5,7) │ Bi-LSTM(128) │ Dense(32,16) │
    │ MaxPool1D  │ Attention(8)   │ BatchNorm    │
    │ BatchNorm  │ LayerNorm      │ Dropout      │
    └────────────┴────────────────┴──────────────┘
                     │
              Fusion Layer (256, 128, 64)
                     │
             Classification Output (3 classes)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Interactive Brokers TWS or IB Gateway
- TA-Lib (Technical Analysis Library)

### Quick Install
```bash
# Clone the repository
git clone https://github.com/Visharath/TradingBot.git
cd TradingBot

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (if not already installed)
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib

# On Windows:
# Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

### Development Install
```bash
pip install -e .[dev,notebooks]
pre-commit install
```

## 🚀 Quick Start

### Basic Usage

```python
from src.unified_trading_bot import UnifiedTradingBot
from src.config import config

# Initialize the trading bot
bot = UnifiedTradingBot()

# Load a pre-trained model (optional)
bot.load_model("models/my_trained_model.h5")

# Start trading (this will run continuously)
bot.start_trading()
```

### Training a New Model

```python
from src.data_processor import DataProcessor
from src.model_builder import UnifiedModelBuilder
import yfinance as yf

# Get training data
data = yf.download("AAPL", period="2y", interval="1h")

# Process data
processor = DataProcessor()
features, targets, feature_names = processor.process_data(data)

# Build and train model
model_builder = UnifiedModelBuilder()
model = model_builder.build_model(
    input_shape=(60, len(feature_names)),
    static_features_dim=10
)

# Train the model
history = model_builder.train_model(
    X_train=(sequence_data, static_data),
    y_train=targets
)
```

### Configuration

```python
from src.config import config

# Modify trading parameters
config.trading.max_position_size = 0.05  # 5% max position size
config.trading.confidence_threshold = 0.75  # Higher confidence threshold
config.data.symbols = ["AAPL", "GOOGL", "MSFT"]  # Trade these symbols

# Modify model parameters
config.model.sequence_length = 120  # Look back 120 periods
config.model.cnn_filters = [64, 128, 256, 512]  # More CNN filters
```

## 📊 Technical Indicators

The bot calculates 150+ technical indicators across multiple categories:

### Trend Indicators (25+)
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA
- MACD family: MACD, MACD Signal, MACD Histogram
- Directional Movement: ADX, ADXR, +DI, -DI, DX
- Parabolic SAR, Ichimoku Cloud components

### Momentum Indicators (30+)
- Oscillators: RSI, Stochastic, Williams %R, CCI
- Rate of Change: ROC, ROCP, ROCR, Momentum
- Money Flow Index, Aroon, Balance of Power
- Ultimate Oscillator, Trix, PPO

### Volatility Indicators (20+)
- Bollinger Bands, Keltner Channels, Donchian Channels
- Average True Range (ATR), Normalized ATR
- Historical Volatility (multiple periods)
- True Range calculations

### Volume Indicators (15+)
- On Balance Volume (OBV), Accumulation/Distribution
- Volume-weighted Average Price (VWAP)
- Chaikin Money Flow, Volume Price Trend
- Ease of Movement

### Pattern Recognition (50+)
- All major candlestick patterns
- Doji variations, Hammer patterns
- Engulfing patterns, Star patterns
- And many more...

### Statistical Features (30+)
- Rolling statistics: mean, std, skewness, kurtosis
- Quantiles and z-scores
- Autocorrelation analysis
- Returns-based features

## 💼 Risk Management

The bot includes comprehensive risk management features:

- **Position Sizing**: Kelly Criterion, Fixed Fractional, Volatility-based
- **Stop Loss**: ATR-based, Percentage-based, Trailing stops
- **Risk Limits**: Maximum drawdown, position concentration limits
- **Portfolio Metrics**: Real-time Sharpe ratio, Sortino ratio, VaR calculation

## 📈 Performance Monitoring

- **Real-time Metrics**: Live calculation of performance metrics
- **Trade Analytics**: Win rate, profit factor, average win/loss
- **Risk Metrics**: Maximum drawdown, volatility, beta calculation
- **Visualization**: Performance dashboards and equity curves

## 🔧 Configuration Options

### Model Configuration
```python
# CNN Branch
config.model.cnn_filters = [64, 128, 256]
config.model.cnn_kernel_sizes = [3, 5, 7]
config.model.cnn_dropout = 0.2

# LSTM Branch  
config.model.lstm_units = 128
config.model.bidirectional = True
config.model.attention_heads = 8

# Training
config.model.batch_size = 32
config.model.learning_rate = 0.001
config.model.epochs = 100
```

### Trading Configuration
```python
# Risk Management
config.trading.max_position_size = 0.1
config.trading.max_portfolio_risk = 0.02
config.trading.stop_loss_atr_multiplier = 2.0

# Execution
config.trading.confidence_threshold = 0.7
config.trading.min_hold_period = 5
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Architecture Overview](docs/architecture.md) 
- [Feature Engineering](docs/features.md)
- [API Reference](docs/api_reference.md)

## 🔮 Examples

Check out the `examples/` directory for:
- Basic usage examples
- Advanced configuration
- Backtesting implementations
- Custom strategy development

## 📊 Jupyter Notebooks

Explore the `notebooks/` directory for:
- Data exploration and analysis
- Feature engineering walkthrough
- Model training tutorials
- Performance analysis

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves substantial risk and may not be suitable for all investors. Past performance is not indicative of future results. The authors are not responsible for any financial losses incurred through the use of this software.

## 🙏 Acknowledgments

- [TA-Lib](https://github.com/mrjbq7/ta-lib) for technical analysis functions
- [IB-Insync](https://github.com/erdewit/ib_insync) for Interactive Brokers integration
- [TensorFlow](https://tensorflow.org) for machine learning framework
- [Pandas](https://pandas.pydata.org) for data manipulation

## 📞 Support

- Create an [Issue](https://github.com/Visharath/TradingBot/issues) for bug reports
- Start a [Discussion](https://github.com/Visharath/TradingBot/discussions) for questions
- Check the [Wiki](https://github.com/Visharath/TradingBot/wiki) for additional documentation

---

**Star this repository if you find it helpful! ⭐**