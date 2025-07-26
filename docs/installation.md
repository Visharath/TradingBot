# Installation Guide

## System Requirements

### Operating System
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.14+)
- Windows 10/11 (with WSL recommended)

### Python Requirements
- Python 3.8 or higher
- pip 21.0 or higher
- virtualenv or conda (recommended)

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 10GB storage
- **Recommended**: 16GB RAM, 8 CPU cores, 50GB SSD storage
- **GPU**: Optional but recommended for faster training (CUDA 11.2+)

## Installation Methods

### Method 1: pip Installation (Recommended)

```bash
# Create virtual environment
python -m venv trading-bot-env
source trading-bot-env/bin/activate  # On Windows: trading-bot-env\Scripts\activate

# Clone repository
git clone https://github.com/Visharath/TradingBot.git
cd TradingBot

# Install package
pip install -e .
```

### Method 2: conda Installation

```bash
# Create conda environment
conda create -n trading-bot python=3.9
conda activate trading-bot

# Clone repository
git clone https://github.com/Visharath/TradingBot.git
cd TradingBot

# Install dependencies
conda install --file requirements.txt
pip install -e .
```

### Method 3: Docker Installation

```bash
# Build Docker image
docker build -t trading-bot .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  trading-bot
```

## Dependency Installation

### Core Dependencies

The main dependencies will be installed automatically, but you can install them manually:

```bash
# Core ML and data processing
pip install numpy pandas scikit-learn tensorflow

# Financial data and technical analysis
pip install yfinance ta TA-Lib pandas-ta

# Visualization and monitoring
pip install matplotlib seaborn plotly

# Configuration and utilities
pip install python-dotenv pyyaml loguru click tqdm
```

### Optional Dependencies

```bash
# Interactive Brokers API
pip install ibapi

# Advanced monitoring
pip install wandb mlflow

# Development tools
pip install pytest black isort flake8 mypy pre-commit

# Documentation
pip install sphinx sphinx-rtd-theme

# Jupyter notebooks
pip install jupyter ipykernel notebook
```

### TA-Lib Installation

TA-Lib requires special installation on some systems:

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential wget

# Download and install TA-Lib C library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

#### macOS
```bash
# Using Homebrew
brew install ta-lib
pip install TA-Lib

# Or using MacPorts
sudo port install ta-lib
pip install TA-Lib
```

#### Windows
```bash
# Download pre-compiled wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

# Install the downloaded wheel
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

## Interactive Brokers Setup

### 1. Install TWS or IB Gateway

Download and install from [Interactive Brokers](https://www.interactivebrokers.com/):
- **Trader Workstation (TWS)**: Full trading platform
- **IB Gateway**: Lightweight API interface (recommended for automated trading)

### 2. Configure API Settings

1. Open TWS/IB Gateway
2. Go to **File → Global Configuration → API → Settings**
3. Enable the following:
   - Enable ActiveX and Socket Clients
   - Allow connections from localhost
   - Master API client ID: 0
   - Read-Only API: No (for live trading)

### 3. Paper Trading Setup

1. Log into your paper trading account
2. Use port **7497** for paper trading
3. Use port **7496** for live trading

### 4. Environment Configuration

Create a `.env` file:

```bash
# Interactive Brokers Configuration
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
IB_ACCOUNT_ID=your_paper_account_id
PAPER_TRADING=true

# Logging
LOG_LEVEL=INFO
DEBUG=false

# Monitoring (optional)
WANDB_PROJECT=trading-bot
ENVIRONMENT=development
```

## Verification

### 1. Test Installation

```python
# Test core imports
from src.unified_trading_bot import UnifiedTradingBot
from src.data_processor import DataProcessor
from src.model_builder import ModelBuilder
from src.config import Config

print("All imports successful!")
```

### 2. Test Data Fetching

```python
from src.data_processor import DataProcessor
from src.config import Config

config = Config()
processor = DataProcessor(config)

# Test data fetching
data = processor.fetch_data('AAPL', period='1mo')
if data is not None:
    print(f"Successfully fetched {len(data)} rows of data")
else:
    print("Data fetching failed")
```

### 3. Test Technical Indicators

```python
# Test indicator calculation
if data is not None:
    indicators = processor.calculate_technical_indicators(data)
    print(f"Calculated {len(indicators.columns)} features")
    print("Sample indicators:", indicators.columns[:10].tolist())
```

### 4. Test Model Building

```python
from src.model_builder import ModelBuilder

# Test model creation
model_builder = ModelBuilder(config)
model = model_builder.build_model((60, 100), n_classes=3, static_features=20)
print("Model built successfully!")
print(f"Model parameters: {model.count_params():,}")
```

### 5. Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Common Issues and Solutions

### Issue: TA-Lib installation fails

**Solution**: Install system dependencies first:

```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential python3-dev

# CentOS/RHEL
sudo yum install -y gcc gcc-c++ python3-devel

# macOS
xcode-select --install
```

### Issue: TensorFlow GPU not detected

**Solution**: Install CUDA and cuDNN:

```bash
# Check CUDA installation
nvidia-smi

# Install TensorFlow GPU (if CUDA available)
pip install tensorflow-gpu

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: Interactive Brokers connection fails

**Solutions**:
1. Check TWS/IB Gateway is running
2. Verify API settings are enabled
3. Check port configuration (7497 for paper, 7496 for live)
4. Ensure firewall allows connections
5. Restart TWS/IB Gateway

### Issue: Memory errors during training

**Solutions**:
1. Reduce batch size in config
2. Use smaller sequence length
3. Reduce number of symbols trained simultaneously
4. Use gradient checkpointing
5. Monitor memory usage

### Issue: Missing market data

**Solutions**:
1. Check internet connection
2. Verify symbol spelling
3. Try different data source
4. Check market hours
5. Clear data cache

## Performance Optimization

### 1. GPU Acceleration

```python
# Enable mixed precision
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 2. Memory Management

```python
# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 3. Parallel Processing

```python
# Configure parallel data processing
config.system.n_jobs = -1  # Use all CPU cores
config.system.parallel_backend = 'threading'
```

### 4. Caching

```python
# Enable data caching
config.data.enable_caching = True
config.data.cache_dir = 'data/cache'
```

## Next Steps

1. **Configure your environment**: Update the `.env` file with your settings
2. **Test data connection**: Verify you can fetch market data
3. **Run a simple backtest**: Test the system with historical data
4. **Paper trading**: Start with paper trading before live trading
5. **Read the documentation**: Study the [Architecture Guide](architecture.md) and [Features Guide](features.md)

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](#common-issues-and-solutions)
2. Search [GitHub Issues](https://github.com/Visharath/TradingBot/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Complete error message
   - Steps to reproduce

## Security Considerations

1. **Never commit credentials**: Use environment variables for sensitive data
2. **Use paper trading**: Always start with paper trading
3. **Secure your API**: Limit API access to localhost only
4. **Regular updates**: Keep dependencies updated for security patches
5. **Monitor access**: Regularly check API access logs