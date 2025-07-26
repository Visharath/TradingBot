# Installation Guide

This guide provides detailed instructions for installing and setting up the Unified CNN-LSTM Trading Bot.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB free space
- **Internet**: Stable broadband connection for real-time trading

### Recommended Requirements
- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7+)
- **RAM**: 32 GB for large-scale backtesting
- **GPU**: NVIDIA GPU with CUDA support (optional, for model training acceleration)
- **Storage**: SSD with 20+ GB free space

## Prerequisites Installation

### 1. Python Environment

#### Using Anaconda (Recommended)
```bash
# Download and install Anaconda from https://www.anaconda.com/
# Create a new environment
conda create -n trading_bot python=3.9
conda activate trading_bot
```

#### Using pyenv (Alternative)
```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.9
pyenv install 3.9.18
pyenv local 3.9.18
```

### 2. TA-Lib Installation

TA-Lib is a crucial dependency for technical analysis. Installation varies by platform:

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential wget

# Download and install TA-Lib
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

# Install Python wrapper
pip install TA-Lib
```

#### Windows
```bash
# Download pre-compiled wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

# Install appropriate wheel for your Python version
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

### 3. Interactive Brokers Setup

#### Download and Install TWS or IB Gateway
1. Visit [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=16040)
2. Download TWS (Trader Workstation) or IB Gateway
3. Install and create a paper trading account for testing

#### Enable API Access
1. Open TWS/IB Gateway
2. Go to **File** → **Global Configuration** → **API** → **Settings**
3. Enable **"Enable ActiveX and Socket Clients"**
4. Set **Socket Port** to `7497` (paper trading) or `7496` (live trading)
5. Add `127.0.0.1` to **Trusted IPs**

## Bot Installation

### Method 1: Clone from GitHub
```bash
# Clone the repository
git clone https://github.com/Visharath/TradingBot.git
cd TradingBot

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Method 2: Install from PyPI (when available)
```bash
pip install unified-trading-bot
```

### Method 3: Install with Optional Features
```bash
# Install with development tools
pip install -e .[dev]

# Install with Jupyter notebook support
pip install -e .[notebooks]

# Install with GPU support
pip install -e .[gpu]

# Install all features
pip install -e .[dev,notebooks,gpu]
```

## Configuration Setup

### 1. Environment Variables
Create a `.env` file in the project root:

```bash
# Interactive Brokers Configuration
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
IB_ACCOUNT_ID=DU123456

# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_api_key_here
QUANDL_API_KEY=your_api_key_here

# Logging
LOG_LEVEL=INFO
```

### 2. Trading Configuration
Edit `src/config.py` or create a custom configuration:

```python
from src.config import config

# Trading parameters
config.trading.max_position_size = 0.05  # 5% max position
config.trading.confidence_threshold = 0.75
config.data.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

# Model parameters
config.model.sequence_length = 60
config.model.batch_size = 32
```

## Verification

### 1. Test Python Environment
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import talib; print('TA-Lib installed successfully')"
python -c "import ib_insync; print('IB-Insync installed successfully')"
```

### 2. Test Bot Installation
```bash
# Test imports
python -c "from src.unified_trading_bot import UnifiedTradingBot; print('Bot imported successfully')"

# Run basic functionality test
python examples/basic_usage.py
```

### 3. Test Broker Connection
```bash
# Start TWS/IB Gateway first, then run:
python -c "
import asyncio
from src.unified_trading_bot import UnifiedTradingBot

async def test_connection():
    bot = UnifiedTradingBot()
    connected = await bot.connect_to_broker()
    print(f'Connection successful: {connected}')
    bot.disconnect_from_broker()

asyncio.run(test_connection())
"
```

## Common Installation Issues

### Issue 1: TA-Lib Installation Fails
**Solution for Ubuntu:**
```bash
sudo apt-get install python3-dev
export TA_INCLUDE_PATH=/usr/include
export TA_LIBRARY_PATH=/usr/lib
pip install TA-Lib
```

**Solution for Windows:**
- Use pre-compiled wheels from [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Ensure you download the correct wheel for your Python version and architecture

### Issue 2: TensorFlow GPU Not Detected
```bash
# Install CUDA and cuDNN
# For Ubuntu:
sudo apt install nvidia-cuda-toolkit
pip install tensorflow-gpu

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Issue 3: Interactive Brokers Connection Failed
- Ensure TWS/IB Gateway is running
- Check that API is enabled in TWS settings
- Verify port number (7497 for paper, 7496 for live)
- Add 127.0.0.1 to trusted IPs in TWS

### Issue 4: Import Errors
```bash
# Ensure you're in the correct virtual environment
which python
pip list | grep -E "(tensorflow|pandas|numpy)"

# Reinstall problematic packages
pip uninstall problematic_package
pip install problematic_package
```

## Development Setup

### 1. Install Development Tools
```bash
pip install -e .[dev]
```

### 2. Set Up Pre-commit Hooks
```bash
pre-commit install
```

### 3. Set Up IDE

#### VS Code
Install recommended extensions:
- Python
- Pylance
- Black Formatter
- GitLens

#### PyCharm
Configure interpreter to use your virtual environment.

### 4. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
```

## Production Deployment

### 1. Security Considerations
- Use environment variables for sensitive data
- Set up proper logging and monitoring
- Use HTTPS for all external connections
- Regularly update dependencies

### 2. Server Setup
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv nginx supervisor

# Create application user
sudo useradd -m -s /bin/bash trading_bot
sudo su - trading_bot

# Set up application
git clone https://github.com/Visharath/TradingBot.git
cd TradingBot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Process Management
Create supervisor configuration (`/etc/supervisor/conf.d/trading_bot.conf`):

```ini
[program:trading_bot]
command=/home/trading_bot/TradingBot/venv/bin/python -m src.unified_trading_bot
directory=/home/trading_bot/TradingBot
user=trading_bot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/trading_bot.log
```

## Next Steps

1. **Read the Architecture Guide**: Understand how the bot works internally
2. **Explore Examples**: Check out the `examples/` directory
3. **Run Notebooks**: Go through the Jupyter notebooks for hands-on learning
4. **Start with Paper Trading**: Always test with paper trading before using real money
5. **Customize Configuration**: Adapt the bot to your trading strategy

## Support

If you encounter issues during installation:

1. Check the [troubleshooting section](#common-installation-issues)
2. Search existing [GitHub issues](https://github.com/Visharath/TradingBot/issues)
3. Create a new issue with detailed error messages and system information
4. Join our [community discussions](https://github.com/Visharath/TradingBot/discussions)