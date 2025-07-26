"""
Unified CNN-LSTM Trading Bot Package.

A comprehensive trading bot implementation featuring:
- CNN-LSTM hybrid neural network architecture
- 150+ technical indicators
- Real-time trading with Interactive Brokers
- Advanced risk management
- Performance monitoring and backtesting
"""

__version__ = "1.0.0"
__author__ = "Visharath"
__email__ = "your.email@example.com"
__description__ = "Unified CNN-LSTM Trading Bot for Algorithmic Trading"

# Core imports
from .config import config, Config
from .utils import logger, setup_logging
from .data_processor import DataProcessor
from .model_builder import UnifiedModelBuilder
from .unified_trading_bot import UnifiedTradingBot

# Main classes for easy import
__all__ = [
    "config",
    "Config", 
    "logger",
    "setup_logging",
    "DataProcessor",
    "UnifiedModelBuilder", 
    "UnifiedTradingBot",
    "__version__"
]

# Package metadata
PACKAGE_INFO = {
    "name": "unified_trading_bot",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "license": "MIT",
    "python_requires": ">=3.8",
    "keywords": [
        "trading", "bot", "machine-learning", "cnn", "lstm", 
        "technical-analysis", "algorithmic-trading", "finance"
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
}