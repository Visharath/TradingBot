"""
Unified CNN-LSTM Trading Bot Package

A comprehensive trading bot system combining Convolutional Neural Networks (CNN)
and Long Short-Term Memory (LSTM) networks for advanced pattern recognition
and sequence modeling in financial markets.

Features:
- CNN-LSTM hybrid architecture
- 150+ technical indicators
- Interactive Brokers integration
- Multi-timeframe analysis
- Advanced risk management
- Real-time trading capabilities

Author: Visharath
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Visharath"
__email__ = "visharath@example.com"
__license__ = "MIT"

from src.config import Config
from src.unified_trading_bot import UnifiedTradingBot
from src.data_processor import DataProcessor
from src.model_builder import ModelBuilder
from src.utils import setup_logging, validate_config

__all__ = [
    "Config",
    "UnifiedTradingBot", 
    "DataProcessor",
    "ModelBuilder",
    "setup_logging",
    "validate_config",
]