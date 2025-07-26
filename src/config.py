"""
Configuration management for the Unified CNN-LSTM Trading Bot.

This module contains all configuration parameters for the trading bot,
including model hyperparameters, trading parameters, API configurations,
and system settings.
"""

import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    
    # Input parameters
    sequence_length: int = 60  # Number of time steps to look back
    n_features: int = 150  # Number of input features
    prediction_horizon: int = 1  # Steps ahead to predict
    
    # CNN Branch configuration
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_activation: str = "relu"
    cnn_dropout: float = 0.2
    
    # LSTM Branch configuration
    lstm_units: int = 128
    lstm_dropout: float = 0.3
    lstm_recurrent_dropout: float = 0.2
    bidirectional: bool = True
    
    # Attention mechanism
    attention_heads: int = 8
    attention_key_dim: int = 64
    
    # Static branch configuration
    static_dense_units: List[int] = field(default_factory=lambda: [32, 16])
    static_dropout: float = 0.2
    
    # Fusion layer configuration
    fusion_dense_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    fusion_dropout: float = 0.3
    
    # Output configuration
    num_classes: int = 3  # Buy (0), Hold (1), Sell (2)
    output_activation: str = "softmax"
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 15
    validation_split: float = 0.2
    
    # Optimization
    optimizer: str = "adam"
    loss_function: str = "sparse_categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])


@dataclass
class DataConfig:
    """Data processing and feature engineering configuration."""
    
    # Data sources
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    timeframes: List[str] = field(default_factory=lambda: ["1min", "5min", "15min"])
    data_provider: str = "yfinance"  # "yfinance", "alpha_vantage", "ib"
    
    # Feature engineering
    enable_technical_indicators: bool = True
    enable_statistical_features: bool = True
    enable_pattern_recognition: bool = True
    
    # Technical indicator periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50, 100])
    rsi_period: int = 14
    macd_periods: tuple = (12, 26, 9)
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Data preprocessing
    scaling_method: str = "robust"  # "minmax", "standard", "robust"
    handle_missing_data: str = "forward_fill"  # "forward_fill", "interpolate", "drop"
    outlier_threshold: float = 3.0  # Z-score threshold for outlier detection
    
    # Feature selection
    feature_selection_method: str = "mutual_info"  # "mutual_info", "chi2", "f_classif"
    max_features: Optional[int] = None  # None for all features
    feature_importance_threshold: float = 0.01


@dataclass
class TradingConfig:
    """Trading strategy and risk management configuration."""
    
    # Account and broker settings
    broker: str = "ib"  # Interactive Brokers
    account_id: str = "DU123456"  # Demo account by default
    base_currency: str = "USD"
    
    # Position sizing
    position_sizing_method: str = "kelly"  # "kelly", "fixed_fractional", "volatility_based"
    max_position_size: float = 0.1  # Maximum 10% of portfolio per position
    min_position_size: float = 0.01  # Minimum 1% of portfolio per position
    kelly_fraction: float = 0.25  # Conservative Kelly fraction
    
    # Risk management
    max_portfolio_risk: float = 0.02  # Maximum 2% portfolio risk per trade
    stop_loss_atr_multiplier: float = 2.0  # Stop loss at 2x ATR
    take_profit_ratio: float = 2.0  # Risk:reward ratio
    max_drawdown_threshold: float = 0.15  # Maximum 15% drawdown
    
    # Trade execution
    confidence_threshold: float = 0.7  # Minimum confidence for trade execution
    min_hold_period: int = 5  # Minimum bars to hold position
    max_hold_period: int = 100  # Maximum bars to hold position
    
    # Market regime detection
    enable_regime_filter: bool = True
    volatility_threshold: float = 0.02  # High volatility threshold
    trend_strength_threshold: float = 0.6  # Minimum trend strength


@dataclass
class APIConfig:
    """API and external service configuration."""
    
    # Interactive Brokers
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # TWS paper trading port
    ib_client_id: int = 1
    ib_timeout: int = 30
    
    # Data providers
    alpha_vantage_api_key: Optional[str] = None
    quandl_api_key: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_second: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = str(LOGS_DIR / "trading_bot.log")
    max_log_size_mb: int = 100
    backup_count: int = 5
    
    # Specific logger levels
    tensorflow_log_level: str = "WARNING"
    matplotlib_log_level: str = "WARNING"


class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.trading = TradingConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        
        # Load environment variables
        self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # API keys
        self.api.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.api.quandl_api_key = os.getenv("QUANDL_API_KEY")
        
        # Trading account
        self.trading.account_id = os.getenv("IB_ACCOUNT_ID", self.trading.account_id)
        
        # Interactive Brokers connection
        self.api.ib_host = os.getenv("IB_HOST", self.api.ib_host)
        self.api.ib_port = int(os.getenv("IB_PORT", str(self.api.ib_port)))
        self.api.ib_client_id = int(os.getenv("IB_CLIENT_ID", str(self.api.ib_client_id)))
        
        # Logging
        self.logging.log_level = os.getenv("LOG_LEVEL", self.logging.log_level)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "trading": self.trading.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__,
        }
    
    def update_from_dict(self, config_dict: Dict) -> None:
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)


# Global configuration instance
config = Config()

# Export commonly used paths
__all__ = [
    "config",
    "ModelConfig",
    "DataConfig", 
    "TradingConfig",
    "APIConfig",
    "LoggingConfig",
    "Config",
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
]