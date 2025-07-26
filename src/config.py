"""
Configuration settings for the Unified CNN-LSTM Trading Bot.

This module contains all configuration parameters for the trading bot,
including model architecture, data processing, trading, and system settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the CNN-LSTM model architecture."""
    
    # Model architecture
    sequence_length: int = 60
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    lstm_units: List[int] = field(default_factory=lambda: [100, 50])
    attention_heads: int = 8
    dropout_rate: float = 0.3
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 15
    validation_split: float = 0.2
    
    # Feature configuration
    n_features: int = 150  # Number of technical indicators
    price_features: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    
    # Model output
    n_classes: int = 3  # Buy, Hold, Sell
    class_names: List[str] = field(default_factory=lambda: ["sell", "hold", "buy"])


@dataclass
class DataConfig:
    """Configuration for data processing and feature engineering."""
    
    # Data sources
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "TSLA"])
    timeframes: List[str] = field(default_factory=lambda: ["1min", "5min", "15min"])
    
    # Historical data
    lookback_days: int = 252  # Approximately 1 year of trading days
    min_data_points: int = 1000
    
    # Feature engineering
    technical_indicators: Dict[str, Dict] = field(default_factory=lambda: {
        "trend": ["sma", "ema", "macd", "adx", "aroon", "cci", "dpo"],
        "momentum": ["rsi", "stoch", "williams", "roc", "tsi", "uo"],
        "volatility": ["bbands", "atr", "keltner", "donchian"],
        "volume": ["obv", "vpt", "mfi", "ad", "cmf", "fi", "eom", "vwap"],
        "others": ["ichimoku", "parabolic_sar", "trix"]
    })
    
    # Data processing
    scaling_method: str = "standard"  # "standard", "minmax", "robust"
    handle_missing: str = "forward_fill"  # "forward_fill", "interpolate", "drop"
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    
    # Target creation
    prediction_horizon: int = 5  # Number of periods ahead to predict
    threshold_pct: float = 0.02  # Threshold for buy/sell signals (2%)


@dataclass
class TradingConfig:
    """Configuration for trading and risk management."""
    
    # Account settings
    account_id: Optional[str] = None
    paper_trading: bool = True
    initial_capital: float = 100000.0
    
    # Position sizing
    max_position_size: float = 0.1  # 10% of portfolio per position
    position_sizing_method: str = "fixed_fractional"  # "fixed_fractional", "kelly", "volatility_based"
    
    # Risk management
    max_drawdown: float = 0.15  # Maximum allowed drawdown (15%)
    stop_loss_pct: float = 0.05  # Stop loss percentage (5%)
    take_profit_pct: float = 0.10  # Take profit percentage (10%)
    max_positions: int = 5  # Maximum number of open positions
    
    # Trading hours
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "US/Eastern"
    
    # Signal filtering
    min_confidence: float = 0.6  # Minimum prediction confidence
    lookback_correlation: int = 20  # Periods to check for signal correlation
    
    # Interactive Brokers settings
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # Paper trading port (7496 for live)
    ib_client_id: int = 1


@dataclass
class SystemConfig:
    """Configuration for system-wide settings."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    # Performance monitoring
    enable_wandb: bool = False
    enable_mlflow: bool = False
    wandb_project: str = "trading-bot"
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    
    # API settings
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    
    # Multi-processing
    n_jobs: int = -1  # Use all available cores
    parallel_backend: str = "threading"
    
    # Environment
    environment: str = "development"  # "development", "testing", "production"
    debug: bool = False


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self) -> None:
        """Post-initialization to create directories and load environment variables."""
        self._create_directories()
        self._load_environment_variables()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [
            self.system.data_dir,
            self.system.models_dir,
            self.system.logs_dir,
            self.system.results_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables if available."""
        # Trading configuration from environment
        if os.getenv("IB_ACCOUNT_ID"):
            self.trading.account_id = os.getenv("IB_ACCOUNT_ID")
        
        if os.getenv("IB_HOST"):
            self.trading.ib_host = os.getenv("IB_HOST")
        
        if os.getenv("IB_PORT"):
            self.trading.ib_port = int(os.getenv("IB_PORT"))
        
        if os.getenv("PAPER_TRADING"):
            self.trading.paper_trading = os.getenv("PAPER_TRADING").lower() == "true"
        
        # System configuration from environment
        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")
        
        if os.getenv("WANDB_PROJECT"):
            self.system.wandb_project = os.getenv("WANDB_PROJECT")
        
        if os.getenv("ENVIRONMENT"):
            self.system.environment = os.getenv("ENVIRONMENT")
        
        if os.getenv("DEBUG"):
            self.system.debug = os.getenv("DEBUG").lower() == "true"
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "trading": self.trading.__dict__,
            "system": {k: str(v) if isinstance(v, Path) else v for k, v in self.system.__dict__.items()},
        }
    
    def update_from_dict(self, config_dict: Dict) -> None:
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)


# Default configuration instance
config = Config()