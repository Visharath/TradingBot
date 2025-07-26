"""
Utility functions for the Unified CNN-LSTM Trading Bot.

This module provides common utility functions for logging, validation,
data handling, and other shared functionality across the trading bot system.
"""

import os
import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from loguru import logger

from src.config import Config


def setup_logging(config: Config) -> None:
    """
    Set up logging configuration using loguru.
    
    Args:
        config: Configuration object containing logging settings
    """
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=config.system.log_level,
        format=config.system.log_format,
        colorize=True,
    )
    
    # Add file handler
    log_file = config.system.logs_dir / "trading_bot.log"
    logger.add(
        log_file,
        level=config.system.log_level,
        format=config.system.log_format,
        rotation=config.system.log_rotation,
        retention=config.system.log_retention,
        compression="zip",
    )
    
    logger.info(f"Logging initialized - Level: {config.system.log_level}")


def validate_config(config: Config) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Validate model configuration
        assert config.model.sequence_length > 0, "Sequence length must be positive"
        assert config.model.batch_size > 0, "Batch size must be positive"
        assert 0 < config.model.validation_split < 1, "Validation split must be between 0 and 1"
        assert 0 <= config.model.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
        
        # Validate data configuration
        assert len(config.data.symbols) > 0, "At least one symbol must be specified"
        assert config.data.lookback_days > 0, "Lookback days must be positive"
        assert config.data.min_data_points > 0, "Minimum data points must be positive"
        
        # Validate trading configuration
        assert config.trading.initial_capital > 0, "Initial capital must be positive"
        assert 0 < config.trading.max_position_size <= 1, "Max position size must be between 0 and 1"
        assert 0 < config.trading.max_drawdown <= 1, "Max drawdown must be between 0 and 1"
        assert config.trading.max_positions > 0, "Max positions must be positive"
        
        logger.info("Configuration validation successful")
        return True
        
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during configuration validation: {e}")
        return False


def save_model_metadata(
    model_path: Path,
    config: Config,
    performance_metrics: Dict[str, float],
    feature_names: List[str],
    training_time: float,
) -> None:
    """
    Save model metadata including configuration and performance metrics.
    
    Args:
        model_path: Path where the model is saved
        config: Configuration used for training
        performance_metrics: Dictionary of performance metrics
        feature_names: List of feature names used in training
        training_time: Time taken to train the model in seconds
    """
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "config": config.to_dict(),
        "performance_metrics": performance_metrics,
        "feature_names": feature_names,
        "training_time_seconds": training_time,
        "model_hash": calculate_file_hash(model_path),
    }
    
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model metadata saved to {metadata_path}")


def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load model metadata from JSON file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model metadata or None if not found
    """
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Model metadata loaded from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return None


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal string representation of the file hash
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""


def save_predictions(
    predictions: np.ndarray,
    timestamps: List[datetime],
    symbols: List[str],
    confidence_scores: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Save predictions to CSV file with timestamps and symbols.
    
    Args:
        predictions: Array of predictions
        timestamps: List of timestamps corresponding to predictions
        symbols: List of symbols corresponding to predictions
        confidence_scores: Optional confidence scores for predictions
        save_path: Optional path to save file, if None will use default naming
        
    Returns:
        Path to the saved file
    """
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("results") / f"predictions_{timestamp}.csv"
    
    # Create results directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    data = {
        "timestamp": timestamps,
        "symbol": symbols,
        "prediction": predictions,
    }
    
    if confidence_scores is not None:
        data["confidence"] = confidence_scores
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Predictions saved to {save_path}")
    return save_path


def load_pickle(file_path: Path) -> Any:
    """
    Load object from pickle file with error handling.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded object or None if error occurred
    """
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        return None


def save_pickle(obj: Any, file_path: Path) -> bool:
    """
    Save object to pickle file with error handling.
    
    Args:
        obj: Object to save
        file_path: Path to save the pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}")
        return False


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code (default: USD)
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_market_hours(timezone_name: str = "US/Eastern") -> Tuple[str, str]:
    """
    Get market opening and closing hours for a given timezone.
    
    Args:
        timezone_name: Timezone name (default: US/Eastern)
        
    Returns:
        Tuple of (market_open, market_close) in HH:MM format
    """
    # Default US market hours
    market_hours = {
        "US/Eastern": ("09:30", "16:00"),
        "US/Central": ("08:30", "15:00"),
        "US/Mountain": ("07:30", "14:00"),
        "US/Pacific": ("06:30", "13:00"),
        "Europe/London": ("08:00", "16:30"),
        "Asia/Tokyo": ("09:00", "15:00"),
    }
    
    return market_hours.get(timezone_name, ("09:30", "16:00"))


def is_market_open(
    current_time: Optional[datetime] = None,
    timezone_name: str = "US/Eastern"
) -> bool:
    """
    Check if the market is currently open.
    
    Args:
        current_time: Current time (default: now)
        timezone_name: Market timezone (default: US/Eastern)
        
    Returns:
        True if market is open, False otherwise
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if current_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    market_open, market_close = get_market_hours(timezone_name)
    
    current_time_str = current_time.strftime("%H:%M")
    
    return market_open <= current_time_str <= market_close


def chunked_processing(
    data: List[Any],
    chunk_size: int,
    process_func: callable,
    *args,
    **kwargs
) -> List[Any]:
    """
    Process data in chunks to manage memory usage.
    
    Args:
        data: List of data to process
        chunk_size: Size of each chunk
        process_func: Function to apply to each chunk
        *args: Additional arguments for process_func
        **kwargs: Additional keyword arguments for process_func
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_result = process_func(chunk, *args, **kwargs)
        results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
    
    return results


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information in MB
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent(),
    }


def print_system_info() -> None:
    """Print system information for debugging purposes."""
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    logger.info(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")