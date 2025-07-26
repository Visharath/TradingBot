"""
Utility functions for the Unified CNN-LSTM Trading Bot.

This module provides common utility functions including logging setup,
performance metrics calculation, data visualization, and error handling.
"""

import logging
import logging.handlers
import functools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta

from .config import config


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration for the trading bot.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(config.logging.log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(getattr(logging, config.logging.log_level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.logging.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.logging.log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        config.logging.log_file,
        maxBytes=config.logging.max_log_size_mb * 1024 * 1024,
        backupCount=config.logging.backup_count
    )
    file_handler.setLevel(getattr(logging, config.logging.log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger("tensorflow").setLevel(
        getattr(logging, config.logging.tensorflow_log_level)
    )
    logging.getLogger("matplotlib").setLevel(
        getattr(logging, config.logging.matplotlib_log_level)
    )
    
    return logger


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying function calls on exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger = logging.getLogger("trading_bot")
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
                        raise last_exception
            
            return None  # Should never reach here
        return wrapper
    return decorator


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
    
    Returns:
        Decorated function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger("trading_bot")
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper


def calculate_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for trading strategy.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns (optional)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary containing performance metrics
    """
    metrics = {}
    
    # Basic return metrics
    metrics["total_return"] = (1 + returns).prod() - 1
    metrics["annualized_return"] = (1 + returns.mean()) ** 252 - 1
    metrics["volatility"] = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate / 252
    metrics["sharpe_ratio"] = excess_returns.mean() / returns.std() * np.sqrt(252)
    
    # Downside risk metrics
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        metrics["sortino_ratio"] = (
            excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        )
        metrics["max_drawdown"] = calculate_max_drawdown(returns)
    else:
        metrics["sortino_ratio"] = np.inf
        metrics["max_drawdown"] = 0.0
    
    # Win/loss metrics
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    
    metrics["win_rate"] = len(winning_trades) / len(returns) if len(returns) > 0 else 0
    metrics["avg_win"] = winning_trades.mean() if len(winning_trades) > 0 else 0
    metrics["avg_loss"] = losing_trades.mean() if len(losing_trades) > 0 else 0
    metrics["profit_factor"] = (
        abs(winning_trades.sum() / losing_trades.sum()) 
        if len(losing_trades) > 0 and losing_trades.sum() != 0 else np.inf
    )
    
    # Benchmark comparison
    if benchmark_returns is not None:
        correlation = returns.corr(benchmark_returns)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        beta = correlation * (metrics["volatility"] / benchmark_volatility)
        benchmark_return = (1 + benchmark_returns.mean()) ** 252 - 1
        
        metrics["beta"] = beta
        metrics["alpha"] = metrics["annualized_return"] - (
            risk_free_rate + beta * (benchmark_return - risk_free_rate)
        )
        metrics["information_ratio"] = (
            (returns - benchmark_returns).mean() / 
            (returns - benchmark_returns).std() * np.sqrt(252)
        )
    
    return metrics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns series.
    
    Args:
        returns: Series of returns
    
    Returns:
        Maximum drawdown as a float
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def plot_performance_dashboard(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive performance dashboard.
    
    Args:
        returns: Portfolio returns series
        benchmark_returns: Benchmark returns series (optional)
        save_path: Path to save the plot (optional)
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Trading Performance Dashboard", fontsize=16)
    
    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                    label="Strategy", linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                        label="Benchmark", linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title("Cumulative Returns")
    axes[0, 0].set_ylabel("Cumulative Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.3, color='red')
    axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    axes[0, 1].set_title("Drawdown")
    axes[0, 1].set_ylabel("Drawdown")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Returns distribution
    axes[1, 0].hist(returns.values, bins=50, alpha=0.7, density=True)
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {returns.mean():.4f}')
    axes[1, 0].set_title("Returns Distribution")
    axes[1, 0].set_xlabel("Return")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling Sharpe ratio
    rolling_sharpe = (
        returns.rolling(window=252).mean() / returns.rolling(window=252).std() * 
        np.sqrt(252)
    )
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    axes[1, 1].set_title("Rolling Sharpe Ratio (252 days)")
    axes[1, 1].set_ylabel("Sharpe Ratio")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_model_artifacts(
    model: Any,
    metadata: Dict,
    model_path: str,
    metadata_path: str
) -> None:
    """
    Save model and associated metadata.
    
    Args:
        model: Trained model object
        metadata: Dictionary containing model metadata
        model_path: Path to save the model
        metadata_path: Path to save the metadata
    """
    # Save model
    if hasattr(model, 'save'):
        model.save(model_path)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata
    metadata["saved_at"] = datetime.now().isoformat()
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_model_artifacts(
    model_path: str,
    metadata_path: str
) -> Tuple[Any, Dict]:
    """
    Load model and associated metadata.
    
    Args:
        model_path: Path to the saved model
        metadata_path: Path to the saved metadata
    
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    try:
        # Try TensorFlow/Keras first
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    except:
        # Fall back to pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata


def validate_data_quality(
    data: pd.DataFrame,
    required_columns: List[str],
    max_missing_ratio: float = 0.05
) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        max_missing_ratio: Maximum allowed ratio of missing values
    
    Returns:
        Dictionary containing data quality metrics and issues
    """
    quality_report = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "missing_columns": [],
        "high_missing_columns": [],
        "duplicate_rows": 0,
        "data_types": {},
        "date_range": {},
        "quality_score": 0.0,
        "issues": []
    }
    
    # Check for missing required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        quality_report["missing_columns"] = list(missing_cols)
        quality_report["issues"].append(f"Missing required columns: {missing_cols}")
    
    # Check for high missing value ratios
    missing_ratios = data.isnull().sum() / len(data)
    high_missing = missing_ratios[missing_ratios > max_missing_ratio]
    if not high_missing.empty:
        quality_report["high_missing_columns"] = high_missing.to_dict()
        quality_report["issues"].append("High missing value ratios detected")
    
    # Check for duplicate rows
    duplicates = data.duplicated().sum()
    quality_report["duplicate_rows"] = duplicates
    if duplicates > 0:
        quality_report["issues"].append(f"Found {duplicates} duplicate rows")
    
    # Data types
    quality_report["data_types"] = data.dtypes.astype(str).to_dict()
    
    # Date range (if datetime index)
    if isinstance(data.index, pd.DatetimeIndex):
        quality_report["date_range"] = {
            "start": data.index.min().isoformat(),
            "end": data.index.max().isoformat(),
            "frequency": pd.infer_freq(data.index)
        }
    
    # Calculate quality score
    issues_count = len(quality_report["issues"])
    quality_score = max(0, 1 - (issues_count * 0.2 + duplicates / len(data) * 0.3))
    quality_report["quality_score"] = quality_score
    
    return quality_report


# Initialize logger
logger = setup_logging()

__all__ = [
    "setup_logging",
    "retry_on_exception",
    "timing_decorator",
    "calculate_performance_metrics",
    "calculate_max_drawdown",
    "plot_performance_dashboard",
    "save_model_artifacts",
    "load_model_artifacts",
    "validate_data_quality",
    "logger"
]