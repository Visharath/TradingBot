"""
Basic Usage Example for the Unified CNN-LSTM Trading Bot.

This example demonstrates the fundamental usage patterns for training,
predicting, and basic trading with the system.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.unified_trading_bot import UnifiedTradingBot
from src.config import Config
from src.utils import setup_logging, format_currency


def basic_training_example():
    """Demonstrate basic model training."""
    print("=== Basic Training Example ===")
    
    # Initialize the trading bot
    bot = UnifiedTradingBot()
    
    # Configure symbols to train on
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print(f"Training model on symbols: {symbols}")
    
    # Train the model
    metrics = bot.train_model(symbols=symbols, save_model=True)
    
    print("Training completed!")
    print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
    
    return bot


def basic_prediction_example(bot=None):
    """Demonstrate basic signal generation."""
    print("\n=== Basic Prediction Example ===")
    
    if bot is None:
        bot = UnifiedTradingBot()
        # For this example, assume we have a pre-trained model
        # In practice, you would either train first or load a saved model
        print("Note: Using a pre-configured bot (model should be trained first)")
    
    # Generate predictions for specific symbols
    symbols = ['AAPL', 'GOOGL']
    
    print(f"Generating predictions for: {symbols}")
    
    try:
        predictions = bot.predict_signals(symbols)
        
        for symbol, pred in predictions.items():
            signal_name = pred['signal_name']
            confidence = pred['confidence']
            price = pred['current_price']
            
            print(f"{symbol}:")
            print(f"  Signal: {signal_name}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Current Price: ${price:.2f}")
            print(f"  Probabilities: {pred['probabilities']}")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure to train the model first or load a pre-trained model")
    
    return bot


def basic_backtesting_example(bot=None):
    """Demonstrate basic backtesting functionality."""
    print("\n=== Basic Backtesting Example ===")
    
    if bot is None:
        bot = UnifiedTradingBot()
    
    # Define backtest parameters
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    symbols = ['AAPL', 'MSFT']
    
    print(f"Running backtest from {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    
    try:
        results = bot.backtest(start_date, end_date, symbols)
        
        print("Backtest Results:")
        print(f"  Initial Capital: {format_currency(results['initial_capital'])}")
        print(f"  Final Value: {format_currency(results['final_value'])}")
        print(f"  Total Return: {results['total_return_pct']:.2f}%")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate_pct']:.2f}%")
        
        return results
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure to train the model first")
        return None


def basic_portfolio_monitoring():
    """Demonstrate portfolio status monitoring."""
    print("\n=== Portfolio Monitoring Example ===")
    
    bot = UnifiedTradingBot()
    
    # Get current status
    status = bot.get_status()
    
    print("Bot Status:")
    print(f"  Is Running: {status['is_running']}")
    print(f"  Is Trained: {status['is_trained']}")
    print(f"  Last Prediction: {status['last_prediction_time']}")
    print(f"  Error Count: {status['error_count']}")
    
    portfolio_metrics = status['portfolio_metrics']
    if portfolio_metrics:
        print("\nPortfolio Metrics:")
        print(f"  Current Value: {format_currency(portfolio_metrics.get('current_value', 0))}")
        print(f"  Total Return: {portfolio_metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Max Drawdown: {portfolio_metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Open Positions: {portfolio_metrics.get('num_open_positions', 0)}")
        print(f"  Total Trades: {portfolio_metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {portfolio_metrics.get('win_rate_pct', 0):.2f}%")


def basic_configuration_example():
    """Demonstrate basic configuration customization."""
    print("\n=== Configuration Example ===")
    
    # Create and customize configuration
    config = Config()
    
    # Model configuration
    config.model.sequence_length = 60
    config.model.batch_size = 32
    config.model.learning_rate = 0.001
    config.model.epochs = 50
    
    # Data configuration
    config.data.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    config.data.lookback_days = 365
    config.data.prediction_horizon = 5
    config.data.threshold_pct = 0.02
    
    # Trading configuration
    config.trading.initial_capital = 100000
    config.trading.max_position_size = 0.15
    config.trading.max_positions = 5
    config.trading.paper_trading = True
    config.trading.min_confidence = 0.65
    
    # Risk management
    config.trading.stop_loss_pct = 0.05
    config.trading.take_profit_pct = 0.10
    config.trading.max_drawdown = 0.15
    
    print("Configuration Summary:")
    print(f"  Symbols: {config.data.symbols}")
    print(f"  Initial Capital: {format_currency(config.trading.initial_capital)}")
    print(f"  Max Position Size: {config.trading.max_position_size:.1%}")
    print(f"  Paper Trading: {config.trading.paper_trading}")
    print(f"  Model Sequence Length: {config.model.sequence_length}")
    print(f"  Training Epochs: {config.model.epochs}")
    
    # Initialize bot with custom config
    bot = UnifiedTradingBot()
    bot.config = config
    
    return bot, config


def data_exploration_example():
    """Demonstrate data fetching and exploration."""
    print("\n=== Data Exploration Example ===")
    
    from src.data_processor import DataProcessor
    
    config = Config()
    processor = DataProcessor(config)
    
    # Fetch data for a symbol
    symbol = 'AAPL'
    print(f"Fetching data for {symbol}")
    
    data = processor.fetch_data(symbol, period='1y', interval='1d')
    
    if data is not None:
        print(f"Fetched {len(data)} rows of data")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {list(data.columns)}")
        
        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        indicators = processor.calculate_technical_indicators(data)
        
        print(f"Total features: {len(indicators.columns)}")
        print("Sample indicators:", indicators.columns[:15].tolist())
        
        # Show recent data
        print(f"\nRecent data for {symbol}:")
        recent_data = data.tail(5)[['open', 'high', 'low', 'close', 'volume']]
        print(recent_data.round(2))
        
        return data, indicators
    else:
        print(f"Failed to fetch data for {symbol}")
        return None, None


def feature_analysis_example():
    """Demonstrate feature engineering analysis."""
    print("\n=== Feature Analysis Example ===")
    
    from src.data_processor import DataProcessor
    
    config = Config()
    processor = DataProcessor(config)
    
    # Get feature names without processing full data
    feature_names = processor.get_feature_names()
    
    print(f"Total available features: {len(feature_names)}")
    
    # Categorize features
    categories = {
        'trend': [f for f in feature_names if any(x in f for x in ['sma', 'ema', 'macd', 'adx'])],
        'momentum': [f for f in feature_names if any(x in f for x in ['rsi', 'stoch', 'williams'])],
        'volatility': [f for f in feature_names if any(x in f for x in ['bb_', 'atr', 'vol'])],
        'volume': [f for f in feature_names if any(x in f for x in ['obv', 'vpt', 'mfi', 'volume'])],
        'pattern': [f for f in feature_names if any(x in f for x in ['doji', 'hammer', 'engulfing'])],
        'time': [f for f in feature_names if any(x in f for x in ['hour', 'day', 'month'])]
    }
    
    print("\nFeature Categories:")
    for category, features in categories.items():
        print(f"  {category.capitalize()}: {len(features)} features")
        if features:
            print(f"    Examples: {features[:3]}")


def model_info_example():
    """Demonstrate model architecture information."""
    print("\n=== Model Architecture Example ===")
    
    from src.model_builder import ModelBuilder
    
    config = Config()
    model_builder = ModelBuilder(config)
    
    # Build model with example dimensions
    input_shape = (60, 150)  # 60 time steps, 150 features
    n_classes = 3  # Buy, Hold, Sell
    static_features = 20  # Time-based features
    
    print(f"Building model with:")
    print(f"  Input shape: {input_shape}")
    print(f"  Output classes: {n_classes}")
    print(f"  Static features: {static_features}")
    
    model = model_builder.build_model(input_shape, n_classes, static_features)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Get model summary
    print(f"Model layers: {len(model.layers)}")
    
    return model


def simple_trading_session():
    """Demonstrate a simple trading session."""
    print("\n=== Simple Trading Session Example ===")
    
    bot = UnifiedTradingBot()
    
    # This would typically require a trained model
    print("Running a single trading session...")
    print("Note: This requires a trained model and market data")
    
    try:
        # Run one trading session
        bot.run_trading_session()
        
        # Check status after session
        status = bot.get_status()
        print("Session completed!")
        print(f"Portfolio status: {status['portfolio_metrics']}")
        
    except Exception as e:
        print(f"Session failed: {e}")
        print("Make sure to train the model first and check market connectivity")


def main():
    """Run all basic examples."""
    print("Unified CNN-LSTM Trading Bot - Basic Usage Examples")
    print("=" * 60)
    
    # Setup logging
    config = Config()
    setup_logging(config)
    
    try:
        # Configuration example
        bot, config = basic_configuration_example()
        
        # Data exploration
        data, indicators = data_exploration_example()
        
        # Feature analysis
        feature_analysis_example()
        
        # Model architecture
        model = model_info_example()
        
        # Training example (commented out as it takes time)
        # bot = basic_training_example()
        
        # Prediction example (requires trained model)
        # prediction_bot = basic_prediction_example()
        
        # Backtesting example (requires trained model)
        # backtest_results = basic_backtesting_example()
        
        # Portfolio monitoring
        basic_portfolio_monitoring()
        
        # Simple trading session (requires trained model)
        # simple_trading_session()
        
        print("\n" + "=" * 60)
        print("Basic examples completed!")
        print("\nNext steps:")
        print("1. Train a model: bot.train_model()")
        print("2. Generate predictions: bot.predict_signals()")
        print("3. Run backtesting: bot.backtest()")
        print("4. Start paper trading: bot.start_automated_trading()")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed correctly")


if __name__ == "__main__":
    main()