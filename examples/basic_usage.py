"""
Basic Usage Example for the Unified CNN-LSTM Trading Bot.

This example demonstrates the most basic way to use the trading bot:
1. Initialize the bot
2. Load or train a model
3. Start trading

This is suitable for users who want to get started quickly with default settings.
"""

import asyncio
import yfinance as yf
from datetime import datetime, timedelta

# Import the trading bot components
from src.unified_trading_bot import UnifiedTradingBot
from src.data_processor import DataProcessor
from src.model_builder import UnifiedModelBuilder
from src.config import config
from src.utils import logger


def basic_model_training_example():
    """
    Example of training a basic model with default settings.
    This should be run before using the trading bot for the first time.
    """
    logger.info("Starting basic model training example...")
    
    # Step 1: Get training data
    print("📊 Downloading training data...")
    symbol = "AAPL"  # Apple stock as example
    data = yf.download(symbol, period="1y", interval="1h")
    
    if len(data) < 1000:
        print("❌ Insufficient data for training. Need at least 1000 data points.")
        return None
    
    print(f"✅ Downloaded {len(data)} data points for {symbol}")
    
    # Step 2: Process the data
    print("🔧 Processing data and generating features...")
    processor = DataProcessor()
    features, targets, feature_names = processor.process_data(data)
    
    print(f"✅ Generated {len(feature_names)} features")
    print(f"📈 Processing {len(features)} samples")
    
    # Step 3: Prepare data for training
    from src.model_builder import prepare_model_inputs
    
    # Create some static features (simplified for example)
    static_features = processor._extract_static_features(
        datetime.now(), data
    )
    static_data = [static_features] * len(features)
    
    # Prepare training inputs
    model_inputs = prepare_model_inputs(
        sequence_data=features,
        static_data=static_data,
        targets=targets,
        sequence_length=config.model.sequence_length
    )
    
    # Step 4: Build and train the model
    print("🏗️ Building model architecture...")
    model_builder = UnifiedModelBuilder()
    
    # Calculate input dimensions
    n_features = len(feature_names)
    static_dim = len(static_features)
    
    model = model_builder.build_model(
        input_shape=(config.model.sequence_length, n_features),
        static_features_dim=static_dim
    )
    
    print("🚀 Starting model training...")
    print("This may take several minutes to hours depending on your hardware...")
    
    # Train the model
    training_results = model_builder.train_model(
        X_train=model_inputs["train"][0],
        y_train=model_inputs["train"][1],
        X_val=model_inputs["val"][0],
        y_val=model_inputs["val"][1]
    )
    
    print("✅ Model training completed!")
    print(f"📊 Best validation accuracy: {training_results['best_val_accuracy']:.4f}")
    print(f"💾 Model saved to: {training_results['model_path']}")
    
    return training_results["model_path"]


async def basic_trading_example(model_path=None):
    """
    Example of basic trading bot usage.
    
    Args:
        model_path: Path to a trained model (optional)
    """
    logger.info("Starting basic trading example...")
    
    # Step 1: Initialize the trading bot
    print("🤖 Initializing trading bot...")
    bot = UnifiedTradingBot(model_path=model_path)
    
    # Step 2: Configure for paper trading (safe for testing)
    print("⚙️ Configuring for paper trading...")
    config.api.ib_port = 7497  # Paper trading port
    config.trading.confidence_threshold = 0.8  # Higher threshold for safety
    config.data.symbols = ["AAPL"]  # Trade only Apple for this example
    
    # Step 3: Connect to Interactive Brokers
    print("🔌 Connecting to Interactive Brokers...")
    print("Make sure TWS or IB Gateway is running with API enabled!")
    
    connected = await bot.connect_to_broker()
    if not connected:
        print("❌ Failed to connect to broker. Please check:")
        print("   1. TWS or IB Gateway is running")
        print("   2. API is enabled in TWS settings")
        print("   3. Port 7497 is configured for paper trading")
        print("   4. 127.0.0.1 is in trusted IPs")
        return
    
    print("✅ Successfully connected to broker!")
    
    # Step 4: Get current bot status
    status = bot.get_status()
    print(f"📊 Bot Status:")
    print(f"   - Connected to broker: {status['connected_to_broker']}")
    print(f"   - Model loaded: {status['model_loaded']}")
    print(f"   - Portfolio value: ${status['portfolio_value']:,.2f}")
    
    # Step 5: Demonstrate single trading cycle
    print("\n🔄 Running single trading cycle...")
    
    try:
        # Get market data
        symbol = "AAPL"
        market_data = await bot.get_market_data(symbol, timeframe="1min")
        print(f"📈 Retrieved {len(market_data)} data points for {symbol}")
        
        # Process the data
        processed_data = bot.process_market_data(symbol, market_data)
        print(f"🔧 Data processing result: {processed_data.get('data_quality', 'unknown')}")
        
        if processed_data.get("data_quality") == "good":
            # Make trading decision
            decision = bot.make_trading_decision(processed_data)
            print(f"🎯 Trading Decision:")
            print(f"   - Action: {decision['action'].upper()}")
            print(f"   - Confidence: {decision['confidence']:.3f}")
            print(f"   - Reason: {decision['reason']}")
            
            # Calculate position size
            current_price = market_data['close'].iloc[-1]
            position_size = bot.calculate_position_size(
                symbol, decision["action"], decision["confidence"], current_price
            )
            
            print(f"📊 Position Sizing:")
            print(f"   - Current price: ${current_price:.2f}")
            print(f"   - Recommended size: {position_size} shares")
            print(f"   - Estimated value: ${position_size * current_price:.2f}")
            
            # Note: In this example, we don't actually execute trades
            # Uncomment the following lines to execute real trades (use with caution!)
            
            # if decision["action"] != "hold" and position_size > 0:
            #     trade_result = await bot.execute_trade(
            #         symbol, decision["action"], position_size, current_price
            #     )
            #     print(f"💼 Trade Result: {trade_result}")
        
    except Exception as e:
        print(f"❌ Error during trading cycle: {e}")
    
    # Step 6: Disconnect from broker
    print("\n🔌 Disconnecting from broker...")
    bot.disconnect_from_broker()
    print("✅ Disconnected successfully")


def continuous_trading_example(model_path=None):
    """
    Example of running the bot in continuous trading mode.
    
    WARNING: This will run indefinitely and execute real trades!
    Only use this with paper trading or small amounts.
    """
    print("⚠️  WARNING: This will start continuous trading!")
    print("Make sure you're using paper trading before proceeding.")
    
    response = input("Continue with continuous trading? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Initialize bot
    bot = UnifiedTradingBot(model_path=model_path)
    
    # Configure for conservative trading
    config.trading.max_position_size = 0.02  # Max 2% per position
    config.trading.confidence_threshold = 0.85  # High confidence required
    
    try:
        print("🚀 Starting continuous trading...")
        print("Press Ctrl+C to stop")
        
        # This will run until manually stopped
        bot.start_trading()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping trading bot...")
        bot.stop_trading()
        print("✅ Trading bot stopped")


def main():
    """Main function to run the basic examples."""
    print("🎯 Unified CNN-LSTM Trading Bot - Basic Usage Example")
    print("=" * 55)
    
    # Choose what to run
    print("\nWhat would you like to do?")
    print("1. Train a new model")
    print("2. Run a single trading cycle (safe demo)")
    print("3. Start continuous trading (use with caution!)")
    print("4. Run complete pipeline (train + demo)")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Train a new model
            model_path = basic_model_training_example()
            if model_path:
                print(f"\n✅ Model training completed! Model saved to: {model_path}")
                
        elif choice == "2":
            # Demo trading cycle
            model_path = input("Enter model path (or press Enter to skip): ").strip()
            if not model_path:
                model_path = None
            asyncio.run(basic_trading_example(model_path))
            
        elif choice == "3":
            # Continuous trading
            model_path = input("Enter model path (required): ").strip()
            if not model_path:
                print("❌ Model path is required for continuous trading")
                return
            continuous_trading_example(model_path)
            
        elif choice == "4":
            # Complete pipeline
            print("\n🔄 Running complete pipeline...")
            model_path = basic_model_training_example()
            if model_path:
                print("\n🎯 Now running trading demo...")
                asyncio.run(basic_trading_example(model_path))
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()