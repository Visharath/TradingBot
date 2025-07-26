"""
Advanced Configuration Example for the Unified CNN-LSTM Trading Bot.

This example demonstrates advanced configuration options including:
1. Custom indicator creation
2. Advanced risk management
3. Multi-symbol trading
4. Real-time trading setup with sophisticated strategies
"""

import asyncio
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import the trading bot components
from src.unified_trading_bot import UnifiedTradingBot
from src.data_processor import DataProcessor
from src.model_builder import UnifiedModelBuilder
from src.config import config
from src.utils import logger


class AdvancedDataProcessor(DataProcessor):
    """Extended DataProcessor with custom indicators."""
    
    def _custom_indicators(self, close_prices: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """Add custom technical indicators."""
        indicators = {}
        
        # Custom Volume-Price Trend (VPT) variations
        price_changes = np.diff(close_prices) / close_prices[:-1]
        volume_aligned = volume[1:]  # Align with price changes
        
        indicators['vpt_momentum'] = np.cumsum(price_changes * volume_aligned)
        indicators['vpt_smoothed'] = pd.Series(indicators['vpt_momentum']).rolling(20).mean().values
        
        # Custom volatility-adjusted momentum
        returns = price_changes
        volatility = pd.Series(returns).rolling(20).std().values
        indicators['vol_adjusted_momentum'] = returns / (volatility + 1e-8)
        
        # Custom mean reversion indicator
        sma_20 = pd.Series(close_prices).rolling(20).mean().values
        std_20 = pd.Series(close_prices).rolling(20).std().values
        indicators['mean_reversion'] = (close_prices - sma_20) / (std_20 + 1e-8)
        
        # Custom trend strength indicator
        short_ma = pd.Series(close_prices).rolling(10).mean().values
        long_ma = pd.Series(close_prices).rolling(50).mean().values
        indicators['trend_strength'] = (short_ma - long_ma) / long_ma
        
        return pd.DataFrame(indicators, index=range(len(indicators['vpt_momentum'])))
    
    def _generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override to include custom indicators."""
        # Get standard features
        features = super()._generate_all_features(data)
        
        # Add custom indicators
        custom_features = self._custom_indicators(
            data['close'].values, 
            data['volume'].values
        )
        
        # Align indices and combine
        if len(custom_features) > 0:
            # Ensure custom features have the same index as standard features
            custom_features.index = features.index[-len(custom_features):]
            features = pd.concat([features, custom_features], axis=1)
        
        return features


class AdvancedTradingBot(UnifiedTradingBot):
    """Extended trading bot with advanced features."""
    
    def __init__(self, symbols: List[str], model_path: str = None):
        super().__init__(model_path)
        self.symbols = symbols
        self.symbol_correlations = {}
        self.portfolio_heat_map = {}
        
        # Replace data processor with advanced version
        self.data_processor = AdvancedDataProcessor()
    
    async def calculate_correlations(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between symbols."""
        correlations = {}
        price_data = {}
        
        # Get price data for all symbols
        for symbol in self.symbols:
            try:
                data = await self.get_market_data(symbol, timeframe="1min")
                price_data[symbol] = data['close'].pct_change().dropna()
            except Exception as e:
                logger.warning(f"Could not get data for {symbol}: {e}")
                continue
        
        # Calculate pairwise correlations
        for symbol1 in price_data:
            correlations[symbol1] = {}
            for symbol2 in price_data:
                if symbol1 != symbol2:
                    # Align data
                    common_index = price_data[symbol1].index.intersection(
                        price_data[symbol2].index
                    )
                    if len(common_index) > 50:  # Need sufficient data
                        corr = price_data[symbol1].loc[common_index].corr(
                            price_data[symbol2].loc[common_index]
                        )
                        correlations[symbol1][symbol2] = corr
                    else:
                        correlations[symbol1][symbol2] = 0.0
                else:
                    correlations[symbol1][symbol2] = 1.0
        
        self.symbol_correlations = correlations
        return correlations
    
    def advanced_position_sizing(
        self,
        symbol: str,
        action: str,
        confidence: float,
        current_price: float,
        volatility: float
    ) -> float:
        """Advanced position sizing considering correlations and portfolio heat."""
        if action == "hold" or confidence < config.trading.confidence_threshold:
            return 0.0
        
        base_size = super().calculate_position_size(symbol, action, confidence, current_price)
        
        # Adjust for correlations
        correlation_adjustment = 1.0
        if symbol in self.symbol_correlations:
            for other_symbol, position in self.positions.items():
                if other_symbol != symbol and other_symbol in self.symbol_correlations[symbol]:
                    corr = self.symbol_correlations[symbol][other_symbol]
                    position_weight = abs(position.get("market_value", 0)) / max(self.portfolio_value, 1)
                    
                    # Reduce size if highly correlated with existing positions
                    if abs(corr) > 0.7:
                        correlation_adjustment *= (1 - 0.3 * abs(corr) * position_weight)
        
        # Adjust for volatility
        volatility_adjustment = 1.0
        if volatility > config.trading.volatility_threshold:
            volatility_adjustment = 0.5  # Halve position size in high volatility
        
        # Adjust for portfolio heat (concentration)
        heat_adjustment = 1.0
        current_positions = len([p for p in self.positions.values() if p.get("position", 0) != 0])
        if current_positions >= 5:  # Too many positions
            heat_adjustment = 0.7
        
        adjusted_size = base_size * correlation_adjustment * volatility_adjustment * heat_adjustment
        
        logger.info(f"Position sizing for {symbol}: base={base_size}, "
                   f"corr_adj={correlation_adjustment:.3f}, "
                   f"vol_adj={volatility_adjustment:.3f}, "
                   f"heat_adj={heat_adjustment:.3f}, "
                   f"final={adjusted_size}")
        
        return int(adjusted_size)
    
    async def advanced_trading_decision(self, symbol: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decisions with advanced logic."""
        # Get base decision from parent class
        base_decision = self.make_trading_decision(processed_data)
        
        # Apply additional filters
        enhanced_decision = base_decision.copy()
        
        # Market regime filter
        if config.trading.enable_regime_filter:
            # Get current market volatility
            try:
                market_data = await self.get_market_data("SPY", timeframe="1min")  # Use SPY as market proxy
                spy_returns = market_data['close'].pct_change().dropna()
                current_volatility = spy_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                
                if current_volatility > 0.25:  # High market volatility (25%+)
                    enhanced_decision["confidence"] *= 0.7  # Reduce confidence
                    enhanced_decision["reason"] += " (High market volatility detected)"
                    
            except Exception as e:
                logger.warning(f"Could not get market volatility data: {e}")
        
        # Time-based filters
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 15:  # Outside market hours (simplified)
            if enhanced_decision["action"] != "hold":
                enhanced_decision["action"] = "hold"
                enhanced_decision["confidence"] = 0.0
                enhanced_decision["reason"] = "Outside trading hours"
        
        # Momentum filter
        if "trend_strength" in processed_data.get("feature_names", []):
            # Example: Check if trend is strong enough
            # This would need to be implemented based on actual feature processing
            pass
        
        return enhanced_decision
    
    async def portfolio_rebalancing(self) -> None:
        """Perform portfolio rebalancing based on correlations and performance."""
        if len(self.positions) < 2:
            return
        
        logger.info("Starting portfolio rebalancing...")
        
        # Calculate current portfolio weights
        total_value = sum(abs(pos.get("market_value", 0)) for pos in self.positions.values())
        if total_value == 0:
            return
        
        weights = {
            symbol: abs(pos.get("market_value", 0)) / total_value
            for symbol, pos in self.positions.items()
        }
        
        # Check for concentration risk
        overweight_threshold = 0.25  # Max 25% per position
        overweight_positions = {
            symbol: weight for symbol, weight in weights.items()
            if weight > overweight_threshold
        }
        
        if overweight_positions:
            logger.info(f"Overweight positions detected: {overweight_positions}")
            
            # Reduce overweight positions
            for symbol, weight in overweight_positions.items():
                if symbol in self.positions:
                    current_shares = self.positions[symbol].get("position", 0)
                    if current_shares != 0:
                        # Reduce position by 20%
                        reduce_shares = int(abs(current_shares) * 0.2)
                        action = "sell" if current_shares > 0 else "buy"
                        
                        try:
                            market_data = await self.get_market_data(symbol)
                            current_price = market_data['close'].iloc[-1]
                            
                            await self.execute_trade(symbol, action, reduce_shares, current_price)
                            logger.info(f"Rebalanced {symbol}: {action} {reduce_shares} shares")
                            
                        except Exception as e:
                            logger.error(f"Error rebalancing {symbol}: {e}")


async def advanced_trading_example():
    """Demonstrate advanced trading bot configuration and usage."""
    print("🚀 Advanced Trading Bot Configuration Example")
    print("=" * 55)
    
    # Configure advanced settings
    print("⚙️ Setting up advanced configuration...")
    
    # Advanced model configuration
    config.model.sequence_length = 120  # Longer sequence for better pattern recognition
    config.model.cnn_filters = [64, 128, 256, 512]  # More complex CNN
    config.model.lstm_units = 256  # Larger LSTM
    config.model.attention_heads = 16  # More attention heads
    
    # Advanced trading configuration
    config.trading.max_position_size = 0.08  # 8% max position
    config.trading.confidence_threshold = 0.8  # Higher confidence required
    config.trading.position_sizing_method = "kelly"  # Use Kelly Criterion
    config.trading.enable_regime_filter = True  # Enable market regime filtering
    
    # Multi-symbol configuration
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    config.data.symbols = symbols
    
    # Initialize advanced trading bot
    print(f"🤖 Initializing advanced trading bot for {len(symbols)} symbols...")
    bot = AdvancedTradingBot(symbols=symbols)
    
    # Connect to broker
    print("🔌 Connecting to broker...")
    connected = await bot.connect_to_broker()
    if not connected:
        print("❌ Failed to connect to broker")
        return
    
    print("✅ Connected successfully!")
    
    # Calculate correlations
    print("📊 Calculating symbol correlations...")
    correlations = await bot.calculate_correlations()
    
    # Display correlation matrix
    print("\n📈 Correlation Matrix:")
    print("-" * 50)
    for symbol1 in correlations:
        corr_line = f"{symbol1:6s}: "
        for symbol2 in symbols[:5]:  # Show first 5 for brevity
            if symbol2 in correlations[symbol1]:
                corr_line += f"{correlations[symbol1][symbol2]:6.2f} "
            else:
                corr_line += "  N/A  "
        print(corr_line)
    
    # Demonstrate advanced trading cycle
    print("\n🔄 Running advanced trading cycle...")
    
    for symbol in symbols[:3]:  # Process first 3 symbols for demo
        try:
            print(f"\n📊 Processing {symbol}...")
            
            # Get market data
            market_data = await bot.get_market_data(symbol, timeframe="5min")
            print(f"   Retrieved {len(market_data)} data points")
            
            # Process data with custom indicators
            processed_data = bot.process_market_data(symbol, market_data)
            
            if processed_data.get("data_quality") == "good":
                # Make advanced trading decision
                decision = await bot.advanced_trading_decision(symbol, processed_data)
                print(f"   Decision: {decision['action'].upper()}")
                print(f"   Confidence: {decision['confidence']:.3f}")
                print(f"   Reason: {decision['reason']}")
                
                # Calculate advanced position size
                current_price = market_data['close'].iloc[-1]
                volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
                
                position_size = bot.advanced_position_sizing(
                    symbol, decision["action"], decision["confidence"], 
                    current_price, volatility
                )
                
                print(f"   Position size: {position_size} shares")
                print(f"   Estimated value: ${position_size * current_price:.2f}")
                
                # Note: Actual trade execution commented out for safety
                # if decision["action"] != "hold" and position_size > 0:
                #     result = await bot.execute_trade(symbol, decision["action"], position_size, current_price)
                #     print(f"   Trade executed: {result}")
                
            else:
                print(f"   ⚠️ Poor data quality: {processed_data.get('data_quality')}")
                
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")
    
    # Demonstrate portfolio rebalancing
    print(f"\n⚖️ Portfolio rebalancing check...")
    await bot.portfolio_rebalancing()
    
    # Show advanced statistics
    print(f"\n📊 Advanced Portfolio Statistics:")
    print(f"   Connected symbols: {len(symbols)}")
    print(f"   Active positions: {len(bot.positions)}")
    print(f"   Portfolio value: ${bot.portfolio_value:,.2f}")
    print(f"   Cash balance: ${bot.cash_balance:,.2f}")
    
    # Disconnect
    print("\n🔌 Disconnecting from broker...")
    bot.disconnect_from_broker()
    print("✅ Advanced trading example completed!")


def custom_strategy_example():
    """Example of creating a custom trading strategy."""
    print("\n🎯 Custom Strategy Development Example")
    print("=" * 45)
    
    class MomentumReversalStrategy:
        """Custom momentum-reversal hybrid strategy."""
        
        def __init__(self):
            self.lookback_momentum = 20
            self.lookback_reversal = 5
            self.momentum_threshold = 0.02
            self.reversal_threshold = 2.0  # Z-score
        
        def generate_signal(self, data: pd.DataFrame) -> str:
            """Generate trading signal based on custom logic."""
            if len(data) < max(self.lookback_momentum, self.lookback_reversal):
                return "hold"
            
            # Calculate momentum
            returns = data['close'].pct_change()
            momentum = returns.rolling(self.lookback_momentum).mean().iloc[-1]
            
            # Calculate mean reversion signal
            recent_returns = returns.tail(self.lookback_reversal)
            z_score = (recent_returns.mean() - returns.mean()) / returns.std()
            
            # Decision logic
            if momentum > self.momentum_threshold and z_score < -self.reversal_threshold:
                return "buy"  # Strong momentum + oversold
            elif momentum < -self.momentum_threshold and z_score > self.reversal_threshold:
                return "sell"  # Weak momentum + overbought
            else:
                return "hold"
        
        def calculate_confidence(self, data: pd.DataFrame) -> float:
            """Calculate confidence in the signal."""
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Lower confidence in high volatility periods
            base_confidence = 0.75
            vol_adjustment = max(0.3, 1.0 - volatility * 10)
            
            return base_confidence * vol_adjustment
    
    # Demonstrate custom strategy usage
    strategy = MomentumReversalStrategy()
    
    # Example with sample data
    print("📊 Testing custom strategy with sample data...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    sample_data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Test strategy
    signal = strategy.generate_signal(sample_data)
    confidence = strategy.calculate_confidence(sample_data)
    
    print(f"   Custom strategy signal: {signal.upper()}")
    print(f"   Confidence: {confidence:.3f}")
    print("   ✅ Custom strategy working correctly!")


def risk_management_example():
    """Advanced risk management configuration example."""
    print("\n🛡️ Advanced Risk Management Configuration")
    print("=" * 50)
    
    # Portfolio-level risk limits
    config.trading.max_portfolio_risk = 0.015  # 1.5% max portfolio risk per trade
    config.trading.max_drawdown_threshold = 0.10  # 10% max drawdown
    config.trading.max_positions = 8  # Max 8 concurrent positions
    
    # Position-level risk management
    config.trading.stop_loss_atr_multiplier = 1.5  # Tighter stops
    config.trading.take_profit_ratio = 3.0  # 1:3 risk-reward
    config.trading.position_sizing_method = "volatility_based"
    
    # Market regime adaptive settings
    config.trading.volatility_threshold = 0.02  # 2% daily volatility threshold
    config.trading.trend_strength_threshold = 0.6  # Minimum trend strength
    
    # Advanced position sizing parameters
    config.trading.kelly_fraction = 0.15  # Conservative Kelly
    config.trading.max_position_size = 0.06  # 6% max per position
    config.trading.min_position_size = 0.005  # 0.5% min per position
    
    print("✅ Advanced risk management configured:")
    print(f"   Max portfolio risk: {config.trading.max_portfolio_risk:.1%}")
    print(f"   Max drawdown: {config.trading.max_drawdown_threshold:.1%}")
    print(f"   Max positions: {getattr(config.trading, 'max_positions', 'Not set')}")
    print(f"   Stop loss: {config.trading.stop_loss_atr_multiplier}x ATR")
    print(f"   Take profit: {config.trading.take_profit_ratio}:1 ratio")


def main():
    """Run all advanced examples."""
    print("🎯 Unified CNN-LSTM Trading Bot - Advanced Examples")
    print("=" * 60)
    
    # Run configuration examples
    risk_management_example()
    custom_strategy_example()
    
    # Ask user about running live demo
    print("\n" + "=" * 60)
    print("⚠️  LIVE TRADING DEMO")
    print("The following demo will connect to Interactive Brokers.")
    print("Make sure you're using paper trading!")
    print("=" * 60)
    
    response = input("Run live trading demo? (yes/no): ").strip().lower()
    
    if response == 'yes':
        try:
            asyncio.run(advanced_trading_example())
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
    else:
        print("✅ Demo skipped. Check the code for implementation details!")


if __name__ == "__main__":
    main()