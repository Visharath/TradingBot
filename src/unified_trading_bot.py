"""
Unified CNN-LSTM Trading Bot - Main Implementation.

This module contains the main trading bot class that coordinates:
- Real-time data processing and feature engineering
- Model inference for trading decisions
- Risk management and position sizing
- Trade execution through Interactive Brokers
- Performance monitoring and logging
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import ib_insync as ib
from threading import Thread, Event
import time
import json
from pathlib import Path

from .config import config
from .utils import logger, retry_on_exception, timing_decorator, calculate_performance_metrics
from .data_processor import DataProcessor
from .model_builder import UnifiedModelBuilder


class UnifiedTradingBot:
    """
    Main trading bot class that orchestrates the entire trading pipeline.
    
    Features:
    - Real-time data ingestion and processing
    - ML model inference for trading decisions
    - Risk management and position sizing
    - Automated trade execution
    - Performance monitoring and reporting
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the trading bot.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.data_processor = DataProcessor()
        self.model_builder = UnifiedModelBuilder()
        self.ib_client = None
        self.is_running = False
        self.stop_event = Event()
        
        # Trading state
        self.positions = {}
        self.open_orders = {}
        self.portfolio_value = 0.0
        self.cash_balance = 0.0
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        self.daily_returns = []
        
        # Data buffers
        self.price_data = {}
        self.processed_features = {}
        self.prediction_buffer = {}
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        
        logger.info("Unified Trading Bot initialized")
    
    async def connect_to_broker(self) -> bool:
        """
        Connect to Interactive Brokers.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ib_client = ib.IB()
            await self.ib_client.connectAsync(
                host=config.api.ib_host,
                port=config.api.ib_port,
                clientId=config.api.ib_client_id,
                timeout=config.api.ib_timeout
            )
            
            # Set up event handlers
            self.ib_client.orderStatusEvent += self._on_order_status
            self.ib_client.positionEvent += self._on_position_update
            self.ib_client.accountValueEvent += self._on_account_update
            
            logger.info("Successfully connected to Interactive Brokers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Interactive Brokers: {e}")
            return False
    
    def disconnect_from_broker(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib_client and self.ib_client.isConnected():
            self.ib_client.disconnect()
            logger.info("Disconnected from Interactive Brokers")
    
    @retry_on_exception(max_retries=3, delay=5.0)
    async def get_market_data(self, symbol: str, timeframe: str = "1min") -> pd.DataFrame:
        """
        Fetch real-time market data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            timeframe: Data timeframe ("1min", "5min", "15min")
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.ib_client or not self.ib_client.isConnected():
            raise ConnectionError("Not connected to broker")
        
        # Create contract
        contract = ib.Stock(symbol, "SMART", "USD")
        
        # Get historical data
        bars = await self.ib_client.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=timeframe,
            whatToShow="TRADES",
            useRTH=True
        )
        
        # Convert to DataFrame
        df = ib.util.df(bars)
        df.set_index('date', inplace=True)
        
        # Rename columns to standard format
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'average', 'barCount']
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    @timing_decorator
    def process_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process market data and generate features.
        
        Args:
            symbol: Trading symbol
            data: Raw market data
            
        Returns:
            Dictionary containing processed features and metadata
        """
        try:
            # Process data through the data processor
            features, targets, feature_names = self.data_processor.process_data(
                data, fit_transformers=False
            )
            
            # Extract latest features for prediction
            if len(features) >= config.model.sequence_length:
                # Sequence features (last sequence_length periods)
                sequence_features = features[-config.model.sequence_length:].reshape(
                    1, config.model.sequence_length, -1
                )
                
                # Static features (time-based, latest period)
                latest_time = data.index[-1] if hasattr(data, 'index') else datetime.now()
                static_features = self._extract_static_features(latest_time, data).reshape(1, -1)
                
                return {
                    "symbol": symbol,
                    "timestamp": latest_time,
                    "sequence_features": sequence_features,
                    "static_features": static_features,
                    "feature_names": feature_names,
                    "data_quality": "good",
                    "processed_at": datetime.now()
                }
            else:
                logger.warning(f"Insufficient data for {symbol}: {len(features)} < {config.model.sequence_length}")
                return {"symbol": symbol, "data_quality": "insufficient"}
                
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")
            return {"symbol": symbol, "data_quality": "error", "error": str(e)}
    
    def _extract_static_features(self, timestamp: datetime, data: pd.DataFrame) -> np.ndarray:
        """Extract static features for the current timestamp."""
        features = []
        
        # Time-based features
        features.extend([
            timestamp.hour,
            timestamp.dayofweek,
            timestamp.day,
            timestamp.month,
            np.sin(2 * np.pi * timestamp.hour / 24),
            np.cos(2 * np.pi * timestamp.hour / 24),
            np.sin(2 * np.pi * timestamp.dayofweek / 7),
            np.cos(2 * np.pi * timestamp.dayofweek / 7),
        ])
        
        # Market regime features (if we have enough data)
        if len(data) >= 50:
            close_prices = data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            
            features.extend([
                volatility,
                1 if volatility > config.trading.volatility_threshold else 0
            ])
        else:
            features.extend([0.02, 0])  # Default values
        
        return np.array(features)
    
    @timing_decorator
    def make_trading_decision(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make trading decision using the ML model.
        
        Args:
            processed_data: Processed market data with features
            
        Returns:
            Dictionary containing trading decision and confidence
        """
        if self.model_builder.model is None:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": "No model loaded"
            }
        
        if processed_data.get("data_quality") != "good":
            return {
                "action": "hold", 
                "confidence": 0.0,
                "reason": f"Poor data quality: {processed_data.get('data_quality')}"
            }
        
        try:
            # Make prediction
            sequence_features = processed_data["sequence_features"]
            static_features = processed_data["static_features"] 
            
            predictions = self.model_builder.predict(
                (sequence_features, static_features),
                return_probabilities=True
            )[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            # Map class to action
            action_map = {0: "buy", 1: "hold", 2: "sell"}
            action = action_map[predicted_class]
            
            # Apply confidence threshold
            if confidence < config.trading.confidence_threshold:
                action = "hold"
                reason = f"Low confidence: {confidence:.3f} < {config.trading.confidence_threshold}"
            else:
                reason = f"Model prediction: {action} with confidence {confidence:.3f}"
            
            return {
                "action": action,
                "confidence": confidence,
                "probabilities": predictions.tolist(),
                "reason": reason,
                "timestamp": processed_data.get("timestamp", datetime.now())
            }
            
        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": f"Error in prediction: {str(e)}"
            }
    
    def calculate_position_size(
        self,
        symbol: str,
        action: str,
        confidence: float,
        current_price: float
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            action: Trading action ("buy", "sell")
            confidence: Model confidence
            current_price: Current market price
            
        Returns:
            Position size (number of shares)
        """
        if action == "hold" or confidence < config.trading.confidence_threshold:
            return 0.0
        
        # Get portfolio value
        portfolio_value = max(self.portfolio_value, self.cash_balance)
        if portfolio_value <= 0:
            return 0.0
        
        # Calculate risk amount
        risk_amount = portfolio_value * config.trading.max_portfolio_risk
        
        # Position sizing based on method
        if config.trading.position_sizing_method == "fixed_fractional":
            base_position_value = portfolio_value * config.trading.max_position_size
        elif config.trading.position_sizing_method == "volatility_based":
            # Simplified volatility-based sizing
            volatility_factor = min(confidence * 2, 1.0)  # Use confidence as proxy
            base_position_value = portfolio_value * config.trading.max_position_size * volatility_factor
        else:  # Kelly criterion (simplified)
            kelly_fraction = min(confidence * config.trading.kelly_fraction, config.trading.max_position_size)
            base_position_value = portfolio_value * kelly_fraction
        
        # Calculate shares
        shares = base_position_value / current_price
        
        # Apply minimum and maximum position constraints
        min_value = portfolio_value * config.trading.min_position_size
        max_value = portfolio_value * config.trading.max_position_size
        
        min_shares = min_value / current_price
        max_shares = max_value / current_price
        
        shares = max(min_shares, min(shares, max_shares))
        
        # Round to whole shares
        return int(shares)
    
    async def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Execute a trade through Interactive Brokers.
        
        Args:
            symbol: Trading symbol
            action: "buy" or "sell"
            quantity: Number of shares
            current_price: Current market price
            
        Returns:
            Dictionary containing trade execution results
        """
        if not self.ib_client or not self.ib_client.isConnected():
            return {"status": "failed", "reason": "Not connected to broker"}
        
        if quantity <= 0:
            return {"status": "skipped", "reason": "Zero quantity"}
        
        try:
            # Create contract
            contract = ib.Stock(symbol, "SMART", "USD")
            
            # Create order
            order = ib.MarketOrder(action.upper(), quantity)
            
            # Add stop loss if configured
            if config.trading.stop_loss_atr_multiplier > 0:
                # Simplified stop loss calculation
                stop_price = current_price * (
                    1 - config.trading.stop_loss_atr_multiplier * 0.01
                    if action == "buy" else
                    1 + config.trading.stop_loss_atr_multiplier * 0.01
                )
                order.auxPrice = stop_price
            
            # Place order
            trade = self.ib_client.placeOrder(contract, order)
            
            # Wait for order confirmation
            await asyncio.sleep(1)
            
            # Log trade
            trade_record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "order_id": trade.order.orderId,
                "status": "submitted"
            }
            
            self.trade_history.append(trade_record)
            logger.info(f"Trade executed: {action} {quantity} shares of {symbol} at ${current_price:.2f}")
            
            return {
                "status": "success",
                "trade_record": trade_record,
                "order_id": trade.order.orderId
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Check if current portfolio state violates risk limits.
        
        Returns:
            Dictionary containing risk check results
        """
        risk_status = {
            "within_limits": True,
            "violations": [],
            "current_metrics": {}
        }
        
        # Calculate current drawdown
        if len(self.daily_returns) > 0:
            cumulative_returns = np.cumprod(1 + np.array(self.daily_returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            current_drawdown = abs(drawdown[-1])
            
            risk_status["current_metrics"]["drawdown"] = current_drawdown
            
            if current_drawdown > config.trading.max_drawdown_threshold:
                risk_status["within_limits"] = False
                risk_status["violations"].append(
                    f"Maximum drawdown exceeded: {current_drawdown:.2%} > {config.trading.max_drawdown_threshold:.2%}"
                )
        
        # Check position concentration
        if self.portfolio_value > 0:
            for symbol, position in self.positions.items():
                position_weight = abs(position.get("market_value", 0)) / self.portfolio_value
                if position_weight > config.trading.max_position_size:
                    risk_status["within_limits"] = False
                    risk_status["violations"].append(
                        f"Position size exceeded for {symbol}: {position_weight:.2%} > {config.trading.max_position_size:.2%}"
                    )
        
        return risk_status
    
    async def trading_loop(self) -> None:
        """Main trading loop that runs continuously."""
        logger.info("Starting trading loop...")
        
        while not self.stop_event.is_set():
            try:
                # Check risk limits first
                risk_check = self.check_risk_limits()
                if not risk_check["within_limits"]:
                    logger.warning(f"Risk limits violated: {risk_check['violations']}")
                    # In a real implementation, you might want to close positions or halt trading
                
                # Process each symbol
                for symbol in config.data.symbols:
                    try:
                        # Get market data
                        market_data = await self.get_market_data(symbol)
                        
                        if len(market_data) < config.model.sequence_length:
                            logger.warning(f"Insufficient data for {symbol}")
                            continue
                        
                        # Process data
                        processed_data = self.process_market_data(symbol, market_data)
                        
                        if processed_data.get("data_quality") != "good":
                            continue
                        
                        # Make trading decision
                        decision = self.make_trading_decision(processed_data)
                        
                        # Calculate position size
                        current_price = market_data['close'].iloc[-1]
                        position_size = self.calculate_position_size(
                            symbol, decision["action"], decision["confidence"], current_price
                        )
                        
                        # Execute trade if needed
                        if decision["action"] != "hold" and position_size > 0:
                            trade_result = await self.execute_trade(
                                symbol, decision["action"], position_size, current_price
                            )
                            
                            logger.info(f"Trade decision for {symbol}: {decision}")
                            logger.info(f"Trade result: {trade_result}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update performance metrics
                self.update_performance_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    def update_performance_metrics(self) -> None:
        """Update performance metrics and daily returns."""
        if len(self.trade_history) < 2:
            return
        
        # Calculate daily returns (simplified)
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history if t["timestamp"].date() == today]
        
        if today_trades:
            # Simple P&L calculation
            daily_pnl = sum(
                t["quantity"] * t["price"] * (1 if t["action"] == "sell" else -1)
                for t in today_trades
            )
            
            if self.portfolio_value > 0:
                daily_return = daily_pnl / self.portfolio_value
                self.daily_returns.append(daily_return)
                
                # Keep only last 252 days (1 year)
                if len(self.daily_returns) > 252:
                    self.daily_returns = self.daily_returns[-252:]
                
                # Calculate performance metrics
                if len(self.daily_returns) > 30:  # Need at least 30 days
                    returns_series = pd.Series(self.daily_returns)
                    self.performance_metrics = calculate_performance_metrics(returns_series)
                    
                    logger.info(f"Performance update - Sharpe: {self.performance_metrics.get('sharpe_ratio', 0):.2f}, "
                              f"Max DD: {self.performance_metrics.get('max_drawdown', 0):.2%}")
    
    def start_trading(self) -> None:
        """Start the trading bot."""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        async def run_bot():
            # Connect to broker
            connected = await self.connect_to_broker()
            if not connected:
                logger.error("Failed to connect to broker, cannot start trading")
                return
            
            self.is_running = True
            self.stop_event.clear()
            
            try:
                await self.trading_loop()
            finally:
                self.is_running = False
                self.disconnect_from_broker()
        
        # Run in asyncio event loop
        asyncio.run(run_bot())
    
    def stop_trading(self) -> None:
        """Stop the trading bot."""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        logger.info("Stopping trading bot...")
        self.stop_event.set()
        self.is_running = False
    
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model."""
        try:
            self.model_builder.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    
    def save_trading_state(self, filepath: str) -> None:
        """Save current trading state to file."""
        state = {
            "positions": self.positions,
            "trade_history": [
                {**trade, "timestamp": trade["timestamp"].isoformat()}
                for trade in self.trade_history
            ],
            "performance_metrics": self.performance_metrics,
            "daily_returns": self.daily_returns,
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.cash_balance
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Trading state saved to {filepath}")
    
    def load_trading_state(self, filepath: str) -> None:
        """Load trading state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.positions = state.get("positions", {})
            self.performance_metrics = state.get("performance_metrics", {})
            self.daily_returns = state.get("daily_returns", [])
            self.portfolio_value = state.get("portfolio_value", 0.0)
            self.cash_balance = state.get("cash_balance", 0.0)
            
            # Restore trade history with datetime objects
            self.trade_history = []
            for trade in state.get("trade_history", []):
                trade["timestamp"] = datetime.fromisoformat(trade["timestamp"])
                self.trade_history.append(trade)
            
            logger.info(f"Trading state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load trading state from {filepath}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status and metrics."""
        return {
            "is_running": self.is_running,
            "connected_to_broker": self.ib_client is not None and self.ib_client.isConnected() if self.ib_client else False,
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.cash_balance,
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "performance_metrics": self.performance_metrics,
            "model_loaded": self.model_builder.model is not None,
            "last_update": datetime.now().isoformat()
        }
    
    # Event handlers for Interactive Brokers
    def _on_order_status(self, trade):
        """Handle order status updates."""
        logger.info(f"Order status update: {trade.order.orderId} - {trade.orderStatus.status}")
        if trade.order.orderId in self.open_orders:
            self.open_orders[trade.order.orderId]["status"] = trade.orderStatus.status
    
    def _on_position_update(self, position):
        """Handle position updates."""
        symbol = position.contract.symbol
        self.positions[symbol] = {
            "position": position.position,
            "market_price": position.marketPrice,
            "market_value": position.marketValue,
            "average_cost": position.averageCost,
            "unrealized_pnl": position.unrealizedPNL
        }
        logger.debug(f"Position update for {symbol}: {position.position} shares")
    
    def _on_account_update(self, account_value):
        """Handle account value updates."""
        if account_value.tag == "NetLiquidation":
            self.portfolio_value = float(account_value.value)
        elif account_value.tag == "CashBalance":
            self.cash_balance = float(account_value.value)


__all__ = ["UnifiedTradingBot"]