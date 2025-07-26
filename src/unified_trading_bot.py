"""
Unified CNN-LSTM Trading Bot - Main Implementation

This module contains the main TradingBot class that orchestrates all components:
- Data processing and feature engineering
- Model training and inference
- Risk management and position sizing
- Trade execution and monitoring
- Performance tracking and reporting
"""

import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from loguru import logger
import schedule

from src.config import Config
from src.data_processor import DataProcessor
from src.model_builder import ModelBuilder
from src.utils import (
    setup_logging, validate_config, save_predictions,
    format_currency, calculate_percentage_change, is_market_open,
    memory_usage, print_system_info
)


class PortfolioManager:
    """Manages portfolio state, positions, and risk metrics."""
    
    def __init__(self, initial_capital: float, max_positions: int = 5):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"Portfolio initialized with {format_currency(initial_capital)}")
    
    def add_position(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: int,
        confidence: float,
        timestamp: datetime
    ) -> bool:
        """Add a new position to the portfolio."""
        if len(self.positions) >= self.max_positions and action in ['buy', 'long']:
            logger.warning(f"Maximum positions ({self.max_positions}) reached, cannot open new position")
            return False
        
        position_value = price * quantity
        
        if action in ['buy', 'long']:
            if position_value > self.current_capital:
                logger.warning(f"Insufficient capital for position: {format_currency(position_value)}")
                return False
            
            self.positions[symbol] = {
                'action': action,
                'entry_price': price,
                'quantity': quantity,
                'entry_time': timestamp,
                'confidence': confidence,
                'unrealized_pnl': 0.0
            }
            
            self.current_capital -= position_value
            
        elif action in ['sell', 'short']:
            if symbol in self.positions:
                # Close existing position
                position = self.positions[symbol]
                pnl = (price - position['entry_price']) * position['quantity']
                
                if position['action'] in ['short', 'sell']:
                    pnl = -pnl  # Reverse for short positions
                
                self.current_capital += (position['entry_price'] * position['quantity']) + pnl
                
                # Record trade
                self._record_trade(symbol, position, price, timestamp, pnl)
                
                del self.positions[symbol]
            else:
                logger.warning(f"No position to close for {symbol}")
                return False
        
        self._update_equity_curve(timestamp)
        logger.info(f"Position {action} {symbol} at {format_currency(price)} (confidence: {confidence:.3f})")
        return True
    
    def _record_trade(
        self,
        symbol: str,
        position: Dict,
        exit_price: float,
        exit_time: datetime,
        pnl: float
    ):
        """Record a completed trade."""
        trade = {
            'symbol': symbol,
            'action': position['action'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'duration': exit_time - position['entry_time'],
            'pnl': pnl,
            'pnl_pct': pnl / (position['entry_price'] * position['quantity']) * 100,
            'confidence': position['confidence']
        }
        
        self.trade_history.append(trade)
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        
        logger.info(f"Trade closed: {symbol} PnL: {format_currency(pnl)} ({trade['pnl_pct']:.2f}%)")
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update the equity curve with current portfolio value."""
        total_value = self.current_capital
        
        # Add unrealized PnL from open positions
        for symbol, position in self.positions.items():
            # Note: We would need current market price to calculate unrealized PnL
            # For now, we'll use entry price (no unrealized PnL)
            total_value += position['entry_price'] * position['quantity']
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': total_value - self.current_capital,
            'num_positions': len(self.positions)
        })
        
        # Update drawdown metrics
        if total_value > self.peak_capital:
            self.peak_capital = total_value
        
        current_drawdown = (self.peak_capital - total_value) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if not self.equity_curve:
            return {}
        
        current_value = self.equity_curve[-1]['total_value']
        total_return = (current_value - self.initial_capital) / self.initial_capital * 100
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate average trade metrics
        if self.trade_history:
            avg_trade_pnl = np.mean([trade['pnl'] for trade in self.trade_history])
            avg_trade_duration = np.mean([(trade['exit_time'] - trade['entry_time']).total_seconds() / 3600 
                                        for trade in self.trade_history])  # in hours
        else:
            avg_trade_pnl = 0
            avg_trade_duration = 0
        
        return {
            'total_return_pct': total_return,
            'current_value': current_value,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': self.total_trades,
            'win_rate_pct': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_trade_duration_hours': avg_trade_duration,
            'num_open_positions': len(self.positions)
        }


class RiskManager:
    """Manages trading risks and position sizing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_position_size = config.trading.max_position_size
        self.max_drawdown = config.trading.max_drawdown
        self.stop_loss_pct = config.trading.stop_loss_pct
        self.take_profit_pct = config.trading.take_profit_pct
        self.min_confidence = config.trading.min_confidence
        
    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float,
        volatility: float = 0.02
    ) -> int:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            portfolio_value: Current portfolio value
            price: Entry price
            confidence: Model confidence score
            volatility: Asset volatility (default 2%)
            
        Returns:
            Number of shares to trade
        """
        if confidence < self.min_confidence:
            return 0
        
        # Base position size as percentage of portfolio
        base_size = portfolio_value * self.max_position_size
        
        # Adjust for confidence (higher confidence = larger position)
        confidence_multiplier = confidence / 1.0  # Normalize confidence
        adjusted_size = base_size * confidence_multiplier
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_multiplier = min(1.0, 0.02 / max(volatility, 0.01))  # Cap at 2% base volatility
        risk_adjusted_size = adjusted_size * volatility_multiplier
        
        # Calculate number of shares
        shares = int(risk_adjusted_size / price)
        
        return max(0, shares)
    
    def should_enter_trade(
        self,
        signal: int,
        confidence: float,
        current_positions: int,
        portfolio_drawdown: float
    ) -> bool:
        """
        Determine if a trade should be entered based on risk criteria.
        
        Args:
            signal: Trading signal (0=sell, 1=hold, 2=buy)
            confidence: Model confidence
            current_positions: Number of current positions
            portfolio_drawdown: Current portfolio drawdown
            
        Returns:
            True if trade should be entered
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return False
        
        # Check if signal is actionable
        if signal == 1:  # Hold signal
            return False
        
        # Check maximum drawdown
        if portfolio_drawdown > self.max_drawdown:
            logger.warning(f"Maximum drawdown exceeded: {portfolio_drawdown:.2%}")
            return False
        
        # Check position limits for new positions
        if signal == 2 and current_positions >= self.config.trading.max_positions:
            return False
        
        return True
    
    def check_stop_loss_take_profit(
        self,
        position: Dict,
        current_price: float
    ) -> Optional[str]:
        """
        Check if position should be closed due to stop loss or take profit.
        
        Args:
            position: Position dictionary
            current_price: Current market price
            
        Returns:
            'stop_loss' or 'take_profit' if position should be closed, None otherwise
        """
        entry_price = position['entry_price']
        action = position['action']
        
        if action in ['buy', 'long']:
            # Long position
            loss_pct = (entry_price - current_price) / entry_price
            gain_pct = (current_price - entry_price) / entry_price
            
            if loss_pct >= self.stop_loss_pct:
                return 'stop_loss'
            elif gain_pct >= self.take_profit_pct:
                return 'take_profit'
        
        elif action in ['sell', 'short']:
            # Short position
            loss_pct = (current_price - entry_price) / entry_price
            gain_pct = (entry_price - current_price) / entry_price
            
            if loss_pct >= self.stop_loss_pct:
                return 'stop_loss'
            elif gain_pct >= self.take_profit_pct:
                return 'take_profit'
        
        return None


class UnifiedTradingBot:
    """
    Main unified CNN-LSTM trading bot that orchestrates all components.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the unified trading bot.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        if config_path and config_path.exists():
            # TODO: Implement config loading from file
            self.config = Config()
        else:
            self.config = Config()
        
        # Setup logging
        setup_logging(self.config)
        
        # Validate configuration
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.portfolio_manager = PortfolioManager(
            self.config.trading.initial_capital,
            self.config.trading.max_positions
        )
        self.risk_manager = RiskManager(self.config)
        
        # State variables
        self.is_running = False
        self.is_trained = False
        self.last_prediction_time = None
        self.current_market_data = {}
        
        # Performance tracking
        self.prediction_history = []
        self.error_count = 0
        self.last_error_time = None
        
        logger.info("UnifiedTradingBot initialized successfully")
        print_system_info()
    
    def train_model(
        self,
        symbols: Optional[List[str]] = None,
        save_model: bool = True
    ) -> Dict[str, float]:
        """
        Train the CNN-LSTM model with historical data.
        
        Args:
            symbols: List of symbols to train on (uses config default if None)
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        if symbols is None:
            symbols = self.config.data.symbols
        
        logger.info(f"Starting model training for symbols: {symbols}")
        
        all_features = []
        all_targets = []
        
        # Process each symbol
        for symbol in symbols:
            logger.info(f"Processing training data for {symbol}")
            
            result = self.data_processor.process_symbol(symbol)
            if result is not None:
                features, targets = result
                all_features.append(features)
                all_targets.append(targets)
            else:
                logger.warning(f"Failed to process {symbol}, skipping")
        
        if not all_features:
            raise ValueError("No training data available")
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        logger.info(f"Combined training data: {len(combined_features)} samples")
        
        # Scale features
        scaled_features, self.feature_scaler = self.data_processor.scale_features(combined_features)
        
        # Train model
        start_time = time.time()
        metrics = self.model_builder.train(scaled_features, combined_targets)
        training_time = time.time() - start_time
        
        metrics['training_time_seconds'] = training_time
        
        # Save model if requested
        if save_model:
            model_dir = self.config.system.models_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model_builder.save_model(model_dir, metrics)
            
            # Save feature scaler
            from src.utils import save_pickle
            save_pickle(self.feature_scaler, model_dir / 'feature_scaler.pkl')
            
            logger.info(f"Model saved to {model_dir}")
        
        self.is_trained = True
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return metrics
    
    def load_model(self, model_path: Path) -> bool:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            if not self.model_builder.load_model(model_path):
                return False
            
            # Load feature scaler
            scaler_path = model_path / 'feature_scaler.pkl'
            if scaler_path.exists():
                from src.utils import load_pickle
                self.feature_scaler = load_pickle(scaler_path)
            else:
                logger.warning("Feature scaler not found, will need to retrain or provide scaler")
            
            self.is_trained = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_signals(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Generate trading signals for the specified symbols.
        
        Args:
            symbols: List of symbols to predict (uses config default if None)
            
        Returns:
            Dictionary with predictions for each symbol
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() or load_model() first.")
        
        if symbols is None:
            symbols = self.config.data.symbols
        
        predictions = {}
        
        for symbol in symbols:
            try:
                # Get recent data
                data = self.data_processor.fetch_data(
                    symbol,
                    period="1y",  # Get enough history for indicators
                    interval="1d"
                )
                
                if data is None or len(data) < self.config.model.sequence_length:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Calculate technical indicators
                data_with_indicators = self.data_processor.calculate_technical_indicators(data)
                
                # Prepare features (take only the most recent data needed for prediction)
                recent_data = data_with_indicators.tail(self.config.model.sequence_length + 10)
                features, _ = self.data_processor.create_features_and_targets(recent_data)
                
                if len(features) == 0:
                    logger.warning(f"No features generated for {symbol}")
                    continue
                
                # Scale features
                scaled_features, _ = self.data_processor.scale_features(features, self.feature_scaler)
                
                # Take only the last row for prediction
                latest_features = scaled_features.tail(1)
                
                # Make prediction
                prediction, probabilities = self.model_builder.predict(
                    latest_features,
                    return_probabilities=True
                )
                
                # Get confidence score
                if len(probabilities.shape) > 1:
                    confidence = np.max(probabilities[0])
                else:
                    confidence = max(probabilities[0], 1 - probabilities[0])
                
                signal = prediction[0]
                current_price = data['close'].iloc[-1]
                
                predictions[symbol] = {
                    'signal': int(signal),
                    'signal_name': self.config.model.class_names[signal],
                    'confidence': float(confidence),
                    'probabilities': probabilities[0].tolist() if len(probabilities.shape) > 1 else [1-probabilities[0], probabilities[0]],
                    'current_price': float(current_price),
                    'timestamp': datetime.now()
                }
                
                logger.info(f"{symbol}: {predictions[symbol]['signal_name']} (confidence: {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
                self.error_count += 1
                self.last_error_time = datetime.now()
        
        # Store prediction history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predictions': predictions.copy()
        })
        
        self.last_prediction_time = datetime.now()
        
        return predictions
    
    def execute_trades(self, predictions: Dict[str, Dict]) -> List[Dict]:
        """
        Execute trades based on model predictions.
        
        Args:
            predictions: Dictionary with predictions for each symbol
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        portfolio_metrics = self.portfolio_manager.get_performance_metrics()
        current_drawdown = portfolio_metrics.get('max_drawdown_pct', 0) / 100
        
        for symbol, pred in predictions.items():
            signal = pred['signal']
            confidence = pred['confidence']
            price = pred['current_price']
            
            # Check if trade should be executed
            if not self.risk_manager.should_enter_trade(
                signal,
                confidence,
                len(self.portfolio_manager.positions),
                current_drawdown
            ):
                continue
            
            # Calculate position size
            portfolio_value = portfolio_metrics.get('current_value', self.config.trading.initial_capital)
            position_size = self.risk_manager.calculate_position_size(
                portfolio_value,
                price,
                confidence
            )
            
            if position_size == 0:
                continue
            
            # Execute trade
            if signal == 2:  # Buy signal
                action = 'buy'
            elif signal == 0:  # Sell signal
                action = 'sell'
            else:
                continue  # Hold signal
            
            success = self.portfolio_manager.add_position(
                symbol,
                action,
                price,
                position_size,
                confidence,
                pred['timestamp']
            )
            
            if success:
                trade = {
                    'symbol': symbol,
                    'action': action,
                    'price': price,
                    'quantity': position_size,
                    'confidence': confidence,
                    'timestamp': pred['timestamp']
                }
                executed_trades.append(trade)
                
                logger.info(f"Trade executed: {action} {position_size} shares of {symbol} at {format_currency(price)}")
        
        return executed_trades
    
    def check_existing_positions(self):
        """Check existing positions for stop loss or take profit conditions."""
        positions_to_close = []
        
        for symbol, position in self.portfolio_manager.positions.items():
            try:
                # Get current price
                current_data = self.data_processor.fetch_data(symbol, period="1d", interval="1m")
                if current_data is None or len(current_data) == 0:
                    continue
                
                current_price = current_data['close'].iloc[-1]
                
                # Check stop loss / take profit
                close_reason = self.risk_manager.check_stop_loss_take_profit(position, current_price)
                
                if close_reason:
                    positions_to_close.append((symbol, close_reason, current_price))
                    
            except Exception as e:
                logger.error(f"Error checking position for {symbol}: {e}")
        
        # Close positions
        for symbol, reason, price in positions_to_close:
            success = self.portfolio_manager.add_position(
                symbol,
                'sell',
                price,
                self.portfolio_manager.positions[symbol]['quantity'],
                1.0,  # Full confidence for risk management closure
                datetime.now()
            )
            
            if success:
                logger.info(f"Position closed due to {reason}: {symbol} at {format_currency(price)}")
    
    def run_trading_session(self):
        """Run a single trading session (prediction + execution)."""
        try:
            logger.info("Starting trading session...")
            
            # Check if market is open
            if not is_market_open():
                logger.info("Market is closed, skipping trading session")
                return
            
            # Check existing positions
            self.check_existing_positions()
            
            # Generate predictions
            predictions = self.predict_signals()
            
            if not predictions:
                logger.warning("No predictions generated")
                return
            
            # Execute trades
            executed_trades = self.execute_trades(predictions)
            
            # Log session summary
            portfolio_metrics = self.portfolio_manager.get_performance_metrics()
            logger.info(f"Trading session completed:")
            logger.info(f"  - Predictions: {len(predictions)}")
            logger.info(f"  - Executed trades: {len(executed_trades)}")
            logger.info(f"  - Portfolio value: {format_currency(portfolio_metrics.get('current_value', 0))}")
            logger.info(f"  - Open positions: {portfolio_metrics.get('num_open_positions', 0)}")
            logger.info(f"  - Total return: {portfolio_metrics.get('total_return_pct', 0):.2f}%")
            
            # Log memory usage
            mem_usage = memory_usage()
            logger.info(f"Memory usage: {mem_usage['rss_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
            self.error_count += 1
            self.last_error_time = datetime.now()
    
    def start_automated_trading(self, interval_minutes: int = 60):
        """
        Start automated trading with specified interval.
        
        Args:
            interval_minutes: Minutes between trading sessions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() or load_model() first.")
        
        logger.info(f"Starting automated trading with {interval_minutes} minute intervals")
        
        # Schedule trading sessions
        schedule.every(interval_minutes).minutes.do(self.run_trading_session)
        
        self.is_running = True
        
        # Run in separate thread
        def trading_loop():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        
        logger.info("Automated trading started")
    
    def stop_automated_trading(self):
        """Stop automated trading."""
        self.is_running = False
        schedule.clear()
        logger.info("Automated trading stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status and metrics."""
        portfolio_metrics = self.portfolio_manager.get_performance_metrics()
        
        status = {
            'is_running': self.is_running,
            'is_trained': self.is_trained,
            'last_prediction_time': self.last_prediction_time,
            'error_count': self.error_count,
            'last_error_time': self.last_error_time,
            'portfolio_metrics': portfolio_metrics,
            'memory_usage': memory_usage(),
            'open_positions': list(self.portfolio_manager.positions.keys()),
            'recent_predictions': self.prediction_history[-5:] if self.prediction_history else []
        }
        
        return status
    
    def backtest(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run backtesting on historical data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of symbols to backtest
            
        Returns:
            Backtesting results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() or load_model() first.")
        
        if symbols is None:
            symbols = self.config.data.symbols
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize backtest portfolio
        backtest_portfolio = PortfolioManager(self.config.trading.initial_capital)
        
        # Get historical data for the period
        all_data = {}
        for symbol in symbols:
            data = self.data_processor.fetch_data(symbol, period="max", interval="1d")
            if data is not None:
                # Filter for backtest period
                mask = (data.index >= start_date) & (data.index <= end_date)
                all_data[symbol] = data.loc[mask]
        
        if not all_data:
            raise ValueError("No historical data available for backtesting")
        
        # Run backtest day by day
        all_dates = sorted(set().union(*[data.index for data in all_data.values()]))
        
        for date in all_dates:
            # Get data up to current date for each symbol
            predictions = {}
            
            for symbol in symbols:
                if symbol not in all_data or date not in all_data[symbol].index:
                    continue
                
                # Get data up to current date
                historical_data = all_data[symbol].loc[:date]
                
                if len(historical_data) < self.config.model.sequence_length:
                    continue
                
                try:
                    # Calculate indicators
                    data_with_indicators = self.data_processor.calculate_technical_indicators(historical_data)
                    
                    # Create features
                    features, _ = self.data_processor.create_features_and_targets(data_with_indicators)
                    
                    if len(features) == 0:
                        continue
                    
                    # Scale and predict
                    scaled_features, _ = self.data_processor.scale_features(features, self.feature_scaler)
                    latest_features = scaled_features.tail(1)
                    
                    prediction, probabilities = self.model_builder.predict(
                        latest_features,
                        return_probabilities=True
                    )
                    
                    confidence = np.max(probabilities[0]) if len(probabilities.shape) > 1 else max(probabilities[0], 1 - probabilities[0])
                    current_price = historical_data['close'].iloc[-1]
                    
                    predictions[symbol] = {
                        'signal': int(prediction[0]),
                        'confidence': float(confidence),
                        'current_price': float(current_price),
                        'timestamp': date
                    }
                    
                except Exception as e:
                    logger.debug(f"Error processing {symbol} on {date}: {e}")
                    continue
            
            # Execute backtest trades (simplified version)
            for symbol, pred in predictions.items():
                signal = pred['signal']
                confidence = pred['confidence']
                price = pred['current_price']
                
                if confidence >= self.config.trading.min_confidence:
                    if signal == 2 and len(backtest_portfolio.positions) < self.config.trading.max_positions:
                        # Buy signal
                        portfolio_value = backtest_portfolio.current_capital + sum(
                            pos['entry_price'] * pos['quantity'] for pos in backtest_portfolio.positions.values()
                        )
                        position_size = self.risk_manager.calculate_position_size(portfolio_value, price, confidence)
                        
                        if position_size > 0:
                            backtest_portfolio.add_position(
                                symbol, 'buy', price, position_size, confidence, date
                            )
                    
                    elif signal == 0 and symbol in backtest_portfolio.positions:
                        # Sell signal
                        position = backtest_portfolio.positions[symbol]
                        backtest_portfolio.add_position(
                            symbol, 'sell', price, position['quantity'], confidence, date
                        )
        
        # Calculate backtest results
        final_metrics = backtest_portfolio.get_performance_metrics()
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'symbols': symbols,
            'initial_capital': self.config.trading.initial_capital,
            'final_value': final_metrics.get('current_value', 0),
            'total_return_pct': final_metrics.get('total_return_pct', 0),
            'max_drawdown_pct': final_metrics.get('max_drawdown_pct', 0),
            'total_trades': final_metrics.get('total_trades', 0),
            'win_rate_pct': final_metrics.get('win_rate_pct', 0),
            'equity_curve': backtest_portfolio.equity_curve,
            'trade_history': backtest_portfolio.trade_history
        }
        
        logger.info(f"Backtest completed: {final_metrics.get('total_return_pct', 0):.2f}% return")
        
        return results


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified CNN-LSTM Trading Bot')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--trade', action='store_true', help='Start automated trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--symbols', nargs='+', help='List of symbols to use')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize bot
    config_path = Path(args.config) if args.config else None
    bot = UnifiedTradingBot(config_path)
    
    try:
        if args.train:
            logger.info("Training model...")
            metrics = bot.train_model(symbols=args.symbols)
            logger.info(f"Training completed: {metrics}")
        
        elif args.predict:
            logger.info("Generating predictions...")
            predictions = bot.predict_signals(symbols=args.symbols)
            logger.info(f"Predictions: {predictions}")
        
        elif args.trade:
            logger.info("Starting automated trading...")
            bot.start_automated_trading()
            
            # Keep running until keyboard interrupt
            try:
                while True:
                    time.sleep(60)
                    status = bot.get_status()
                    logger.info(f"Status: {status['portfolio_metrics']}")
            except KeyboardInterrupt:
                logger.info("Stopping automated trading...")
                bot.stop_automated_trading()
        
        elif args.backtest:
            logger.info("Running backtest...")
            results = bot.backtest("2023-01-01", "2023-12-31", symbols=args.symbols)
            logger.info(f"Backtest results: {results}")
        
        else:
            logger.info("No action specified. Use --help for options.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()