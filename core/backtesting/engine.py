"""
Professional Backtesting Engine
Event-driven backtesting system with realistic execution simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: PositionSide
    entry_time: datetime
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    commission_paid: float = 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    def update_price(self, price: float):
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

@dataclass
class Trade:
    """Completed trade record"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    commission: float
    slippage: float
    holding_period: int  # days
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 1_000_000
    commission_rate: float = 0.001  # 10 bps
    slippage_rate: float = 0.0005  # 5 bps
    margin_requirement: float = 0.5  # 50% margin
    max_leverage: float = 2.0
    risk_free_rate: float = 0.05
    trading_days_per_year: int = 252
    allow_shorting: bool = True
    fractional_shares: bool = True
    reinvest_dividends: bool = True

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    # Performance
    total_return: float
    annualized_return: float
    benchmark_return: float
    excess_return: float

    # Risk
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int

    # Risk-Adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_return: float
    avg_holding_period: float

    # Costs
    total_commission: float
    total_slippage: float

    # Time Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    returns: pd.Series
    positions_history: pd.DataFrame
    trades: List[Trade]

    # Metadata
    start_date: datetime
    end_date: datetime
    config: BacktestConfig

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }


class ExecutionSimulator:
    """
    Realistic order execution simulation
    Models slippage, market impact, and partial fills
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def execute_order(
        self,
        order: Order,
        current_price: float,
        volume: float,
        timestamp: datetime
    ) -> Order:
        """
        Simulate order execution with realistic slippage

        Args:
            order: Order to execute
            current_price: Current market price
            volume: Current market volume
            timestamp: Current timestamp

        Returns:
            Updated order with fill information
        """
        if order.status == OrderStatus.FILLED:
            return order

        # Check if order can be filled
        can_fill = self._check_fill_conditions(order, current_price)

        if not can_fill:
            return order

        # Calculate slippage based on order size and volume
        slippage = self._calculate_slippage(order, current_price, volume)

        # Calculate fill price
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + slippage)
        else:
            fill_price = current_price * (1 - slippage)

        # Calculate commission
        commission = abs(order.quantity * fill_price * self.config.commission_rate)

        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.slippage = slippage * current_price * order.quantity
        order.commission = commission
        order.status = OrderStatus.FILLED
        order.filled_at = timestamp

        return order

    def _check_fill_conditions(self, order: Order, current_price: float) -> bool:
        """Check if order conditions are met for fill"""
        if order.order_type == OrderType.MARKET:
            return True

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:
                return current_price >= order.price

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price

        return False

    def _calculate_slippage(
        self,
        order: Order,
        current_price: float,
        volume: float
    ) -> float:
        """
        Calculate realistic slippage based on:
        - Order size relative to volume
        - Market impact model
        """
        base_slippage = self.config.slippage_rate

        # Market impact: larger orders have more slippage
        order_value = abs(order.quantity * current_price)
        avg_volume_value = volume * current_price / 10  # Assume 10% participation

        if avg_volume_value > 0:
            participation_rate = order_value / avg_volume_value
            # Square root market impact model
            market_impact = 0.001 * np.sqrt(participation_rate)
        else:
            market_impact = 0

        return base_slippage + market_impact


class Portfolio:
    """
    Portfolio management with position tracking
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.order_history: List[Order] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self._trade_counter = 0

    @property
    def equity(self) -> float:
        """Total portfolio equity"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def gross_exposure(self) -> float:
        """Gross exposure (sum of absolute position values)"""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Net exposure"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def leverage(self) -> float:
        """Current leverage ratio"""
        if self.equity > 0:
            return self.gross_exposure / self.equity
        return 0

    def process_order(self, order: Order) -> bool:
        """
        Process a filled order and update positions

        Returns:
            True if order was processed successfully
        """
        if order.status != OrderStatus.FILLED:
            return False

        symbol = order.symbol
        self.order_history.append(order)

        # Deduct commission from cash
        self.cash -= order.commission

        if order.side == OrderSide.BUY:
            return self._process_buy(order)
        else:
            return self._process_sell(order)

    def _process_buy(self, order: Order) -> bool:
        """Process buy order"""
        cost = order.filled_quantity * order.filled_price

        # Check if we have enough cash/margin
        required_cash = cost * self.config.margin_requirement
        if self.cash < required_cash:
            logger.warning(f"Insufficient cash for order: {order.id}")
            return False

        self.cash -= cost

        if order.symbol in self.positions:
            # Add to existing position
            pos = self.positions[order.symbol]
            total_cost = pos.quantity * pos.entry_price + cost
            total_quantity = pos.quantity + order.filled_quantity
            pos.entry_price = total_cost / total_quantity
            pos.quantity = total_quantity
            pos.commission_paid += order.commission
        else:
            # Create new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                side=PositionSide.LONG,
                entry_time=order.filled_at,
                commission_paid=order.commission
            )

        return True

    def _process_sell(self, order: Order) -> bool:
        """Process sell order"""
        symbol = order.symbol
        proceeds = order.filled_quantity * order.filled_price

        if symbol in self.positions:
            pos = self.positions[symbol]

            if pos.quantity >= order.filled_quantity:
                # Reduce or close position
                pnl = (order.filled_price - pos.entry_price) * order.filled_quantity
                pos.realized_pnl += pnl
                pos.quantity -= order.filled_quantity

                self.cash += proceeds

                # Record trade if position closed
                if pos.quantity == 0:
                    self._record_trade(pos, order)
                    del self.positions[symbol]

                return True

        # Short selling
        if self.config.allow_shorting:
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-order.filled_quantity,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    side=PositionSide.SHORT,
                    entry_time=order.filled_at,
                    commission_paid=order.commission
                )
                self.cash += proceeds
                return True

        return False

    def _record_trade(self, position: Position, exit_order: Order):
        """Record completed trade"""
        self._trade_counter += 1

        holding_period = (exit_order.filled_at - position.entry_time).days

        pnl = position.realized_pnl
        return_pct = (exit_order.filled_price - position.entry_price) / position.entry_price

        if position.side == PositionSide.SHORT:
            return_pct = -return_pct

        trade = Trade(
            id=f"trade_{self._trade_counter}",
            symbol=position.symbol,
            side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
            quantity=abs(position.quantity) if position.quantity != 0 else exit_order.filled_quantity,
            entry_price=position.entry_price,
            exit_price=exit_order.filled_price,
            entry_time=position.entry_time,
            exit_time=exit_order.filled_at,
            pnl=pnl,
            return_pct=return_pct,
            commission=position.commission_paid + exit_order.commission,
            slippage=exit_order.slippage,
            holding_period=holding_period
        )

        self.closed_trades.append(trade)

    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        """Update position prices and record equity"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

        self.equity_history.append((timestamp, self.equity))


class Strategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Strategy.{name}")

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime,
        portfolio: Portfolio
    ) -> List[Order]:
        """
        Generate trading signals

        Args:
            data: Historical data for each symbol
            timestamp: Current timestamp
            portfolio: Current portfolio state

        Returns:
            List of orders to execute
        """
        pass

    @abstractmethod
    def on_trade(self, trade: Trade):
        """Called when a trade is completed"""
        pass


class BacktestEngine:
    """
    Event-driven backtesting engine
    Simulates realistic trading with proper execution
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.executor = ExecutionSimulator(self.config)
        self.portfolio = Portfolio(self.config)
        self.strategy: Optional[Strategy] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.Series] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._order_counter = 0

    def set_strategy(self, strategy: Strategy):
        """Set the trading strategy"""
        self.strategy = strategy

    def set_data(
        self,
        data: Dict[str, pd.DataFrame],
        benchmark: Optional[pd.Series] = None
    ):
        """
        Set historical data for backtesting

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            benchmark: Optional benchmark returns series
        """
        self.data = data
        self.benchmark_data = benchmark

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run the backtest

        Args:
            start_date: Start date (default: first date in data)
            end_date: End date (default: last date in data)

        Returns:
            BacktestResult with all metrics
        """
        if not self.strategy:
            raise ValueError("No strategy set")

        if not self.data:
            raise ValueError("No data set")

        # Get common date range
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        self.logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
        self.logger.info(f"Strategy: {self.strategy.name}")

        # Run simulation
        for timestamp in all_dates:
            self._process_day(timestamp)

        # Calculate results
        return self._calculate_results(all_dates[0], all_dates[-1])

    def _process_day(self, timestamp: datetime):
        """Process a single day"""
        # Get current prices
        current_prices = {}
        current_volumes = {}

        for symbol, df in self.data.items():
            if timestamp in df.index:
                current_prices[symbol] = df.loc[timestamp, 'close']
                current_volumes[symbol] = df.loc[timestamp, 'volume']

        # Update portfolio with current prices
        self.portfolio.update_prices(current_prices, timestamp)

        # Get data up to current timestamp for strategy
        historical_data = {}
        for symbol, df in self.data.items():
            historical_data[symbol] = df[df.index <= timestamp]

        # Generate signals
        orders = self.strategy.generate_signals(
            historical_data,
            timestamp,
            self.portfolio
        )

        # Execute orders
        for order in orders:
            self._order_counter += 1
            order.id = f"order_{self._order_counter}"

            if order.symbol in current_prices:
                # Execute order
                filled_order = self.executor.execute_order(
                    order,
                    current_prices[order.symbol],
                    current_volumes.get(order.symbol, 0),
                    timestamp
                )

                # Process filled order
                if filled_order.status == OrderStatus.FILLED:
                    self.portfolio.process_order(filled_order)

    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate backtest results"""
        # Build equity curve
        equity_series = pd.Series(
            dict(self.portfolio.equity_history)
        )

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max

        # Performance metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        n_years = len(returns) / self.config.trading_days_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        volatility = returns.std() * np.sqrt(self.config.trading_days_per_year)
        max_drawdown = drawdown.min()

        # Calculate max drawdown duration
        dd_duration = self._calculate_dd_duration(drawdown)

        # Risk-adjusted metrics
        excess_returns = returns - self.config.risk_free_rate / self.config.trading_days_per_year
        sharpe_ratio = np.sqrt(self.config.trading_days_per_year) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

        downside_returns = returns[returns < 0]
        sortino_ratio = (
            np.sqrt(self.config.trading_days_per_year) *
            excess_returns.mean() / downside_returns.std()
        ) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Benchmark comparison
        benchmark_return = 0
        information_ratio = 0
        if self.benchmark_data is not None:
            aligned_benchmark = self.benchmark_data.reindex(returns.index).fillna(0)
            benchmark_return = (1 + aligned_benchmark).prod() - 1
            active_returns = returns - aligned_benchmark
            tracking_error = active_returns.std() * np.sqrt(self.config.trading_days_per_year)
            information_ratio = (annualized_return - benchmark_return) / tracking_error if tracking_error > 0 else 0

        # Trade statistics
        trades = self.portfolio.closed_trades
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else np.inf

        avg_trade_return = np.mean([t.return_pct for t in trades]) if trades else 0
        avg_holding_period = np.mean([t.holding_period for t in trades]) if trades else 0

        # Costs
        total_commission = sum(o.commission for o in self.portfolio.order_history)
        total_slippage = sum(o.slippage for o in self.portfolio.order_history)

        # Build positions history DataFrame
        positions_history = pd.DataFrame([
            {
                "timestamp": ts,
                "equity": eq,
                "cash": self.portfolio.cash,
                "positions_value": eq - self.portfolio.cash
            }
            for ts, eq in self.portfolio.equity_history
        ])

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            excess_return=annualized_return - benchmark_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            avg_holding_period=avg_holding_period,
            total_commission=total_commission,
            total_slippage=total_slippage,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            returns=returns,
            positions_history=positions_history,
            trades=trades,
            start_date=start_date,
            end_date=end_date,
            config=self.config
        )

    def _calculate_dd_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_dd = drawdown < 0
        durations = []
        current = 0

        for is_dd in in_dd:
            if is_dd:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                current = 0

        if current > 0:
            durations.append(current)

        return max(durations) if durations else 0


# Example Strategy Implementation
class MomentumStrategy(Strategy):
    """
    Simple momentum strategy for demonstration
    """

    def __init__(
        self,
        lookback: int = 20,
        top_n: int = 5,
        rebalance_freq: int = 20
    ):
        super().__init__("Momentum")
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self._days_since_rebalance = 0

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime,
        portfolio: Portfolio
    ) -> List[Order]:
        """Generate momentum signals"""
        self._days_since_rebalance += 1

        # Only rebalance at specified frequency
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        self._days_since_rebalance = 0

        # Calculate momentum for each symbol
        momentum_scores = {}
        for symbol, df in data.items():
            if len(df) >= self.lookback:
                momentum = df['close'].iloc[-1] / df['close'].iloc[-self.lookback] - 1
                momentum_scores[symbol] = momentum

        # Sort by momentum
        sorted_symbols = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top N
        top_symbols = [s[0] for s in sorted_symbols[:self.top_n]]

        orders = []

        # Close positions not in top N
        for symbol in list(portfolio.positions.keys()):
            if symbol not in top_symbols:
                pos = portfolio.positions[symbol]
                orders.append(Order(
                    id="",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity
                ))

        # Open positions in top N
        target_weight = 1.0 / self.top_n
        target_value = portfolio.equity * target_weight

        for symbol in top_symbols:
            if symbol not in portfolio.positions:
                if symbol in data and len(data[symbol]) > 0:
                    price = data[symbol]['close'].iloc[-1]
                    quantity = target_value / price
                    orders.append(Order(
                        id="",
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity
                    ))

        return orders

    def on_trade(self, trade: Trade):
        """Log completed trade"""
        self.logger.info(
            f"Trade: {trade.symbol} {trade.side.value} "
            f"PnL: ${trade.pnl:.2f} ({trade.return_pct:.2%})"
        )
