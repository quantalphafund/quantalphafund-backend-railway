"""
Real-Time Data Streaming Engine
WebSocket-based real-time market data with multiple providers
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TickType(Enum):
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"

@dataclass
class Tick:
    """Real-time tick data"""
    symbol: str
    timestamp: datetime
    tick_type: TickType
    price: float
    size: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None
    conditions: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "type": self.tick_type.value,
            "price": self.price,
            "size": self.size,
            "bid": self.bid,
            "ask": self.ask,
            "volume": self.volume,
            "vwap": self.vwap
        }

@dataclass
class RealTimeBar:
    """Real-time OHLCV bar"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trades": self.trades
        }

class RealTimeDataProvider(ABC):
    """Abstract base for real-time data providers"""

    def __init__(self):
        self.subscriptions: Set[str] = set()
        self.callbacks: List[Callable[[Tick], None]] = []
        self.bar_callbacks: List[Callable[[RealTimeBar], None]] = []
        self.is_connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def connect(self):
        """Connect to data provider"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from data provider"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        pass

    def add_tick_callback(self, callback: Callable[[Tick], None]):
        """Register callback for tick data"""
        self.callbacks.append(callback)

    def add_bar_callback(self, callback: Callable[[RealTimeBar], None]):
        """Register callback for bar data"""
        self.bar_callbacks.append(callback)

    async def _emit_tick(self, tick: Tick):
        """Emit tick to all callbacks"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick)
                else:
                    callback(tick)
            except Exception as e:
                self.logger.error(f"Tick callback error: {e}")

    async def _emit_bar(self, bar: RealTimeBar):
        """Emit bar to all callbacks"""
        for callback in self.bar_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(bar)
                else:
                    callback(bar)
            except Exception as e:
                self.logger.error(f"Bar callback error: {e}")


class PolygonRealTimeProvider(RealTimeDataProvider):
    """
    Polygon.io WebSocket real-time data provider
    Provides trades, quotes, and minute bars for US equities
    """

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self._ws = None
        self._session = None
        self._receive_task = None

    async def connect(self):
        """Connect to Polygon WebSocket"""
        if self.is_connected:
            return

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self.WS_URL)

        # Authenticate
        auth_msg = {"action": "auth", "params": self.api_key}
        await self._ws.send_json(auth_msg)

        # Wait for auth response
        msg = await self._ws.receive_json()
        if msg.get("status") == "auth_success":
            self.is_connected = True
            self.logger.info("Connected to Polygon WebSocket")

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
        else:
            raise ConnectionError(f"Polygon auth failed: {msg}")

    async def disconnect(self):
        """Disconnect from Polygon WebSocket"""
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self.is_connected = False
        self.logger.info("Disconnected from Polygon WebSocket")

    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols for trades, quotes, and bars"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Polygon")

        for symbol in symbols:
            self.subscriptions.add(symbol)

        # Subscribe to trades (T), quotes (Q), and minute bars (AM)
        channels = []
        for symbol in symbols:
            channels.extend([f"T.{symbol}", f"Q.{symbol}", f"AM.{symbol}"])

        sub_msg = {"action": "subscribe", "params": ",".join(channels)}
        await self._ws.send_json(sub_msg)
        self.logger.info(f"Subscribed to: {symbols}")

    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if not self.is_connected:
            return

        for symbol in symbols:
            self.subscriptions.discard(symbol)

        channels = []
        for symbol in symbols:
            channels.extend([f"T.{symbol}", f"Q.{symbol}", f"AM.{symbol}"])

        unsub_msg = {"action": "unsubscribe", "params": ",".join(channels)}
        await self._ws.send_json(unsub_msg)

    async def _receive_loop(self):
        """Receive and process WebSocket messages"""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._process_messages(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {msg}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Receive loop error: {e}")

    async def _process_messages(self, messages: List[Dict]):
        """Process incoming messages"""
        for msg in messages:
            ev = msg.get("ev")

            if ev == "T":  # Trade
                tick = Tick(
                    symbol=msg["sym"],
                    timestamp=datetime.fromtimestamp(msg["t"] / 1000),
                    tick_type=TickType.TRADE,
                    price=msg["p"],
                    size=msg["s"],
                    conditions=msg.get("c", [])
                )
                await self._emit_tick(tick)

            elif ev == "Q":  # Quote
                tick = Tick(
                    symbol=msg["sym"],
                    timestamp=datetime.fromtimestamp(msg["t"] / 1000),
                    tick_type=TickType.QUOTE,
                    price=(msg["bp"] + msg["ap"]) / 2,  # Midpoint
                    size=0,
                    bid=msg["bp"],
                    ask=msg["ap"],
                    bid_size=msg["bs"],
                    ask_size=msg["as"]
                )
                await self._emit_tick(tick)

            elif ev == "AM":  # Minute bar
                bar = RealTimeBar(
                    symbol=msg["sym"],
                    timestamp=datetime.fromtimestamp(msg["s"] / 1000),
                    open=msg["o"],
                    high=msg["h"],
                    low=msg["l"],
                    close=msg["c"],
                    volume=msg["v"],
                    vwap=msg.get("vw"),
                    trades=msg.get("n")
                )
                await self._emit_bar(bar)


class SimulatedRealTimeProvider(RealTimeDataProvider):
    """
    Simulated real-time data provider for testing/demo
    Generates realistic tick data based on historical patterns
    """

    def __init__(self, base_prices: Optional[Dict[str, float]] = None):
        super().__init__()
        self.base_prices = base_prices or {
            "AAPL": 185.0,
            "GOOGL": 140.0,
            "MSFT": 375.0,
            "AMZN": 155.0,
            "NVDA": 480.0,
            "META": 350.0,
            "TSLA": 250.0,
            "JPM": 170.0,
            "V": 260.0,
            "JNJ": 155.0,
        }
        self.current_prices: Dict[str, float] = {}
        self._running = False
        self._task = None

    async def connect(self):
        """Start simulation"""
        self.is_connected = True
        self.current_prices = self.base_prices.copy()
        self.logger.info("Simulated real-time provider connected")

    async def disconnect(self):
        """Stop simulation"""
        self._running = False
        if self._task:
            self._task.cancel()
        self.is_connected = False
        self.logger.info("Simulated real-time provider disconnected")

    async def subscribe(self, symbols: List[str]):
        """Subscribe and start generating data"""
        for symbol in symbols:
            self.subscriptions.add(symbol)
            if symbol not in self.current_prices:
                self.current_prices[symbol] = 100.0  # Default price

        # Start simulation if not running
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._simulation_loop())

        self.logger.info(f"Subscribed to simulated data: {symbols}")

    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscriptions.discard(symbol)

        if not self.subscriptions:
            self._running = False

    async def _simulation_loop(self):
        """Generate simulated ticks"""
        import random

        try:
            while self._running:
                for symbol in list(self.subscriptions):
                    if symbol not in self.current_prices:
                        continue

                    # Random walk with mean reversion
                    current = self.current_prices[symbol]
                    base = self.base_prices.get(symbol, 100.0)

                    # Mean reversion factor
                    reversion = (base - current) * 0.001

                    # Random component
                    volatility = current * 0.0001  # 0.01% tick volatility
                    change = random.gauss(reversion, volatility)

                    new_price = current + change
                    self.current_prices[symbol] = new_price

                    # Generate trade tick
                    tick = Tick(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        tick_type=TickType.TRADE,
                        price=round(new_price, 2),
                        size=random.randint(100, 10000),
                        volume=random.randint(1000000, 50000000)
                    )
                    await self._emit_tick(tick)

                    # Generate quote tick
                    spread = new_price * 0.0001  # 1 bp spread
                    quote = Tick(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        tick_type=TickType.QUOTE,
                        price=new_price,
                        size=0,
                        bid=round(new_price - spread/2, 2),
                        ask=round(new_price + spread/2, 2),
                        bid_size=random.randint(100, 5000),
                        ask_size=random.randint(100, 5000)
                    )
                    await self._emit_tick(quote)

                # Generate minute bars
                if datetime.now().second == 0:
                    for symbol in list(self.subscriptions):
                        price = self.current_prices.get(symbol, 100)
                        bar = RealTimeBar(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=price * (1 + random.gauss(0, 0.001)),
                            high=price * (1 + abs(random.gauss(0, 0.002))),
                            low=price * (1 - abs(random.gauss(0, 0.002))),
                            close=price,
                            volume=random.randint(100000, 1000000),
                            trades=random.randint(100, 1000)
                        )
                        await self._emit_bar(bar)

                await asyncio.sleep(0.1)  # 10 ticks per second

        except asyncio.CancelledError:
            pass


class RealTimeDataManager:
    """
    Manages real-time data subscriptions and aggregation
    Provides unified interface for multiple providers
    """

    def __init__(self):
        self.providers: Dict[str, RealTimeDataProvider] = {}
        self.tick_buffer: Dict[str, List[Tick]] = {}
        self.bar_buffer: Dict[str, List[RealTimeBar]] = {}
        self.buffer_size = 1000  # Keep last 1000 ticks per symbol
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_provider(self, name: str, provider: RealTimeDataProvider):
        """Add a data provider"""
        provider.add_tick_callback(self._on_tick)
        provider.add_bar_callback(self._on_bar)
        self.providers[name] = provider
        self.logger.info(f"Added provider: {name}")

    async def connect_all(self):
        """Connect all providers"""
        for name, provider in self.providers.items():
            try:
                await provider.connect()
            except Exception as e:
                self.logger.error(f"Failed to connect {name}: {e}")

    async def disconnect_all(self):
        """Disconnect all providers"""
        for name, provider in self.providers.items():
            try:
                await provider.disconnect()
            except Exception as e:
                self.logger.error(f"Failed to disconnect {name}: {e}")

    async def subscribe(self, symbols: List[str], provider_name: Optional[str] = None):
        """Subscribe to symbols on specified or all providers"""
        if provider_name:
            if provider_name in self.providers:
                await self.providers[provider_name].subscribe(symbols)
        else:
            for provider in self.providers.values():
                await provider.subscribe(symbols)

        # Initialize buffers
        for symbol in symbols:
            if symbol not in self.tick_buffer:
                self.tick_buffer[symbol] = []
            if symbol not in self.bar_buffer:
                self.bar_buffer[symbol] = []

    async def _on_tick(self, tick: Tick):
        """Handle incoming tick"""
        symbol = tick.symbol

        # Buffer tick
        if symbol not in self.tick_buffer:
            self.tick_buffer[symbol] = []
        self.tick_buffer[symbol].append(tick)

        # Trim buffer
        if len(self.tick_buffer[symbol]) > self.buffer_size:
            self.tick_buffer[symbol] = self.tick_buffer[symbol][-self.buffer_size:]

        # Notify subscribers
        await self._notify_subscribers(symbol, "tick", tick)

    async def _on_bar(self, bar: RealTimeBar):
        """Handle incoming bar"""
        symbol = bar.symbol

        # Buffer bar
        if symbol not in self.bar_buffer:
            self.bar_buffer[symbol] = []
        self.bar_buffer[symbol].append(bar)

        # Trim buffer (keep more bars)
        if len(self.bar_buffer[symbol]) > self.buffer_size * 10:
            self.bar_buffer[symbol] = self.bar_buffer[symbol][-self.buffer_size * 10:]

        # Notify subscribers
        await self._notify_subscribers(symbol, "bar", bar)

    async def _notify_subscribers(self, symbol: str, data_type: str, data: Any):
        """Notify all subscribers of new data"""
        key = f"{symbol}:{data_type}"
        if key in self.subscribers:
            for callback in self.subscribers[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Subscriber callback error: {e}")

    def add_subscriber(
        self,
        symbol: str,
        data_type: str,
        callback: Callable
    ):
        """Add subscriber for symbol data"""
        key = f"{symbol}:{data_type}"
        if key not in self.subscribers:
            self.subscribers[key] = set()
        self.subscribers[key].add(callback)

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get latest tick for symbol"""
        if symbol in self.tick_buffer and self.tick_buffer[symbol]:
            return self.tick_buffer[symbol][-1]
        return None

    def get_latest_bar(self, symbol: str) -> Optional[RealTimeBar]:
        """Get latest bar for symbol"""
        if symbol in self.bar_buffer and self.bar_buffer[symbol]:
            return self.bar_buffer[symbol][-1]
        return None

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Tick]:
        """Get recent ticks for symbol"""
        if symbol in self.tick_buffer:
            return self.tick_buffer[symbol][-count:]
        return []

    def get_recent_bars(self, symbol: str, count: int = 60) -> List[RealTimeBar]:
        """Get recent bars for symbol"""
        if symbol in self.bar_buffer:
            return self.bar_buffer[symbol][-count:]
        return []
