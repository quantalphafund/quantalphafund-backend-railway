"""
Base Data Source Abstract Class
Foundation for all market data connectors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from enum import Enum
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)

class DataFrequency(Enum):
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    THIRTY_MINUTE = "30min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"
    YEARLY = "1y"

@dataclass
class SecurityIdentifier:
    """Universal security identifier supporting multiple markets"""
    symbol: str
    exchange: Optional[str] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None
    sedol: Optional[str] = None
    figi: Optional[str] = None
    market: Optional[str] = None
    asset_class: Optional[str] = None

    def to_yahoo_symbol(self) -> str:
        """Convert to Yahoo Finance symbol format"""
        market_suffixes = {
            "singapore": ".SI",
            "india_nse": ".NS",
            "india_bse": ".BO",
            "uae_dfm": ".DFM",
            "uae_adx": ".ADX",
            "usa": "",
        }
        suffix = market_suffixes.get(self.market, "")
        return f"{self.symbol}{suffix}"

    def to_polygon_symbol(self) -> str:
        """Convert to Polygon.io symbol format"""
        return self.symbol  # Polygon uses standard US tickers

    def to_alpha_vantage_symbol(self) -> str:
        """Convert to Alpha Vantage symbol format"""
        if self.market == "india_nse":
            return f"{self.symbol}.NSE"
        elif self.market == "india_bse":
            return f"{self.symbol}.BSE"
        return self.symbol

@dataclass
class OHLCV:
    """Standard OHLCV data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    vwap: Optional[float] = None
    trades: Optional[int] = None

@dataclass
class FundamentalData:
    """Comprehensive fundamental data structure"""
    symbol: str
    timestamp: datetime
    period: str  # Q1, Q2, Q3, Q4, FY
    fiscal_year: int

    # Income Statement
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    ebit: Optional[float] = None
    interest_expense: Optional[float] = None
    pretax_income: Optional[float] = None
    income_tax: Optional[float] = None
    net_income: Optional[float] = None
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None
    shares_outstanding: Optional[float] = None
    shares_diluted: Optional[float] = None

    # Balance Sheet
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    retained_earnings: Optional[float] = None
    book_value: Optional[float] = None
    tangible_book_value: Optional[float] = None

    # Cash Flow
    operating_cash_flow: Optional[float] = None
    capital_expenditures: Optional[float] = None
    free_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    dividends_paid: Optional[float] = None
    share_repurchases: Optional[float] = None
    debt_repayment: Optional[float] = None

    # Calculated Metrics (will be computed by analysis engine)
    calculated_metrics: Optional[Dict[str, float]] = None

@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    volume: int
    change: float
    change_percent: float

class BaseDataSource(ABC):
    """Abstract base class for all data sources"""

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 60):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self._request_times: List[datetime] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def _rate_limit_check(self):
        """Implement rate limiting"""
        now = datetime.now()
        minute_ago = now.timestamp() - 60

        # Remove old request times
        self._request_times = [
            t for t in self._request_times
            if t.timestamp() > minute_ago
        ]

        # Check if we're at the limit
        if len(self._request_times) >= self.rate_limit:
            sleep_time = 60 - (now.timestamp() - self._request_times[0].timestamp())
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self._request_times.append(now)

    async def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request with rate limiting and error handling"""
        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            async with self._session.request(
                method, url, params=params, headers=headers, json=json_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited, retrying after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(url, method, params, headers, json_data)
                else:
                    self.logger.error(f"Request failed with status {response.status}")
                    raise Exception(f"API request failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            raise

    @abstractmethod
    async def get_price_history(
        self,
        security: SecurityIdentifier,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        pass

    @abstractmethod
    async def get_fundamentals(
        self,
        security: SecurityIdentifier,
        periods: int = 12  # Last 12 quarters/periods
    ) -> List[FundamentalData]:
        """Fetch fundamental financial data"""
        pass

    @abstractmethod
    async def get_real_time_quote(
        self,
        security: SecurityIdentifier
    ) -> Quote:
        """Fetch real-time quote"""
        pass

    @abstractmethod
    async def search_securities(
        self,
        query: str,
        asset_class: Optional[str] = None,
        market: Optional[str] = None
    ) -> List[Dict]:
        """Search for securities by name/symbol"""
        pass

    @abstractmethod
    def get_supported_markets(self) -> List[str]:
        """Return list of supported markets"""
        pass

    @abstractmethod
    def get_supported_asset_classes(self) -> List[str]:
        """Return list of supported asset classes"""
        pass

class DataAggregator:
    """
    Aggregates data from multiple sources with intelligent fallback
    Inspired by Medallion's obsessive data quality approach
    """

    def __init__(self, sources: List[BaseDataSource]):
        self.sources = sources
        self.source_priority: Dict[str, List[BaseDataSource]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_source_priority(self, data_type: str, priority: List[BaseDataSource]):
        """Set priority order for data sources by data type"""
        self.source_priority[data_type] = priority

    async def get_price_history(
        self,
        security: SecurityIdentifier,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """
        Get price history with automatic fallback between sources
        Validates and reconciles data from multiple sources when available
        """
        sources = self.source_priority.get("price", self.sources)
        errors = []

        for source in sources:
            try:
                data = await source.get_price_history(
                    security, start_date, end_date, frequency
                )
                if data is not None and len(data) > 0:
                    return self._validate_price_data(data)
            except Exception as e:
                errors.append(f"{source.__class__.__name__}: {e}")
                continue

        raise Exception(f"All data sources failed: {errors}")

    def _validate_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean price data
        Applies Medallion-style data quality standards
        """
        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Validate OHLC relationships
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        if invalid_rows.any():
            self.logger.warning(f"Found {invalid_rows.sum()} invalid OHLC rows, correcting...")
            # Fix invalid rows by using close price for all OHLC
            df.loc[invalid_rows, ['open', 'high', 'low']] = df.loc[invalid_rows, 'close'].values[:, None]

        # Handle negative volumes
        df.loc[df['volume'] < 0, 'volume'] = 0

        # Forward fill missing values (limited)
        df = df.fillna(method='ffill', limit=3)

        # Drop remaining NaN rows
        df = df.dropna()

        return df

    async def get_fundamentals(
        self,
        security: SecurityIdentifier,
        periods: int = 12
    ) -> List[FundamentalData]:
        """Get fundamental data with fallback"""
        sources = self.source_priority.get("fundamentals", self.sources)

        for source in sources:
            try:
                data = await source.get_fundamentals(security, periods)
                if data and len(data) > 0:
                    return data
            except Exception as e:
                self.logger.warning(f"Source {source.__class__.__name__} failed: {e}")
                continue

        raise Exception("All fundamental data sources failed")

    async def get_comprehensive_data(
        self,
        security: SecurityIdentifier,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Get all available data for a security
        Returns a comprehensive data package
        """
        results = {}

        # Gather all data concurrently
        tasks = [
            self.get_price_history(security, start_date, end_date),
            self.get_fundamentals(security),
        ]

        price_data, fundamental_data = await asyncio.gather(*tasks, return_exceptions=True)

        if not isinstance(price_data, Exception):
            results['price_history'] = price_data
        else:
            self.logger.error(f"Price data error: {price_data}")

        if not isinstance(fundamental_data, Exception):
            results['fundamentals'] = fundamental_data
        else:
            self.logger.error(f"Fundamental data error: {fundamental_data}")

        return results
