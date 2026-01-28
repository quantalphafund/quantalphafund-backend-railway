"""
Yahoo Finance Data Source
Supports USA, Singapore, India, UAE markets
Free tier with comprehensive coverage
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import json
import re

from .base import (
    BaseDataSource, SecurityIdentifier, OHLCV, FundamentalData,
    Quote, DataFrequency
)

class YahooFinanceSource(BaseDataSource):
    """
    Yahoo Finance data connector
    Provides free access to global market data
    """

    BASE_URL = "https://query1.finance.yahoo.com"
    QUOTE_URL = "https://query2.finance.yahoo.com"

    MARKET_SUFFIXES = {
        "usa": "",
        "singapore": ".SI",
        "india_nse": ".NS",
        "india_bse": ".BO",
        "uae_dfm": ".AE",  # Dubai Financial Market
        "uae_adx": ".AE",  # Abu Dhabi Securities Exchange
    }

    FREQUENCY_MAP = {
        DataFrequency.MINUTE: "1m",
        DataFrequency.FIVE_MINUTE: "5m",
        DataFrequency.FIFTEEN_MINUTE: "15m",
        DataFrequency.THIRTY_MINUTE: "30m",
        DataFrequency.HOURLY: "1h",
        DataFrequency.DAILY: "1d",
        DataFrequency.WEEKLY: "1wk",
        DataFrequency.MONTHLY: "1mo",
        DataFrequency.QUARTERLY: "3mo",
    }

    def __init__(self, rate_limit: int = 100):
        super().__init__(api_key=None, rate_limit=rate_limit)
        self._crumb = None
        self._cookies = None

    def _get_symbol(self, security: SecurityIdentifier) -> str:
        """Convert security identifier to Yahoo symbol"""
        suffix = self.MARKET_SUFFIXES.get(security.market, "")
        return f"{security.symbol}{suffix}"

    async def _init_session(self):
        """Initialize session with cookies and crumb for authenticated requests"""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Get initial cookies
        url = "https://fc.yahoo.com"
        async with self._session.get(url) as response:
            pass  # Just to get cookies

        # Get crumb
        url = f"{self.BASE_URL}/v1/test/getcrumb"
        async with self._session.get(url) as response:
            if response.status == 200:
                self._crumb = await response.text()

    async def get_price_history(
        self,
        security: SecurityIdentifier,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance"""

        symbol = self._get_symbol(security)
        interval = self.FREQUENCY_MAP.get(frequency, "1d")

        # Convert dates to timestamps
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        url = f"{self.BASE_URL}/v8/finance/chart/{symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "includeAdjustedClose": "true",
            "events": "div,split",
        }

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance API error: {response.status}")

            data = await response.json()

        return self._parse_chart_data(data, symbol)

    def _parse_chart_data(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse Yahoo Finance chart response into DataFrame"""
        try:
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, unit='s'),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            })

            # Add adjusted close if available
            if "adjclose" in result["indicators"]:
                df["adjusted_close"] = result["indicators"]["adjclose"][0]["adjclose"]
            else:
                df["adjusted_close"] = df["close"]

            df["symbol"] = symbol
            df = df.set_index("timestamp")

            # Handle splits and dividends if present
            if "events" in result:
                events = result["events"]
                if "splits" in events:
                    df["split"] = self._map_events(timestamps, events["splits"])
                if "dividends" in events:
                    df["dividend"] = self._map_events(timestamps, events["dividends"])

            return df.dropna()

        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing Yahoo data: {e}")
            raise Exception(f"Failed to parse Yahoo Finance data: {e}")

    def _map_events(self, timestamps: List[int], events: Dict) -> pd.Series:
        """Map events (splits/dividends) to timestamp index"""
        event_map = {}
        for ts, event in events.items():
            event_map[int(ts)] = event.get("splitRatio", event.get("amount", 0))

        return pd.Series([event_map.get(ts, 0) for ts in timestamps])

    async def get_fundamentals(
        self,
        security: SecurityIdentifier,
        periods: int = 12
    ) -> List[FundamentalData]:
        """Fetch fundamental financial data"""

        symbol = self._get_symbol(security)

        # Fetch multiple modules for comprehensive fundamentals
        modules = [
            "incomeStatementHistory",
            "incomeStatementHistoryQuarterly",
            "balanceSheetHistory",
            "balanceSheetHistoryQuarterly",
            "cashflowStatementHistory",
            "cashflowStatementHistoryQuarterly",
            "defaultKeyStatistics",
            "financialData",
            "earnings",
        ]

        url = f"{self.QUOTE_URL}/v10/finance/quoteSummary/{symbol}"
        params = {"modules": ",".join(modules)}

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance fundamentals error: {response.status}")

            data = await response.json()

        return self._parse_fundamentals(data, symbol, periods)

    def _parse_fundamentals(
        self,
        data: Dict,
        symbol: str,
        periods: int
    ) -> List[FundamentalData]:
        """Parse Yahoo Finance fundamentals into FundamentalData objects"""
        try:
            result = data["quoteSummary"]["result"][0]
            fundamentals = []

            # Get quarterly income statements
            quarterly_income = result.get("incomeStatementHistoryQuarterly", {}).get(
                "incomeStatementHistory", []
            )
            quarterly_balance = result.get("balanceSheetHistoryQuarterly", {}).get(
                "balanceSheetStatements", []
            )
            quarterly_cashflow = result.get("cashflowStatementHistoryQuarterly", {}).get(
                "cashflowStatements", []
            )

            # Merge quarterly data
            for i, income in enumerate(quarterly_income[:periods]):
                end_date = datetime.fromtimestamp(income["endDate"]["raw"])
                fiscal_quarter = (end_date.month - 1) // 3 + 1

                balance = quarterly_balance[i] if i < len(quarterly_balance) else {}
                cashflow = quarterly_cashflow[i] if i < len(quarterly_cashflow) else {}

                fundamental = FundamentalData(
                    symbol=symbol,
                    timestamp=end_date,
                    period=f"Q{fiscal_quarter}",
                    fiscal_year=end_date.year,

                    # Income Statement
                    revenue=self._get_raw(income, "totalRevenue"),
                    cost_of_revenue=self._get_raw(income, "costOfRevenue"),
                    gross_profit=self._get_raw(income, "grossProfit"),
                    operating_expenses=self._get_raw(income, "totalOperatingExpenses"),
                    operating_income=self._get_raw(income, "operatingIncome"),
                    ebitda=self._get_raw(income, "ebitda"),
                    ebit=self._get_raw(income, "ebit"),
                    interest_expense=self._get_raw(income, "interestExpense"),
                    pretax_income=self._get_raw(income, "incomeBeforeTax"),
                    income_tax=self._get_raw(income, "incomeTaxExpense"),
                    net_income=self._get_raw(income, "netIncome"),
                    eps_basic=self._get_raw(income, "basicEPS"),
                    eps_diluted=self._get_raw(income, "dilutedEPS"),

                    # Balance Sheet
                    total_assets=self._get_raw(balance, "totalAssets"),
                    current_assets=self._get_raw(balance, "totalCurrentAssets"),
                    cash_and_equivalents=self._get_raw(balance, "cash"),
                    short_term_investments=self._get_raw(balance, "shortTermInvestments"),
                    accounts_receivable=self._get_raw(balance, "netReceivables"),
                    inventory=self._get_raw(balance, "inventory"),
                    total_liabilities=self._get_raw(balance, "totalLiab"),
                    current_liabilities=self._get_raw(balance, "totalCurrentLiabilities"),
                    accounts_payable=self._get_raw(balance, "accountsPayable"),
                    short_term_debt=self._get_raw(balance, "shortLongTermDebt"),
                    long_term_debt=self._get_raw(balance, "longTermDebt"),
                    total_debt=self._get_raw(balance, "totalDebt"),
                    total_equity=self._get_raw(balance, "totalStockholderEquity"),
                    retained_earnings=self._get_raw(balance, "retainedEarnings"),

                    # Cash Flow
                    operating_cash_flow=self._get_raw(cashflow, "totalCashFromOperatingActivities"),
                    capital_expenditures=self._get_raw(cashflow, "capitalExpenditures"),
                    free_cash_flow=self._get_raw(cashflow, "freeCashFlow"),
                    investing_cash_flow=self._get_raw(cashflow, "totalCashflowsFromInvestingActivities"),
                    financing_cash_flow=self._get_raw(cashflow, "totalCashFromFinancingActivities"),
                    dividends_paid=self._get_raw(cashflow, "dividendsPaid"),
                    share_repurchases=self._get_raw(cashflow, "repurchaseOfStock"),
                )

                fundamentals.append(fundamental)

            return fundamentals

        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing fundamentals: {e}")
            raise Exception(f"Failed to parse fundamentals: {e}")

    def _get_raw(self, data: Dict, key: str) -> Optional[float]:
        """Safely extract raw value from Yahoo data"""
        if key in data and isinstance(data[key], dict) and "raw" in data[key]:
            return data[key]["raw"]
        return None

    async def get_real_time_quote(self, security: SecurityIdentifier) -> Quote:
        """Fetch real-time quote"""
        symbol = self._get_symbol(security)

        url = f"{self.QUOTE_URL}/v6/finance/quote"
        params = {"symbols": symbol}

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance quote error: {response.status}")

            data = await response.json()

        try:
            quote_data = data["quoteResponse"]["result"][0]
            return Quote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=quote_data.get("bid", 0),
                ask=quote_data.get("ask", 0),
                bid_size=quote_data.get("bidSize", 0),
                ask_size=quote_data.get("askSize", 0),
                last_price=quote_data.get("regularMarketPrice", 0),
                last_size=0,
                volume=quote_data.get("regularMarketVolume", 0),
                change=quote_data.get("regularMarketChange", 0),
                change_percent=quote_data.get("regularMarketChangePercent", 0),
            )
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse quote: {e}")

    async def search_securities(
        self,
        query: str,
        asset_class: Optional[str] = None,
        market: Optional[str] = None
    ) -> List[Dict]:
        """Search for securities"""
        url = f"{self.BASE_URL}/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": 20,
            "newsCount": 0,
        }

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance search error: {response.status}")

            data = await response.json()

        results = []
        for quote in data.get("quotes", []):
            # Filter by asset class if specified
            quote_type = quote.get("quoteType", "").lower()
            if asset_class:
                if asset_class.lower() == "equity" and quote_type != "equity":
                    continue
                elif asset_class.lower() == "etf" and quote_type != "etf":
                    continue

            # Filter by market if specified
            exchange = quote.get("exchange", "")
            if market:
                market_exchanges = {
                    "usa": ["NYQ", "NMS", "NGM", "NCM", "ASE"],
                    "singapore": ["SES"],
                    "india": ["NSI", "BSE"],
                    "uae": ["DFM", "ADX"],
                }
                if exchange not in market_exchanges.get(market, []):
                    continue

            results.append({
                "symbol": quote.get("symbol"),
                "name": quote.get("longname") or quote.get("shortname"),
                "exchange": exchange,
                "type": quote_type,
                "industry": quote.get("industry"),
                "sector": quote.get("sector"),
            })

        return results

    async def get_key_statistics(self, security: SecurityIdentifier) -> Dict:
        """Get key statistics and ratios"""
        symbol = self._get_symbol(security)

        url = f"{self.QUOTE_URL}/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "defaultKeyStatistics,financialData,summaryDetail"}

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance statistics error: {response.status}")

            data = await response.json()

        result = data["quoteSummary"]["result"][0]
        stats = {}

        # Key Statistics
        key_stats = result.get("defaultKeyStatistics", {})
        stats.update({
            "beta": self._get_raw(key_stats, "beta"),
            "book_value": self._get_raw(key_stats, "bookValue"),
            "price_to_book": self._get_raw(key_stats, "priceToBook"),
            "trailing_eps": self._get_raw(key_stats, "trailingEps"),
            "forward_eps": self._get_raw(key_stats, "forwardEps"),
            "peg_ratio": self._get_raw(key_stats, "pegRatio"),
            "enterprise_value": self._get_raw(key_stats, "enterpriseValue"),
            "enterprise_to_revenue": self._get_raw(key_stats, "enterpriseToRevenue"),
            "enterprise_to_ebitda": self._get_raw(key_stats, "enterpriseToEbitda"),
            "profit_margins": self._get_raw(key_stats, "profitMargins"),
            "float_shares": self._get_raw(key_stats, "floatShares"),
            "shares_outstanding": self._get_raw(key_stats, "sharesOutstanding"),
            "shares_short": self._get_raw(key_stats, "sharesShort"),
            "short_ratio": self._get_raw(key_stats, "shortRatio"),
            "short_percent_of_float": self._get_raw(key_stats, "shortPercentOfFloat"),
            "held_percent_insiders": self._get_raw(key_stats, "heldPercentInsiders"),
            "held_percent_institutions": self._get_raw(key_stats, "heldPercentInstitutions"),
        })

        # Financial Data
        fin_data = result.get("financialData", {})
        stats.update({
            "current_price": self._get_raw(fin_data, "currentPrice"),
            "target_high_price": self._get_raw(fin_data, "targetHighPrice"),
            "target_low_price": self._get_raw(fin_data, "targetLowPrice"),
            "target_mean_price": self._get_raw(fin_data, "targetMeanPrice"),
            "recommendation_mean": self._get_raw(fin_data, "recommendationMean"),
            "recommendation_key": fin_data.get("recommendationKey"),
            "number_of_analyst_opinions": self._get_raw(fin_data, "numberOfAnalystOpinions"),
            "total_cash": self._get_raw(fin_data, "totalCash"),
            "total_cash_per_share": self._get_raw(fin_data, "totalCashPerShare"),
            "ebitda": self._get_raw(fin_data, "ebitda"),
            "total_debt": self._get_raw(fin_data, "totalDebt"),
            "quick_ratio": self._get_raw(fin_data, "quickRatio"),
            "current_ratio": self._get_raw(fin_data, "currentRatio"),
            "total_revenue": self._get_raw(fin_data, "totalRevenue"),
            "debt_to_equity": self._get_raw(fin_data, "debtToEquity"),
            "revenue_per_share": self._get_raw(fin_data, "revenuePerShare"),
            "return_on_assets": self._get_raw(fin_data, "returnOnAssets"),
            "return_on_equity": self._get_raw(fin_data, "returnOnEquity"),
            "gross_profits": self._get_raw(fin_data, "grossProfits"),
            "free_cashflow": self._get_raw(fin_data, "freeCashflow"),
            "operating_cashflow": self._get_raw(fin_data, "operatingCashflow"),
            "earnings_growth": self._get_raw(fin_data, "earningsGrowth"),
            "revenue_growth": self._get_raw(fin_data, "revenueGrowth"),
            "gross_margins": self._get_raw(fin_data, "grossMargins"),
            "operating_margins": self._get_raw(fin_data, "operatingMargins"),
            "profit_margins": self._get_raw(fin_data, "profitMargins"),
        })

        # Summary Detail
        summary = result.get("summaryDetail", {})
        stats.update({
            "pe_ratio": self._get_raw(summary, "trailingPE"),
            "forward_pe": self._get_raw(summary, "forwardPE"),
            "price_to_sales": self._get_raw(summary, "priceToSalesTrailing12Months"),
            "dividend_yield": self._get_raw(summary, "dividendYield"),
            "dividend_rate": self._get_raw(summary, "dividendRate"),
            "payout_ratio": self._get_raw(summary, "payoutRatio"),
            "five_year_avg_dividend_yield": self._get_raw(summary, "fiveYearAvgDividendYield"),
            "ex_dividend_date": self._get_raw(summary, "exDividendDate"),
            "market_cap": self._get_raw(summary, "marketCap"),
            "fifty_two_week_low": self._get_raw(summary, "fiftyTwoWeekLow"),
            "fifty_two_week_high": self._get_raw(summary, "fiftyTwoWeekHigh"),
            "fifty_day_average": self._get_raw(summary, "fiftyDayAverage"),
            "two_hundred_day_average": self._get_raw(summary, "twoHundredDayAverage"),
            "volume": self._get_raw(summary, "volume"),
            "average_volume": self._get_raw(summary, "averageVolume"),
            "average_volume_10days": self._get_raw(summary, "averageVolume10days"),
        })

        return stats

    async def get_insider_transactions(self, security: SecurityIdentifier) -> List[Dict]:
        """Get insider trading activity"""
        symbol = self._get_symbol(security)

        url = f"{self.QUOTE_URL}/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "insiderTransactions,insiderHolders"}

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance insider error: {response.status}")

            data = await response.json()

        result = data["quoteSummary"]["result"][0]
        transactions = []

        insider_txns = result.get("insiderTransactions", {}).get("transactions", [])
        for txn in insider_txns:
            transactions.append({
                "name": txn.get("filerName"),
                "relation": txn.get("filerRelation"),
                "transaction_type": txn.get("transactionText"),
                "date": txn.get("startDate", {}).get("fmt"),
                "shares": self._get_raw(txn, "shares"),
                "value": self._get_raw(txn, "value"),
                "ownership": txn.get("ownership"),
            })

        return transactions

    async def get_institutional_holders(self, security: SecurityIdentifier) -> List[Dict]:
        """Get institutional ownership"""
        symbol = self._get_symbol(security)

        url = f"{self.QUOTE_URL}/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "institutionOwnership,majorHoldersBreakdown"}

        await self._rate_limit_check()

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Yahoo Finance institutions error: {response.status}")

            data = await response.json()

        result = data["quoteSummary"]["result"][0]
        holders = []

        ownership = result.get("institutionOwnership", {}).get("ownershipList", [])
        for holder in ownership:
            holders.append({
                "organization": holder.get("organization"),
                "pct_held": self._get_raw(holder, "pctHeld"),
                "position": self._get_raw(holder, "position"),
                "value": self._get_raw(holder, "value"),
                "report_date": holder.get("reportDate", {}).get("fmt"),
                "pct_change": self._get_raw(holder, "pctChange"),
            })

        return holders

    def get_supported_markets(self) -> List[str]:
        """Return list of supported markets"""
        return list(self.MARKET_SUFFIXES.keys())

    def get_supported_asset_classes(self) -> List[str]:
        """Return list of supported asset classes"""
        return ["equity", "etf", "bond", "commodity", "reit", "forex", "crypto"]
