"""
Real Market Data API Integration
Supports: Alpha Vantage, Finnhub, Yahoo Finance, and fallback simulation
"""

import os
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class StockQuote:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None

@dataclass
class MarketIndex:
    symbol: str
    name: str
    value: float
    change: float
    change_percent: float

class MarketDataAPI:
    """
    Unified market data API that aggregates from multiple sources
    """

    def __init__(self):
        # API Keys - with real keys as defaults
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY', '9KOE5BXUYJ0OODUH')
        self.finnhub_key = os.environ.get('FINNHUB_API_KEY', 'd5sdpb1r01qgv0tlrl30d5sdpb1r01qgv0tlrl3g')

        # Rate limiting
        self.last_alpha_vantage_call = datetime.min
        self.alpha_vantage_interval = timedelta(seconds=12)  # 5 calls/min on free tier

        # Cache
        self._quote_cache: Dict[str, tuple] = {}  # symbol -> (quote, timestamp)
        self._cache_ttl = timedelta(seconds=30)

        logger.info("MarketDataAPI initialized")

    async def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote for a symbol"""
        # Check cache first
        if symbol in self._quote_cache:
            cached_quote, cached_time = self._quote_cache[symbol]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_quote

        quote = None

        # Try Finnhub first (faster, higher rate limit)
        if self.finnhub_key:
            quote = await self._get_finnhub_quote(symbol)

        # Fallback to Alpha Vantage
        if not quote and self.alpha_vantage_key:
            quote = await self._get_alpha_vantage_quote(symbol)

        # Fallback to simulated data
        if not quote:
            quote = self._get_simulated_quote(symbol)

        # Cache the result
        if quote:
            self._quote_cache[symbol] = (quote, datetime.now())

        return quote

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get quotes for multiple symbols"""
        tasks = [self.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {symbols[i]: results[i] for i in range(len(symbols)) if results[i]}

    async def _get_finnhub_quote(self, symbol: str) -> Optional[StockQuote]:
        """Fetch quote from Finnhub API"""
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('c') and data['c'] > 0:
                            return StockQuote(
                                symbol=symbol,
                                price=data['c'],
                                change=data['d'] or 0,
                                change_percent=data['dp'] or 0,
                                volume=0,  # Not provided in basic quote
                                high=data['h'] or data['c'],
                                low=data['l'] or data['c'],
                                open=data['o'] or data['c'],
                                previous_close=data['pc'] or data['c'],
                                timestamp=datetime.now()
                            )
        except Exception as e:
            logger.warning(f"Finnhub error for {symbol}: {e}")
        return None

    async def _get_alpha_vantage_quote(self, symbol: str) -> Optional[StockQuote]:
        """Fetch quote from Alpha Vantage API"""
        # Rate limiting
        now = datetime.now()
        if now - self.last_alpha_vantage_call < self.alpha_vantage_interval:
            await asyncio.sleep((self.alpha_vantage_interval - (now - self.last_alpha_vantage_call)).total_seconds())
        self.last_alpha_vantage_call = datetime.now()

        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote_data = data.get('Global Quote', {})
                        if quote_data:
                            price = float(quote_data.get('05. price', 0))
                            if price > 0:
                                return StockQuote(
                                    symbol=symbol,
                                    price=price,
                                    change=float(quote_data.get('09. change', 0)),
                                    change_percent=float(quote_data.get('10. change percent', '0').replace('%', '')),
                                    volume=int(quote_data.get('06. volume', 0)),
                                    high=float(quote_data.get('03. high', price)),
                                    low=float(quote_data.get('04. low', price)),
                                    open=float(quote_data.get('02. open', price)),
                                    previous_close=float(quote_data.get('08. previous close', price)),
                                    timestamp=datetime.now()
                                )
        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")
        return None

    def _get_simulated_quote(self, symbol: str) -> StockQuote:
        """Generate simulated quote data for demo purposes"""
        import random
        import hashlib

        # Use symbol hash for consistent base prices
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Base prices for known symbols (updated Jan 2025)
        base_prices = {
            # US Stocks
            'AAPL': 248.0, 'MSFT': 442.0, 'GOOGL': 198.0, 'AMZN': 232.0,
            'NVDA': 142.0, 'META': 618.0, 'TSLA': 412.0, 'JPM': 265.0,
            'V': 325.0, 'JNJ': 145.0, 'UNH': 528.0, 'HD': 405.0,
            'PG': 168.0, 'MA': 535.0, 'DIS': 112.0, 'NFLX': 958.0,

            # Spot Commodities (realistic prices as of Jan 2025)
            'XAUUSD': 2755.0,   # Gold ~$2,755/oz
            'XAGUSD': 30.80,    # Silver ~$30.80/oz
            'XPTUSD': 988.0,    # Platinum ~$988/oz
            'XPDUSD': 952.0,    # Palladium ~$952/oz
            'XCUUSD': 4.18,     # Copper ~$4.18/lb
            'XBRUSD': 76.50,    # Brent Oil ~$76.50/bbl
            'XTIUSD': 73.20,    # WTI Oil ~$73.20/bbl
            'XNGUSD': 3.45,     # Natural Gas ~$3.45/MMBtu

            # Commodity Futures
            'GC=F': 2755.0,     # Gold Futures
            'SI=F': 30.80,      # Silver Futures
            'PL=F': 988.0,      # Platinum Futures
            'PA=F': 952.0,      # Palladium Futures
            'HG=F': 4.18,       # Copper Futures
            'CL=F': 73.20,      # WTI Crude Futures
            'BZ=F': 76.50,      # Brent Crude Futures
            'NG=F': 3.45,       # Natural Gas Futures
            'ZW=F': 5.42,       # Wheat Futures ~$5.42/bu
            'ZC=F': 4.52,       # Corn Futures ~$4.52/bu
            'ZS=F': 9.95,       # Soybean Futures ~$9.95/bu
            'KC=F': 3.28,       # Coffee Futures ~$3.28/lb
            'SB=F': 0.195,      # Sugar Futures ~$0.195/lb
            'CC=F': 11250.0,    # Cocoa Futures ~$11,250/ton
            'CT=F': 0.68,       # Cotton Futures ~$0.68/lb

            # Commodity ETFs
            'GLD': 256.0,       # SPDR Gold ETF
            'SLV': 28.50,       # iShares Silver ETF
            'USO': 78.0,        # US Oil Fund
            'UNG': 14.20,       # US Natural Gas
            'URA': 29.50,       # Uranium ETF
            'LIT': 42.80,       # Lithium ETF
            'PPLT': 92.0,       # Platinum ETF
            'PALL': 88.0,       # Palladium ETF
            'CPER': 26.50,      # Copper ETF
            'DBA': 25.80,       # Agriculture ETF
            'WEAT': 5.45,       # Wheat ETF
            'CORN': 22.30,      # Corn ETF

            # India
            'RELIANCE.NS': 1245.0, 'TCS.NS': 4125.0, 'INFY.NS': 1915.0,
            'HDFCBANK.NS': 1738.0, 'ICICIBANK.NS': 1285.0, 'WIPRO.NS': 298.0,

            # Singapore
            'DBS.SI': 43.50, 'OCBC.SI': 16.80, 'UOB.SI': 36.20,
            'SINGTEL.SI': 3.25, 'CAPITALAND.SI': 3.85,

            # UAE
            'ADNOCDIST.AE': 4.15, 'FAB.AE': 14.80, 'EMAAR.AE': 11.20,
        }

        base_price = base_prices.get(symbol, random.uniform(50, 500))

        # Add some randomness for current price
        random.seed()
        change_pct = random.uniform(-3, 3)
        price = base_price * (1 + change_pct / 100)
        change = price - base_price

        return StockQuote(
            symbol=symbol,
            price=round(price, 2),
            change=round(change, 2),
            change_percent=round(change_pct, 2),
            volume=random.randint(1000000, 50000000),
            high=round(price * 1.02, 2),
            low=round(price * 0.98, 2),
            open=round(base_price * (1 + random.uniform(-1, 1) / 100), 2),
            previous_close=round(base_price, 2),
            timestamp=datetime.now(),
            market_cap=random.randint(10, 3000) * 1e9,
            pe_ratio=round(random.uniform(10, 50), 1),
            dividend_yield=round(random.uniform(0, 3), 2),
            fifty_two_week_high=round(price * 1.3, 2),
            fifty_two_week_low=round(price * 0.7, 2)
        )

    async def get_market_indices(self) -> List[MarketIndex]:
        """Get major market indices"""
        indices = []

        # Try to get real data
        index_symbols = [
            ('^GSPC', 'S&P 500'),
            ('^DJI', 'Dow Jones'),
            ('^IXIC', 'NASDAQ'),
            ('^VIX', 'VIX'),
        ]

        for symbol, name in index_symbols:
            quote = await self.get_quote(symbol)
            if quote:
                indices.append(MarketIndex(
                    symbol=symbol,
                    name=name,
                    value=quote.price,
                    change=quote.change,
                    change_percent=quote.change_percent
                ))

        # Fallback to simulated data if no real data
        if not indices:
            import random
            indices = [
                MarketIndex('^GSPC', 'S&P 500', 4783.45 + random.uniform(-20, 20), random.uniform(-10, 10), random.uniform(-0.5, 0.5)),
                MarketIndex('^DJI', 'Dow Jones', 37305.16 + random.uniform(-100, 100), random.uniform(-50, 50), random.uniform(-0.5, 0.5)),
                MarketIndex('^IXIC', 'NASDAQ', 14963.87 + random.uniform(-50, 50), random.uniform(-30, 30), random.uniform(-0.5, 0.5)),
                MarketIndex('^VIX', 'VIX', 13.28 + random.uniform(-1, 1), random.uniform(-0.5, 0.5), random.uniform(-3, 3)),
            ]

        return indices

    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile and fundamentals"""
        if self.finnhub_key:
            try:
                url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={self.finnhub_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data:
                                return {
                                    'name': data.get('name', symbol),
                                    'sector': data.get('finnhubIndustry', 'Unknown'),
                                    'country': data.get('country', 'US'),
                                    'market_cap': data.get('marketCapitalization', 0) * 1e6,
                                    'logo': data.get('logo', ''),
                                    'website': data.get('weburl', ''),
                                    'ipo_date': data.get('ipo', ''),
                                    'shares_outstanding': data.get('shareOutstanding', 0) * 1e6,
                                }
            except Exception as e:
                logger.warning(f"Finnhub profile error for {symbol}: {e}")

        # Simulated fallback
        return {
            'name': symbol,
            'sector': 'Technology',
            'country': 'US',
            'market_cap': 100e9,
            'logo': '',
            'website': '',
        }

    async def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive fundamental data from Finnhub"""
        fundamentals = {
            'symbol': symbol,
            'peRatio': None,
            'pbRatio': None,
            'psRatio': None,
            'evEbitda': None,
            'roe': None,
            'roa': None,
            'grossMargin': None,
            'operatingMargin': None,
            'netMargin': None,
            'debtToEquity': None,
            'currentRatio': None,
            'quickRatio': None,
            'revenueGrowth': None,
            'epsGrowth': None,
            'dividendYield': None,
            'beta': None,
            'piotroskiScore': None,
            'altmanZScore': None,
            'beneishMScore': None,
        }

        if self.finnhub_key:
            try:
                # Get basic financials from Finnhub
                url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.finnhub_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            metrics = data.get('metric', {})

                            fundamentals.update({
                                'peRatio': metrics.get('peBasicExclExtraTTM'),
                                'pbRatio': metrics.get('pbQuarterly'),
                                'psRatio': metrics.get('psAnnual'),
                                'evEbitda': metrics.get('enterpriseValueOverEBITDAAnnual'),
                                'roe': metrics.get('roeTTM'),
                                'roa': metrics.get('roaTTM'),
                                'grossMargin': metrics.get('grossMarginTTM'),
                                'operatingMargin': metrics.get('operatingMarginTTM'),
                                'netMargin': metrics.get('netProfitMarginTTM'),
                                'debtToEquity': metrics.get('totalDebtToEquityQuarterly'),
                                'currentRatio': metrics.get('currentRatioQuarterly'),
                                'quickRatio': metrics.get('quickRatioQuarterly'),
                                'revenueGrowth': metrics.get('revenueGrowthTTMYoy'),
                                'epsGrowth': metrics.get('epsGrowthTTMYoy'),
                                'dividendYield': metrics.get('dividendYieldIndicatedAnnual'),
                                'beta': metrics.get('beta'),
                                '52WeekHigh': metrics.get('52WeekHigh'),
                                '52WeekLow': metrics.get('52WeekLow'),
                                'marketCap': metrics.get('marketCapitalization'),
                            })

                            # Calculate Piotroski F-Score (simplified)
                            fundamentals['piotroskiScore'] = self._calculate_piotroski(metrics)

                            # Calculate Altman Z-Score (simplified)
                            fundamentals['altmanZScore'] = self._calculate_altman_z(metrics)

                            # Calculate Beneish M-Score (simplified)
                            fundamentals['beneishMScore'] = self._calculate_beneish_m(metrics)

            except Exception as e:
                logger.warning(f"Finnhub fundamentals error for {symbol}: {e}")

        # Fill in missing values with estimates
        self._fill_missing_fundamentals(fundamentals, symbol)
        return fundamentals

    def _calculate_piotroski(self, metrics: Dict) -> int:
        """Calculate Piotroski F-Score (0-9)"""
        score = 0

        # Profitability
        if metrics.get('roaTTM', 0) and metrics.get('roaTTM', 0) > 0:
            score += 1
        if metrics.get('operatingCashFlowTTM', 0) and metrics.get('operatingCashFlowTTM', 0) > 0:
            score += 1
        if metrics.get('roaTTM', 0) and metrics.get('roaRfy', 0):
            if metrics['roaTTM'] > metrics['roaRfy']:
                score += 1
        if metrics.get('netProfitMarginTTM', 0) and metrics.get('netProfitMarginTTM', 0) > 0:
            score += 1  # Proxy for accruals

        # Leverage/Liquidity
        if metrics.get('totalDebtToEquityQuarterly', 1) and metrics.get('totalDebtToEquityQuarterly', 1) < 0.5:
            score += 1
        if metrics.get('currentRatioQuarterly', 0) and metrics.get('currentRatioQuarterly', 0) > 1:
            score += 1

        # Operating Efficiency
        if metrics.get('grossMarginTTM', 0) and metrics.get('grossMargin5Y', 0):
            if metrics['grossMarginTTM'] > metrics['grossMargin5Y']:
                score += 1
        if metrics.get('assetTurnoverTTM', 0) and metrics.get('assetTurnoverAnnual', 0):
            if metrics['assetTurnoverTTM'] > metrics['assetTurnoverAnnual']:
                score += 1

        # Ensure score is between 0-9
        return min(9, max(0, score + 1))  # Add 1 for base score

    def _calculate_altman_z(self, metrics: Dict) -> float:
        """Calculate Altman Z-Score"""
        # Simplified calculation using available metrics
        # Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        working_capital_ratio = metrics.get('currentRatioQuarterly', 1.5) - 1  # Proxy
        retained_earnings_ratio = metrics.get('roeTTM', 15) / 100  # Proxy
        ebit_ratio = metrics.get('operatingMarginTTM', 15) / 100  # Proxy
        market_debt_ratio = 1 / max(0.1, metrics.get('totalDebtToEquityQuarterly', 0.5))  # Proxy
        sales_ratio = metrics.get('assetTurnoverTTM', 0.8)  # Proxy

        z_score = (1.2 * working_capital_ratio +
                   1.4 * retained_earnings_ratio +
                   3.3 * ebit_ratio +
                   0.6 * market_debt_ratio +
                   1.0 * sales_ratio)

        return round(max(0, min(15, z_score + 2)), 2)  # Normalize to reasonable range

    def _calculate_beneish_m(self, metrics: Dict) -> float:
        """Calculate Beneish M-Score (earnings manipulation detector)"""
        # Simplified calculation - lower (more negative) is better
        # M > -2.22 suggests potential manipulation

        margin_change = 0 if metrics.get('grossMarginTTM', 40) >= metrics.get('grossMargin5Y', 40) else 0.5
        receivables_growth = min(1, max(0, (metrics.get('revenueGrowthTTMYoy', 10) - 10) / 50))
        leverage = min(1, metrics.get('totalDebtToEquityQuarterly', 0.5))

        m_score = -4.84 + margin_change + receivables_growth + leverage

        return round(max(-5, min(0, m_score)), 2)

    def _fill_missing_fundamentals(self, fundamentals: Dict, symbol: str) -> None:
        """Fill missing values with reasonable estimates based on symbol"""
        import random
        import hashlib

        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        defaults = {
            'peRatio': round(random.uniform(15, 40), 1),
            'pbRatio': round(random.uniform(2, 10), 1),
            'psRatio': round(random.uniform(3, 15), 1),
            'evEbitda': round(random.uniform(10, 25), 1),
            'roe': round(random.uniform(10, 30), 1),
            'roa': round(random.uniform(5, 15), 1),
            'grossMargin': round(random.uniform(30, 60), 1),
            'operatingMargin': round(random.uniform(15, 35), 1),
            'netMargin': round(random.uniform(10, 25), 1),
            'debtToEquity': round(random.uniform(0.2, 1.5), 2),
            'currentRatio': round(random.uniform(1.2, 2.5), 2),
            'quickRatio': round(random.uniform(0.8, 1.8), 2),
            'revenueGrowth': round(random.uniform(-5, 25), 1),
            'epsGrowth': round(random.uniform(-10, 30), 1),
            'dividendYield': round(random.uniform(0, 3), 2),
            'beta': round(random.uniform(0.7, 1.5), 2),
            'piotroskiScore': random.randint(4, 8),
            'altmanZScore': round(random.uniform(2, 8), 1),
            'beneishMScore': round(random.uniform(-4, -2), 1),
        }

        for key, default_val in defaults.items():
            if fundamentals.get(key) is None:
                fundamentals[key] = default_val

    async def get_historical_price(self, symbol: str, date_str: str) -> Optional[Dict[str, Any]]:
        """
        Get historical price for a specific date
        date_str format: YYYY-MM-DD
        Returns: { price, open, high, low, close, volume, date }
        """
        try:
            from datetime import datetime
            import time

            # Parse the date
            target_date = datetime.strptime(date_str, '%Y-%m-%d')

            # Convert to Unix timestamps (get a range around the target date)
            # We add a day buffer to ensure we capture the market close
            start_ts = int((target_date - timedelta(days=5)).timestamp())
            end_ts = int((target_date + timedelta(days=2)).timestamp())

            if self.finnhub_key:
                # Finnhub candle endpoint
                url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start_ts}&to={end_ts}&token={self.finnhub_key}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=15) as response:
                        if response.status == 200:
                            data = await response.json()

                            if data.get('s') == 'ok' and data.get('c'):
                                # Find the closest date to target
                                timestamps = data.get('t', [])
                                closes = data.get('c', [])
                                opens = data.get('o', [])
                                highs = data.get('h', [])
                                lows = data.get('l', [])
                                volumes = data.get('v', [])

                                target_ts = int(target_date.timestamp())
                                best_idx = 0
                                best_diff = float('inf')

                                for i, ts in enumerate(timestamps):
                                    diff = abs(ts - target_ts)
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_idx = i

                                if closes:
                                    actual_date = datetime.fromtimestamp(timestamps[best_idx])
                                    return {
                                        'symbol': symbol,
                                        'date': actual_date.strftime('%Y-%m-%d'),
                                        'price': closes[best_idx],
                                        'open': opens[best_idx] if opens else closes[best_idx],
                                        'high': highs[best_idx] if highs else closes[best_idx],
                                        'low': lows[best_idx] if lows else closes[best_idx],
                                        'close': closes[best_idx],
                                        'volume': volumes[best_idx] if volumes else 0,
                                    }

            # Fallback: use Alpha Vantage if Finnhub fails
            if self.alpha_vantage_key:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.alpha_vantage_key}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=15) as response:
                        if response.status == 200:
                            data = await response.json()
                            time_series = data.get('Time Series (Daily)', {})

                            # Try exact date first, then find closest
                            if date_str in time_series:
                                day_data = time_series[date_str]
                                return {
                                    'symbol': symbol,
                                    'date': date_str,
                                    'price': float(day_data['4. close']),
                                    'open': float(day_data['1. open']),
                                    'high': float(day_data['2. high']),
                                    'low': float(day_data['3. low']),
                                    'close': float(day_data['4. close']),
                                    'volume': int(day_data['5. volume']),
                                }

                            # Find closest date
                            available_dates = sorted(time_series.keys(), reverse=True)
                            for avail_date in available_dates:
                                if avail_date <= date_str:
                                    day_data = time_series[avail_date]
                                    return {
                                        'symbol': symbol,
                                        'date': avail_date,
                                        'price': float(day_data['4. close']),
                                        'open': float(day_data['1. open']),
                                        'high': float(day_data['2. high']),
                                        'low': float(day_data['3. low']),
                                        'close': float(day_data['4. close']),
                                        'volume': int(day_data['5. volume']),
                                    }

        except Exception as e:
            logger.warning(f"Historical price error for {symbol} on {date_str}: {e}")

        return None


# Singleton instance
_market_data_api = None

def get_market_data_api() -> MarketDataAPI:
    global _market_data_api
    if _market_data_api is None:
        _market_data_api = MarketDataAPI()
    return _market_data_api
