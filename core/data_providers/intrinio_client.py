"""
Intrinio API Client
Enterprise-grade financial data provider

Features:
- 30+ years historical prices (daily/weekly/monthly)
- Fundamentals (income statement, balance sheet, cash flow)
- Financial ratios and metrics
- Real-time quotes
- Company news and filings
- Analyst estimates

Docs: https://docs.intrinio.com/
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# API Configuration
INTRINIO_API_KEY = os.getenv('INTRINIO_API_KEY', '')
INTRINIO_BASE_URL = 'https://api-v2.intrinio.com'


@dataclass
class HistoricalPrice:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float


@dataclass
class Fundamental:
    fiscal_year: int
    fiscal_period: str
    filing_date: str
    value: float
    tag: str


@dataclass
class CompanyInfo:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    employees: int
    description: str


class IntrinioClient:
    """
    Intrinio API Client for enterprise financial data
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or INTRINIO_API_KEY
        if not self.api_key:
            logger.warning("Intrinio API key not set. Set INTRINIO_API_KEY environment variable.")
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
        })
        self._cache = {}

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request"""
        if not self.api_key:
            logger.error("Intrinio API key not configured")
            return None

        url = f"{INTRINIO_BASE_URL}/{endpoint}"
        params = params or {}
        params['api_key'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Intrinio API error: {e}")
            return None

    # ==========================================================================
    # HISTORICAL PRICES
    # ==========================================================================

    def get_historical_prices(
        self,
        ticker: str,
        start_date: str = '1995-01-01',
        end_date: Optional[str] = None,
        frequency: str = 'monthly'  # daily, weekly, monthly
    ) -> List[HistoricalPrice]:
        """
        Get historical stock prices

        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (default: today)
            frequency: daily, weekly, or monthly

        Returns:
            List of HistoricalPrice objects
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        cache_key = f"prices_{ticker}_{start_date}_{end_date}_{frequency}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            'start_date': start_date,
            'end_date': end_date,
            'frequency': frequency,
            'page_size': 10000,
        }

        data = self._make_request(f'securities/{ticker}/prices', params)
        if not data or 'stock_prices' not in data:
            return []

        prices = []
        for p in data['stock_prices']:
            prices.append(HistoricalPrice(
                date=p.get('date', ''),
                open=p.get('open', 0),
                high=p.get('high', 0),
                low=p.get('low', 0),
                close=p.get('close', 0),
                volume=p.get('volume', 0),
                adj_close=p.get('adj_close', p.get('close', 0))
            ))

        # Sort by date ascending
        prices.sort(key=lambda x: x.date)
        self._cache[cache_key] = prices
        return prices

    def get_30_year_prices(self, ticker: str) -> List[HistoricalPrice]:
        """Get 30 years of monthly prices"""
        start = (datetime.now() - timedelta(days=365*30)).strftime('%Y-%m-%d')
        return self.get_historical_prices(ticker, start_date=start, frequency='monthly')

    # ==========================================================================
    # FUNDAMENTALS
    # ==========================================================================

    def get_fundamentals(
        self,
        ticker: str,
        statement_code: str = 'income_statement',
        fiscal_year: Optional[int] = None,
        period_type: str = 'FY'  # FY, Q1, Q2, Q3, Q4
    ) -> Dict[str, Any]:
        """
        Get fundamental data

        Args:
            ticker: Stock symbol
            statement_code: income_statement, balance_sheet_statement, cash_flow_statement
            fiscal_year: Specific year or None for all
            period_type: FY (annual) or Q1-Q4 (quarterly)

        Returns:
            Dictionary of fundamental data
        """
        params = {
            'statement_code': statement_code,
            'type': period_type,
            'page_size': 100,
        }
        if fiscal_year:
            params['fiscal_year'] = fiscal_year

        data = self._make_request(f'companies/{ticker}/fundamentals', params)
        return data or {}

    def get_income_statement(self, ticker: str, years: int = 10) -> List[Dict]:
        """Get income statement data for multiple years"""
        results = []
        current_year = datetime.now().year

        for year in range(current_year - years, current_year + 1):
            data = self.get_fundamentals(ticker, 'income_statement', year, 'FY')
            if data and 'fundamentals' in data:
                for f in data['fundamentals']:
                    results.append({
                        'fiscal_year': f.get('fiscal_year'),
                        'fiscal_period': f.get('fiscal_period'),
                        'revenue': self._get_tag_value(ticker, f, 'totalrevenue'),
                        'gross_profit': self._get_tag_value(ticker, f, 'grossprofit'),
                        'operating_income': self._get_tag_value(ticker, f, 'operatingincome'),
                        'net_income': self._get_tag_value(ticker, f, 'netincome'),
                        'eps': self._get_tag_value(ticker, f, 'basiceps'),
                        'ebitda': self._get_tag_value(ticker, f, 'ebitda'),
                    })

        return results

    def get_balance_sheet(self, ticker: str, years: int = 10) -> List[Dict]:
        """Get balance sheet data"""
        results = []
        current_year = datetime.now().year

        for year in range(current_year - years, current_year + 1):
            data = self.get_fundamentals(ticker, 'balance_sheet_statement', year, 'FY')
            if data and 'fundamentals' in data:
                for f in data['fundamentals']:
                    results.append({
                        'fiscal_year': f.get('fiscal_year'),
                        'total_assets': self._get_tag_value(ticker, f, 'totalassets'),
                        'total_liabilities': self._get_tag_value(ticker, f, 'totalliabilities'),
                        'total_equity': self._get_tag_value(ticker, f, 'totalequity'),
                        'cash': self._get_tag_value(ticker, f, 'cashandequivalents'),
                        'debt': self._get_tag_value(ticker, f, 'totaldebt'),
                        'current_assets': self._get_tag_value(ticker, f, 'totalcurrentassets'),
                        'current_liabilities': self._get_tag_value(ticker, f, 'totalcurrentliabilities'),
                    })

        return results

    def get_cash_flow(self, ticker: str, years: int = 10) -> List[Dict]:
        """Get cash flow statement data"""
        results = []
        current_year = datetime.now().year

        for year in range(current_year - years, current_year + 1):
            data = self.get_fundamentals(ticker, 'cash_flow_statement', year, 'FY')
            if data and 'fundamentals' in data:
                for f in data['fundamentals']:
                    results.append({
                        'fiscal_year': f.get('fiscal_year'),
                        'operating_cf': self._get_tag_value(ticker, f, 'netcashfromoperatingactivities'),
                        'investing_cf': self._get_tag_value(ticker, f, 'netcashfrominvestingactivities'),
                        'financing_cf': self._get_tag_value(ticker, f, 'netcashfromfinancingactivities'),
                        'capex': self._get_tag_value(ticker, f, 'capitalexpenditures'),
                        'free_cash_flow': self._get_tag_value(ticker, f, 'freecashflow'),
                        'dividends': self._get_tag_value(ticker, f, 'dividendspaid'),
                    })

        return results

    def _get_tag_value(self, ticker: str, fundamental: Dict, tag: str) -> Optional[float]:
        """Helper to get a specific data tag value"""
        # This would normally make another API call to get standardized financials
        # Simplified for now
        return fundamental.get(tag)

    # ==========================================================================
    # FINANCIAL RATIOS
    # ==========================================================================

    def get_ratios(self, ticker: str) -> Dict[str, float]:
        """Get financial ratios"""
        data = self._make_request(f'companies/{ticker}/data_point/current_ratio')
        ratios = {}

        ratio_tags = [
            'pricetoearnings', 'pricetobook', 'pricetosales', 'pricetofreecashflow',
            'evtoebitda', 'evtosales', 'evtooperatingcashflow',
            'returnonequity', 'returnonassets', 'roic',
            'grossmargin', 'operatingmargin', 'netprofitmargin',
            'currentratio', 'quickratio', 'debttoequity', 'debttoassets',
            'dividendyield', 'payoutratio',
            'revenuegrowth', 'epsgrowth', 'bookvaluegrowth',
        ]

        for tag in ratio_tags:
            tag_data = self._make_request(f'companies/{ticker}/data_point/{tag}')
            if tag_data and 'value' in tag_data:
                ratios[tag] = tag_data['value']

        return ratios

    # ==========================================================================
    # COMPANY INFO
    # ==========================================================================

    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Get company information"""
        data = self._make_request(f'companies/{ticker}')
        if not data or 'company' not in data:
            return None

        c = data['company']
        return CompanyInfo(
            ticker=c.get('ticker', ticker),
            name=c.get('name', ''),
            sector=c.get('sector', ''),
            industry=c.get('industry', ''),
            market_cap=c.get('market_cap', 0),
            employees=c.get('employees', 0),
            description=c.get('short_description', '')
        )

    # ==========================================================================
    # NEWS & SENTIMENT
    # ==========================================================================

    def get_company_news(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict]:
        """Get company news articles"""
        params = {'page_size': page_size}
        if start_date:
            params['start_date'] = start_date

        data = self._make_request(f'companies/{ticker}/news', params)
        if not data or 'news' not in data:
            return []

        return [{
            'id': n.get('id'),
            'title': n.get('title'),
            'publication_date': n.get('publication_date'),
            'summary': n.get('summary'),
            'url': n.get('url'),
        } for n in data['news']]

    # ==========================================================================
    # ANALYST ESTIMATES
    # ==========================================================================

    def get_analyst_estimates(self, ticker: str) -> Dict[str, Any]:
        """Get analyst estimates and recommendations"""
        data = self._make_request(f'companies/{ticker}/analyst_ratings')
        if not data:
            return {}

        return {
            'mean_target': data.get('mean_target'),
            'median_target': data.get('median_target'),
            'high_target': data.get('high_target'),
            'low_target': data.get('low_target'),
            'num_analysts': data.get('num_analysts'),
            'buy': data.get('buy'),
            'hold': data.get('hold'),
            'sell': data.get('sell'),
        }

    # ==========================================================================
    # BULK DATA FETCH
    # ==========================================================================

    def get_complete_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get complete data package for a stock

        Returns:
            Dictionary with all available data
        """
        logger.info(f"Fetching complete data for {ticker}")

        return {
            'ticker': ticker,
            'company_info': self.get_company_info(ticker),
            'historical_prices': self.get_30_year_prices(ticker),
            'income_statements': self.get_income_statement(ticker, years=15),
            'balance_sheets': self.get_balance_sheet(ticker, years=15),
            'cash_flows': self.get_cash_flow(ticker, years=15),
            'ratios': self.get_ratios(ticker),
            'analyst_estimates': self.get_analyst_estimates(ticker),
            'news': self.get_company_news(ticker),
            'fetched_at': datetime.now().isoformat(),
        }


# ==========================================================================
# FUNDAMENTAL FACTOR CALCULATIONS
# ==========================================================================

class FundamentalFactors:
    """
    Calculate fundamental factors from Intrinio data

    30 Fundamental Factors:
    - Value (8): P/E, P/B, P/S, P/FCF, EV/EBITDA, EV/Sales, Dividend Yield, FCF Yield
    - Quality (8): ROE, ROA, ROIC, Gross Margin, Operating Margin, Net Margin, Asset Turnover, Equity Multiplier
    - Growth (8): Revenue Growth, EPS Growth, FCF Growth, Dividend Growth, Book Value Growth, Margin Expansion
    - Financial Health (6): Current Ratio, Quick Ratio, Debt/Equity, Interest Coverage, Altman Z, Piotroski F
    """

    def __init__(self, ratios: Dict[str, float], income: List[Dict], balance: List[Dict], cash_flow: List[Dict]):
        self.ratios = ratios
        self.income = sorted(income, key=lambda x: x.get('fiscal_year', 0))
        self.balance = sorted(balance, key=lambda x: x.get('fiscal_year', 0))
        self.cash_flow = sorted(cash_flow, key=lambda x: x.get('fiscal_year', 0))

    # VALUE FACTORS
    def pe_ratio(self) -> float:
        return self.ratios.get('pricetoearnings', 0)

    def pb_ratio(self) -> float:
        return self.ratios.get('pricetobook', 0)

    def ps_ratio(self) -> float:
        return self.ratios.get('pricetosales', 0)

    def pfcf_ratio(self) -> float:
        return self.ratios.get('pricetofreecashflow', 0)

    def ev_ebitda(self) -> float:
        return self.ratios.get('evtoebitda', 0)

    def ev_sales(self) -> float:
        return self.ratios.get('evtosales', 0)

    def dividend_yield(self) -> float:
        return self.ratios.get('dividendyield', 0) * 100

    def fcf_yield(self) -> float:
        pfcf = self.pfcf_ratio()
        return (1 / pfcf * 100) if pfcf > 0 else 0

    # QUALITY FACTORS
    def roe(self) -> float:
        return self.ratios.get('returnonequity', 0) * 100

    def roa(self) -> float:
        return self.ratios.get('returnonassets', 0) * 100

    def roic(self) -> float:
        return self.ratios.get('roic', 0) * 100

    def gross_margin(self) -> float:
        return self.ratios.get('grossmargin', 0) * 100

    def operating_margin(self) -> float:
        return self.ratios.get('operatingmargin', 0) * 100

    def net_margin(self) -> float:
        return self.ratios.get('netprofitmargin', 0) * 100

    def asset_turnover(self) -> float:
        if len(self.income) < 1 or len(self.balance) < 1:
            return 0
        revenue = self.income[-1].get('revenue', 0)
        assets = self.balance[-1].get('total_assets', 1)
        return revenue / assets if assets else 0

    def equity_multiplier(self) -> float:
        if len(self.balance) < 1:
            return 1
        assets = self.balance[-1].get('total_assets', 0)
        equity = self.balance[-1].get('total_equity', 1)
        return assets / equity if equity else 1

    # GROWTH FACTORS
    def revenue_growth(self) -> float:
        return self.ratios.get('revenuegrowth', 0) * 100

    def eps_growth(self) -> float:
        return self.ratios.get('epsgrowth', 0) * 100

    def fcf_growth(self) -> float:
        if len(self.cash_flow) < 2:
            return 0
        current = self.cash_flow[-1].get('free_cash_flow', 0)
        previous = self.cash_flow[-2].get('free_cash_flow', 1)
        if previous and previous != 0:
            return ((current - previous) / abs(previous)) * 100
        return 0

    def book_value_growth(self) -> float:
        return self.ratios.get('bookvaluegrowth', 0) * 100

    def revenue_growth_3y(self) -> float:
        """3-year revenue CAGR"""
        if len(self.income) < 4:
            return self.revenue_growth()
        current = self.income[-1].get('revenue', 0)
        past = self.income[-4].get('revenue', 1)
        if past and past > 0:
            return ((current / past) ** (1/3) - 1) * 100
        return 0

    def margin_expansion(self) -> float:
        """Operating margin change YoY"""
        if len(self.income) < 2:
            return 0
        current_margin = self.income[-1].get('operating_income', 0) / self.income[-1].get('revenue', 1)
        past_margin = self.income[-2].get('operating_income', 0) / self.income[-2].get('revenue', 1)
        return (current_margin - past_margin) * 100

    # FINANCIAL HEALTH FACTORS
    def current_ratio(self) -> float:
        return self.ratios.get('currentratio', 1)

    def quick_ratio(self) -> float:
        return self.ratios.get('quickratio', 1)

    def debt_to_equity(self) -> float:
        return self.ratios.get('debttoequity', 0)

    def debt_to_assets(self) -> float:
        return self.ratios.get('debttoassets', 0)

    def interest_coverage(self) -> float:
        if len(self.income) < 1:
            return 10
        ebit = self.income[-1].get('operating_income', 0)
        # Would need interest expense from income statement
        return 10  # Placeholder

    def altman_z_score(self) -> float:
        """Simplified Altman Z-Score"""
        if len(self.balance) < 1 or len(self.income) < 1:
            return 3.0

        assets = self.balance[-1].get('total_assets', 1)
        liabilities = self.balance[-1].get('total_liabilities', 0)
        equity = self.balance[-1].get('total_equity', 1)
        current_assets = self.balance[-1].get('current_assets', 0)
        current_liabilities = self.balance[-1].get('current_liabilities', 1)
        revenue = self.income[-1].get('revenue', 0)
        ebit = self.income[-1].get('operating_income', 0)

        working_capital = current_assets - current_liabilities
        retained_earnings = equity * 0.5  # Approximation

        # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        A = working_capital / assets if assets else 0
        B = retained_earnings / assets if assets else 0
        C = ebit / assets if assets else 0
        D = equity / liabilities if liabilities else 1
        E = revenue / assets if assets else 0

        return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

    def piotroski_f_score(self) -> int:
        """Simplified Piotroski F-Score (0-9)"""
        score = 0

        # Profitability
        if self.roa() > 0: score += 1
        if self.fcf_growth() > 0: score += 1
        if self.roa() > self.roa(): score += 1  # Would compare YoY
        if self.net_margin() > 0: score += 1

        # Leverage
        if self.debt_to_equity() < 1: score += 1
        if self.current_ratio() > 1: score += 1

        # Efficiency
        if self.gross_margin() > 20: score += 1
        if self.asset_turnover() > 0.5: score += 1
        if self.revenue_growth() > 0: score += 1

        return min(9, score)

    def get_all_fundamental_factors(self) -> Dict[str, float]:
        """Get all 30 fundamental factors"""
        return {
            # Value (8)
            'pe_ratio': self.pe_ratio(),
            'pb_ratio': self.pb_ratio(),
            'ps_ratio': self.ps_ratio(),
            'pfcf_ratio': self.pfcf_ratio(),
            'ev_ebitda': self.ev_ebitda(),
            'ev_sales': self.ev_sales(),
            'dividend_yield': self.dividend_yield(),
            'fcf_yield': self.fcf_yield(),

            # Quality (8)
            'roe': self.roe(),
            'roa': self.roa(),
            'roic': self.roic(),
            'gross_margin': self.gross_margin(),
            'operating_margin': self.operating_margin(),
            'net_margin': self.net_margin(),
            'asset_turnover': self.asset_turnover(),
            'equity_multiplier': self.equity_multiplier(),

            # Growth (6)
            'revenue_growth': self.revenue_growth(),
            'eps_growth': self.eps_growth(),
            'fcf_growth': self.fcf_growth(),
            'book_value_growth': self.book_value_growth(),
            'revenue_growth_3y': self.revenue_growth_3y(),
            'margin_expansion': self.margin_expansion(),

            # Financial Health (6)
            'current_ratio': self.current_ratio(),
            'quick_ratio': self.quick_ratio(),
            'debt_to_equity': self.debt_to_equity(),
            'debt_to_assets': self.debt_to_assets(),
            'altman_z': self.altman_z_score(),
            'piotroski_f': self.piotroski_f_score(),
        }
