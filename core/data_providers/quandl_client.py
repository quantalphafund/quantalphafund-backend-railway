"""
Quandl/Nasdaq Data Link API Client
Alternative data and economic indicators

Features:
- Economic indicators (GDP, unemployment, inflation)
- Treasury rates and yield curves
- Commodity prices
- Market sentiment indicators
- Institutional ownership
- Short interest data

Docs: https://data.nasdaq.com/
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# API Configuration
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY', '')
NASDAQ_API_KEY = os.getenv('NASDAQ_DATA_LINK_API_KEY', QUANDL_API_KEY)
QUANDL_BASE_URL = 'https://data.nasdaq.com/api/v3'


@dataclass
class TimeSeriesData:
    date: str
    value: float


@dataclass
class EconomicIndicator:
    name: str
    value: float
    date: str
    period: str
    unit: str


class QuandlClient:
    """
    Quandl/Nasdaq Data Link API Client
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NASDAQ_API_KEY
        if not self.api_key:
            logger.warning("Quandl API key not set. Set QUANDL_API_KEY environment variable.")
        self.session = requests.Session()
        self._cache = {}

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request"""
        if not self.api_key:
            logger.error("Quandl API key not configured")
            return None

        url = f"{QUANDL_BASE_URL}/{endpoint}"
        params = params or {}
        params['api_key'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Quandl API error: {e}")
            return None

    def get_dataset(
        self,
        database_code: str,
        dataset_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[TimeSeriesData]:
        """
        Get time series dataset

        Args:
            database_code: Database code (e.g., 'FRED', 'WIKI')
            dataset_code: Dataset code (e.g., 'GDP', 'AAPL')
            start_date: Start date
            end_date: End date
            limit: Max records

        Returns:
            List of TimeSeriesData
        """
        cache_key = f"{database_code}_{dataset_code}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {'limit': limit}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        data = self._make_request(f'datasets/{database_code}/{dataset_code}.json', params)
        if not data or 'dataset' not in data:
            return []

        dataset = data['dataset']
        column_names = dataset.get('column_names', ['Date', 'Value'])
        values = dataset.get('data', [])

        results = []
        for row in values:
            if len(row) >= 2:
                results.append(TimeSeriesData(
                    date=str(row[0]),
                    value=float(row[1]) if row[1] is not None else 0.0
                ))

        results.sort(key=lambda x: x.date)
        self._cache[cache_key] = results
        return results

    # ==========================================================================
    # ECONOMIC INDICATORS (FRED Database)
    # ==========================================================================

    def get_gdp(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """US GDP (Quarterly)"""
        return self.get_dataset('FRED', 'GDP', start_date)

    def get_gdp_growth(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """US Real GDP Growth Rate"""
        return self.get_dataset('FRED', 'A191RL1Q225SBEA', start_date)

    def get_unemployment_rate(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """US Unemployment Rate"""
        return self.get_dataset('FRED', 'UNRATE', start_date)

    def get_cpi(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Consumer Price Index (Inflation)"""
        return self.get_dataset('FRED', 'CPIAUCSL', start_date)

    def get_core_pce(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Core PCE (Fed's preferred inflation measure)"""
        return self.get_dataset('FRED', 'PCEPILFE', start_date)

    def get_industrial_production(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Industrial Production Index"""
        return self.get_dataset('FRED', 'INDPRO', start_date)

    def get_retail_sales(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Retail Sales"""
        return self.get_dataset('FRED', 'RSAFS', start_date)

    def get_consumer_sentiment(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """University of Michigan Consumer Sentiment"""
        return self.get_dataset('FRED', 'UMCSENT', start_date)

    def get_housing_starts(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Housing Starts"""
        return self.get_dataset('FRED', 'HOUST', start_date)

    def get_pmi(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """ISM Manufacturing PMI"""
        return self.get_dataset('FRED', 'MANEMP', start_date)  # Approximation

    # ==========================================================================
    # TREASURY RATES
    # ==========================================================================

    def get_fed_funds_rate(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Federal Funds Rate"""
        return self.get_dataset('FRED', 'FEDFUNDS', start_date)

    def get_treasury_3m(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """3-Month Treasury Rate"""
        return self.get_dataset('FRED', 'TB3MS', start_date)

    def get_treasury_2y(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """2-Year Treasury Rate"""
        return self.get_dataset('FRED', 'GS2', start_date)

    def get_treasury_10y(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """10-Year Treasury Rate"""
        return self.get_dataset('FRED', 'GS10', start_date)

    def get_treasury_30y(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """30-Year Treasury Rate"""
        return self.get_dataset('FRED', 'GS30', start_date)

    def get_yield_curve_spread(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """10Y-2Y Yield Curve Spread"""
        return self.get_dataset('FRED', 'T10Y2Y', start_date)

    def get_credit_spread(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """High Yield Credit Spread (BAA-AAA)"""
        baa = self.get_dataset('FRED', 'BAA', start_date)
        aaa = self.get_dataset('FRED', 'AAA', start_date)

        # Calculate spread
        spreads = []
        baa_dict = {d.date: d.value for d in baa}
        for aaa_point in aaa:
            if aaa_point.date in baa_dict:
                spreads.append(TimeSeriesData(
                    date=aaa_point.date,
                    value=baa_dict[aaa_point.date] - aaa_point.value
                ))
        return spreads

    # ==========================================================================
    # MARKET INDICATORS
    # ==========================================================================

    def get_vix(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """VIX Volatility Index"""
        return self.get_dataset('FRED', 'VIXCLS', start_date)

    def get_sp500(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """S&P 500 Index"""
        return self.get_dataset('FRED', 'SP500', start_date)

    def get_dollar_index(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """US Dollar Index (Trade Weighted)"""
        return self.get_dataset('FRED', 'DTWEXBGS', start_date)

    # ==========================================================================
    # COMMODITY PRICES
    # ==========================================================================

    def get_gold_price(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Gold Price (London PM Fix)"""
        return self.get_dataset('LBMA', 'GOLD', start_date)

    def get_silver_price(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Silver Price"""
        return self.get_dataset('LBMA', 'SILVER', start_date)

    def get_oil_price(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """WTI Crude Oil Price"""
        return self.get_dataset('FRED', 'DCOILWTICO', start_date)

    def get_copper_price(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Copper Price"""
        return self.get_dataset('ODA', 'PCOPP_USD', start_date)

    # ==========================================================================
    # ALTERNATIVE DATA
    # ==========================================================================

    def get_margin_debt(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """NYSE Margin Debt"""
        return self.get_dataset('FINRA', 'FORF_MARGIN_DEBT', start_date)

    def get_money_supply_m2(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """M2 Money Supply"""
        return self.get_dataset('FRED', 'M2SL', start_date)

    def get_bank_lending(self, start_date: str = '1995-01-01') -> List[TimeSeriesData]:
        """Commercial Bank Lending"""
        return self.get_dataset('FRED', 'TOTLL', start_date)

    # ==========================================================================
    # MACRO FACTORS CALCULATOR
    # ==========================================================================

    def get_current_macro_factors(self) -> Dict[str, float]:
        """
        Get current values of all macro factors

        Returns:
            Dictionary of macro factor values
        """
        factors = {}

        # Get latest values (last data point from each series)
        def get_latest(series: List[TimeSeriesData]) -> float:
            if series:
                return series[-1].value
            return 0.0

        # Economic indicators
        factors['gdp_growth'] = get_latest(self.get_gdp_growth())
        factors['unemployment'] = get_latest(self.get_unemployment_rate())
        factors['inflation_cpi'] = get_latest(self.get_cpi())
        factors['consumer_sentiment'] = get_latest(self.get_consumer_sentiment())
        factors['industrial_production'] = get_latest(self.get_industrial_production())

        # Interest rates
        factors['fed_funds'] = get_latest(self.get_fed_funds_rate())
        factors['treasury_2y'] = get_latest(self.get_treasury_2y())
        factors['treasury_10y'] = get_latest(self.get_treasury_10y())
        factors['yield_curve'] = get_latest(self.get_yield_curve_spread())
        factors['credit_spread'] = get_latest(self.get_credit_spread())

        # Market indicators
        factors['vix'] = get_latest(self.get_vix())
        factors['dollar_index'] = get_latest(self.get_dollar_index())

        # Commodities
        factors['gold'] = get_latest(self.get_gold_price())
        factors['oil'] = get_latest(self.get_oil_price())

        # Liquidity
        factors['m2_money_supply'] = get_latest(self.get_money_supply_m2())

        return factors


# ==========================================================================
# MACRO REGIME CLASSIFIER
# ==========================================================================

class MacroRegimeClassifier:
    """
    Classify current macro regime based on economic indicators
    """

    def __init__(self, quandl_client: QuandlClient):
        self.client = quandl_client
        self.factors = self.client.get_current_macro_factors()

    def get_growth_regime(self) -> str:
        """Classify growth regime"""
        gdp = self.factors.get('gdp_growth', 2)
        pmi = self.factors.get('industrial_production', 50)

        if gdp > 3 and pmi > 55:
            return 'expansion'
        elif gdp > 1 and pmi > 50:
            return 'growth'
        elif gdp > 0:
            return 'slowdown'
        else:
            return 'recession'

    def get_inflation_regime(self) -> str:
        """Classify inflation regime"""
        cpi = self.factors.get('inflation_cpi', 250)
        # Compare to previous year (simplified)
        if cpi > 280:  # High inflation environment
            return 'high_inflation'
        elif cpi > 260:
            return 'moderate_inflation'
        else:
            return 'low_inflation'

    def get_monetary_regime(self) -> str:
        """Classify monetary policy regime"""
        fed_funds = self.factors.get('fed_funds', 5)

        if fed_funds < 1:
            return 'accommodative'
        elif fed_funds < 3:
            return 'neutral'
        elif fed_funds < 5:
            return 'tightening'
        else:
            return 'restrictive'

    def get_risk_regime(self) -> str:
        """Classify risk regime"""
        vix = self.factors.get('vix', 20)
        credit_spread = self.factors.get('credit_spread', 1)

        risk_score = vix * 0.5 + credit_spread * 10

        if risk_score < 15:
            return 'risk_on'
        elif risk_score < 25:
            return 'neutral'
        elif risk_score < 40:
            return 'risk_off'
        else:
            return 'crisis'

    def get_yield_curve_regime(self) -> str:
        """Classify yield curve regime"""
        spread = self.factors.get('yield_curve', 0)

        if spread < -0.5:
            return 'inverted'
        elif spread < 0:
            return 'flat'
        elif spread < 1:
            return 'normal'
        else:
            return 'steep'

    def get_dollar_regime(self) -> str:
        """Classify dollar regime"""
        dxy = self.factors.get('dollar_index', 100)

        if dxy < 95:
            return 'weak'
        elif dxy < 100:
            return 'neutral'
        elif dxy < 105:
            return 'strong'
        else:
            return 'very_strong'

    def get_all_regimes(self) -> Dict[str, str]:
        """Get all regime classifications"""
        return {
            'growth': self.get_growth_regime(),
            'inflation': self.get_inflation_regime(),
            'monetary': self.get_monetary_regime(),
            'risk': self.get_risk_regime(),
            'yield_curve': self.get_yield_curve_regime(),
            'dollar': self.get_dollar_regime(),
        }

    def get_regime_score(self) -> float:
        """
        Calculate overall regime score (0-100)
        Higher = more favorable for equities
        """
        regimes = self.get_all_regimes()
        score = 50

        # Growth positive for stocks
        growth_scores = {'expansion': 20, 'growth': 10, 'slowdown': -5, 'recession': -20}
        score += growth_scores.get(regimes['growth'], 0)

        # High inflation negative
        inflation_scores = {'low_inflation': 5, 'moderate_inflation': 0, 'high_inflation': -10}
        score += inflation_scores.get(regimes['inflation'], 0)

        # Accommodative policy positive
        monetary_scores = {'accommodative': 15, 'neutral': 5, 'tightening': -5, 'restrictive': -15}
        score += monetary_scores.get(regimes['monetary'], 0)

        # Risk-on positive
        risk_scores = {'risk_on': 10, 'neutral': 0, 'risk_off': -10, 'crisis': -25}
        score += risk_scores.get(regimes['risk'], 0)

        # Inverted yield curve negative
        yc_scores = {'steep': 5, 'normal': 0, 'flat': -5, 'inverted': -15}
        score += yc_scores.get(regimes['yield_curve'], 0)

        return np.clip(score, 0, 100)


# ==========================================================================
# SENTIMENT FACTORS FROM MACRO DATA
# ==========================================================================

class MacroSentimentFactors:
    """
    Calculate sentiment factors from macro data
    """

    def __init__(self, quandl_client: QuandlClient):
        self.client = quandl_client

    def fear_greed_index(self) -> float:
        """
        Calculate Fear & Greed index approximation (0-100)
        0 = Extreme Fear, 100 = Extreme Greed
        """
        factors = self.client.get_current_macro_factors()

        # VIX (inverted - low VIX = greed)
        vix = factors.get('vix', 20)
        vix_score = max(0, min(100, (35 - vix) * 4))

        # Yield curve (steep = greed)
        yc = factors.get('yield_curve', 0)
        yc_score = max(0, min(100, (yc + 1) * 30))

        # Consumer sentiment
        sentiment = factors.get('consumer_sentiment', 70)
        sentiment_score = max(0, min(100, sentiment))

        # Credit spread (low = greed)
        credit = factors.get('credit_spread', 1)
        credit_score = max(0, min(100, (3 - credit) * 35))

        # Weighted average
        fear_greed = (
            vix_score * 0.35 +
            yc_score * 0.20 +
            sentiment_score * 0.25 +
            credit_score * 0.20
        )

        return round(fear_greed, 1)

    def market_stress_index(self) -> float:
        """
        Calculate market stress index (0-100)
        0 = Calm, 100 = Extreme Stress
        """
        factors = self.client.get_current_macro_factors()

        vix = factors.get('vix', 20)
        credit = factors.get('credit_spread', 1)

        # VIX contribution
        vix_stress = min(100, vix * 3)

        # Credit spread contribution
        credit_stress = min(100, credit * 30)

        stress = vix_stress * 0.6 + credit_stress * 0.4
        return round(stress, 1)

    def liquidity_index(self) -> float:
        """
        Calculate liquidity conditions (0-100)
        Higher = More liquidity
        """
        factors = self.client.get_current_macro_factors()

        # M2 growth approximation
        m2 = factors.get('m2_money_supply', 20000)

        # Fed funds (lower = more liquidity)
        fed = factors.get('fed_funds', 5)
        fed_liquidity = max(0, min(100, (10 - fed) * 12))

        # Credit spread (lower = more liquidity)
        credit = factors.get('credit_spread', 1)
        credit_liquidity = max(0, min(100, (4 - credit) * 30))

        liquidity = fed_liquidity * 0.5 + credit_liquidity * 0.5
        return round(liquidity, 1)

    def get_all_sentiment_factors(self) -> Dict[str, float]:
        """Get all sentiment-derived factors"""
        return {
            'fear_greed': self.fear_greed_index(),
            'market_stress': self.market_stress_index(),
            'liquidity': self.liquidity_index(),
        }
