"""
Fundamental Analysis Metrics Engine
Computes 100+ fundamental metrics for deep financial analysis
Inspired by value investing masters: Graham, Buffett, Greenblatt, Piotroski
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:
    """Container for all computed metrics"""
    symbol: str
    timestamp: datetime
    valuation: Dict[str, float] = field(default_factory=dict)
    profitability: Dict[str, float] = field(default_factory=dict)
    growth: Dict[str, float] = field(default_factory=dict)
    financial_health: Dict[str, float] = field(default_factory=dict)
    efficiency: Dict[str, float] = field(default_factory=dict)
    cash_flow: Dict[str, float] = field(default_factory=dict)
    dividend: Dict[str, float] = field(default_factory=dict)
    per_share: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    composite_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary"""
        result = {"symbol": self.symbol, "timestamp": self.timestamp}
        for category in [
            "valuation", "profitability", "growth", "financial_health",
            "efficiency", "cash_flow", "dividend", "per_share",
            "quality_scores", "composite_scores"
        ]:
            result.update(getattr(self, category))
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame"""
        return pd.DataFrame([self.to_dict()])


class FundamentalMetricsEngine:
    """
    Comprehensive fundamental analysis engine
    Computes all key financial metrics and quality scores
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_all_metrics(
        self,
        fundamentals: List[Any],  # List of FundamentalData objects
        price_data: pd.DataFrame,
        market_data: Optional[Dict] = None
    ) -> MetricsResult:
        """
        Compute all fundamental metrics

        Args:
            fundamentals: Historical fundamental data (quarterly)
            price_data: Historical price data
            market_data: Market-level data (risk-free rate, market returns, etc.)

        Returns:
            MetricsResult with all computed metrics
        """
        if not fundamentals:
            raise ValueError("No fundamental data provided")

        latest = fundamentals[0]  # Most recent quarter
        symbol = latest.symbol

        result = MetricsResult(
            symbol=symbol,
            timestamp=datetime.now()
        )

        # Get current price
        current_price = price_data['close'].iloc[-1] if len(price_data) > 0 else None

        # Compute all metric categories
        result.valuation = self._compute_valuation_metrics(fundamentals, current_price)
        result.profitability = self._compute_profitability_metrics(fundamentals)
        result.growth = self._compute_growth_metrics(fundamentals)
        result.financial_health = self._compute_financial_health_metrics(fundamentals)
        result.efficiency = self._compute_efficiency_metrics(fundamentals)
        result.cash_flow = self._compute_cash_flow_metrics(fundamentals)
        result.dividend = self._compute_dividend_metrics(fundamentals, current_price)
        result.per_share = self._compute_per_share_metrics(fundamentals, current_price)
        result.quality_scores = self._compute_quality_scores(result)
        result.composite_scores = self._compute_composite_scores(result)

        return result

    def _compute_valuation_metrics(
        self,
        fundamentals: List[Any],
        current_price: Optional[float]
    ) -> Dict[str, float]:
        """Compute valuation metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            # Get TTM (Trailing Twelve Months) figures by summing last 4 quarters
            ttm_revenue = self._get_ttm(fundamentals, 'revenue')
            ttm_net_income = self._get_ttm(fundamentals, 'net_income')
            ttm_ebitda = self._get_ttm(fundamentals, 'ebitda')
            ttm_operating_income = self._get_ttm(fundamentals, 'operating_income')
            ttm_fcf = self._get_ttm(fundamentals, 'free_cash_flow')
            ttm_ocf = self._get_ttm(fundamentals, 'operating_cash_flow')

            shares = latest.shares_outstanding or latest.shares_diluted
            book_value = latest.total_equity
            total_debt = latest.total_debt or 0
            cash = latest.cash_and_equivalents or 0

            if current_price and shares:
                market_cap = current_price * shares
                enterprise_value = market_cap + total_debt - cash

                # P/E Ratios
                ttm_eps = ttm_net_income / shares if ttm_net_income else None
                metrics['pe_ratio'] = self._safe_divide(current_price, ttm_eps)

                # Calculate forward P/E using growth estimation
                if ttm_eps and len(fundamentals) >= 8:
                    growth_rate = self._calculate_growth_rate(
                        [f.net_income for f in fundamentals[:8] if f.net_income]
                    )
                    forward_eps = ttm_eps * (1 + (growth_rate or 0))
                    metrics['forward_pe'] = self._safe_divide(current_price, forward_eps)

                # PEG Ratio
                if metrics.get('pe_ratio') and growth_rate:
                    metrics['peg_ratio'] = self._safe_divide(
                        metrics['pe_ratio'], growth_rate * 100
                    )

                # Price to Book
                book_per_share = book_value / shares if book_value else None
                metrics['price_to_book'] = self._safe_divide(current_price, book_per_share)

                # Price to Sales
                revenue_per_share = ttm_revenue / shares if ttm_revenue else None
                metrics['price_to_sales'] = self._safe_divide(current_price, revenue_per_share)

                # Price to Cash Flow
                ocf_per_share = ttm_ocf / shares if ttm_ocf else None
                metrics['price_to_cash_flow'] = self._safe_divide(current_price, ocf_per_share)

                # Price to Free Cash Flow
                fcf_per_share = ttm_fcf / shares if ttm_fcf else None
                metrics['price_to_free_cash_flow'] = self._safe_divide(current_price, fcf_per_share)

                # EV Ratios
                metrics['ev_to_revenue'] = self._safe_divide(enterprise_value, ttm_revenue)
                metrics['ev_to_ebitda'] = self._safe_divide(enterprise_value, ttm_ebitda)
                metrics['ev_to_ebit'] = self._safe_divide(enterprise_value, ttm_operating_income)

                # Absolute values
                metrics['market_cap'] = market_cap
                metrics['enterprise_value'] = enterprise_value

                # Earnings Yield (inverse of P/E)
                metrics['earnings_yield'] = self._safe_divide(ttm_eps, current_price)

                # Book Value Per Share
                metrics['book_value_per_share'] = book_per_share

                # Tangible Book Value
                intangibles = getattr(latest, 'intangible_assets', 0) or 0
                goodwill = getattr(latest, 'goodwill', 0) or 0
                tangible_equity = (book_value or 0) - intangibles - goodwill
                metrics['tangible_book_value'] = tangible_equity
                metrics['tangible_book_per_share'] = self._safe_divide(tangible_equity, shares)

                # Graham Number (sqrt(22.5 * EPS * BVPS))
                if ttm_eps and book_per_share and ttm_eps > 0 and book_per_share > 0:
                    metrics['graham_number'] = np.sqrt(22.5 * ttm_eps * book_per_share)

                # NCAV (Net Current Asset Value) per share - Benjamin Graham
                current_assets = latest.current_assets or 0
                total_liabilities = latest.total_liabilities or 0
                ncav = current_assets - total_liabilities
                metrics['ncav_per_share'] = self._safe_divide(ncav, shares)

        except Exception as e:
            self.logger.error(f"Error computing valuation metrics: {e}")

        return metrics

    def _compute_profitability_metrics(self, fundamentals: List[Any]) -> Dict[str, float]:
        """Compute profitability metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            ttm_revenue = self._get_ttm(fundamentals, 'revenue')
            ttm_gross_profit = self._get_ttm(fundamentals, 'gross_profit')
            ttm_operating_income = self._get_ttm(fundamentals, 'operating_income')
            ttm_net_income = self._get_ttm(fundamentals, 'net_income')
            ttm_ebitda = self._get_ttm(fundamentals, 'ebitda')
            ttm_fcf = self._get_ttm(fundamentals, 'free_cash_flow')

            # Margins
            metrics['gross_margin'] = self._safe_divide(ttm_gross_profit, ttm_revenue)
            metrics['operating_margin'] = self._safe_divide(ttm_operating_income, ttm_revenue)
            metrics['net_margin'] = self._safe_divide(ttm_net_income, ttm_revenue)
            metrics['ebitda_margin'] = self._safe_divide(ttm_ebitda, ttm_revenue)
            metrics['fcf_margin'] = self._safe_divide(ttm_fcf, ttm_revenue)

            # Return on Equity (ROE)
            avg_equity = self._get_average(fundamentals, 'total_equity', periods=4)
            metrics['return_on_equity'] = self._safe_divide(ttm_net_income, avg_equity)

            # Return on Assets (ROA)
            avg_assets = self._get_average(fundamentals, 'total_assets', periods=4)
            metrics['return_on_assets'] = self._safe_divide(ttm_net_income, avg_assets)

            # Return on Invested Capital (ROIC)
            # ROIC = NOPAT / Invested Capital
            # NOPAT = Operating Income * (1 - Tax Rate)
            if ttm_operating_income and latest.total_equity and latest.total_debt:
                tax_rate = self._estimate_tax_rate(fundamentals)
                nopat = ttm_operating_income * (1 - tax_rate)
                invested_capital = (latest.total_equity or 0) + (latest.total_debt or 0) - (latest.cash_and_equivalents or 0)
                metrics['return_on_invested_capital'] = self._safe_divide(nopat, invested_capital)

            # Return on Capital Employed (ROCE)
            # ROCE = EBIT / (Total Assets - Current Liabilities)
            capital_employed = (latest.total_assets or 0) - (latest.current_liabilities or 0)
            metrics['return_on_capital_employed'] = self._safe_divide(ttm_operating_income, capital_employed)

            # Gross Profit to Assets (Novy-Marx Quality Factor)
            metrics['gross_profit_to_assets'] = self._safe_divide(ttm_gross_profit, latest.total_assets)

            # Operating Income to EV (Greenblatt's Earnings Yield)
            # Computed in valuation metrics with EV

            # DuPont Analysis Components
            # ROE = Net Margin * Asset Turnover * Financial Leverage
            if ttm_revenue and avg_assets and avg_equity:
                asset_turnover = ttm_revenue / avg_assets
                financial_leverage = avg_assets / avg_equity
                metrics['dupont_asset_turnover'] = asset_turnover
                metrics['dupont_financial_leverage'] = financial_leverage

        except Exception as e:
            self.logger.error(f"Error computing profitability metrics: {e}")

        return metrics

    def _compute_growth_metrics(self, fundamentals: List[Any]) -> Dict[str, float]:
        """Compute growth metrics"""
        metrics = {}

        try:
            # YoY Growth (compare this quarter to same quarter last year)
            if len(fundamentals) >= 5:
                current = fundamentals[0]
                year_ago = fundamentals[4]

                metrics['revenue_growth_yoy'] = self._calculate_growth(
                    current.revenue, year_ago.revenue
                )
                metrics['earnings_growth_yoy'] = self._calculate_growth(
                    current.net_income, year_ago.net_income
                )
                metrics['operating_income_growth_yoy'] = self._calculate_growth(
                    current.operating_income, year_ago.operating_income
                )
                metrics['fcf_growth_yoy'] = self._calculate_growth(
                    current.free_cash_flow, year_ago.free_cash_flow
                )

            # 3-Year CAGR (12 quarters)
            if len(fundamentals) >= 12:
                metrics['revenue_growth_3y_cagr'] = self._calculate_cagr(
                    [f.revenue for f in fundamentals[:12] if f.revenue],
                    years=3
                )
                metrics['earnings_growth_3y_cagr'] = self._calculate_cagr(
                    [f.net_income for f in fundamentals[:12] if f.net_income],
                    years=3
                )
                metrics['fcf_growth_3y_cagr'] = self._calculate_cagr(
                    [f.free_cash_flow for f in fundamentals[:12] if f.free_cash_flow],
                    years=3
                )

            # 5-Year CAGR (20 quarters) if available
            if len(fundamentals) >= 20:
                metrics['revenue_growth_5y_cagr'] = self._calculate_cagr(
                    [f.revenue for f in fundamentals[:20] if f.revenue],
                    years=5
                )
                metrics['earnings_growth_5y_cagr'] = self._calculate_cagr(
                    [f.net_income for f in fundamentals[:20] if f.net_income],
                    years=5
                )

            # Sustainable Growth Rate = ROE * Retention Ratio
            if len(fundamentals) >= 4:
                ttm_net_income = self._get_ttm(fundamentals, 'net_income')
                ttm_dividends = self._get_ttm(fundamentals, 'dividends_paid') or 0
                avg_equity = self._get_average(fundamentals, 'total_equity', periods=4)

                if ttm_net_income and avg_equity:
                    roe = ttm_net_income / avg_equity
                    retention_ratio = 1 - abs(ttm_dividends) / ttm_net_income if ttm_net_income > 0 else 0
                    metrics['sustainable_growth_rate'] = roe * retention_ratio

            # Internal Growth Rate = ROA * Retention Ratio / (1 - ROA * Retention Ratio)
            if ttm_net_income and self._get_average(fundamentals, 'total_assets', periods=4):
                avg_assets = self._get_average(fundamentals, 'total_assets', periods=4)
                roa = ttm_net_income / avg_assets
                retention = retention_ratio if 'retention_ratio' in dir() else 0.7
                igr = (roa * retention) / (1 - roa * retention) if roa * retention < 1 else None
                metrics['internal_growth_rate'] = igr

        except Exception as e:
            self.logger.error(f"Error computing growth metrics: {e}")

        return metrics

    def _compute_financial_health_metrics(self, fundamentals: List[Any]) -> Dict[str, float]:
        """Compute financial health and leverage metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            current_assets = latest.current_assets or 0
            current_liabilities = latest.current_liabilities or 0
            cash = latest.cash_and_equivalents or 0
            inventory = latest.inventory or 0
            total_debt = latest.total_debt or 0
            total_assets = latest.total_assets or 0
            total_equity = latest.total_equity or 0

            ttm_ebitda = self._get_ttm(fundamentals, 'ebitda')
            ttm_operating_income = self._get_ttm(fundamentals, 'operating_income')
            ttm_interest = abs(self._get_ttm(fundamentals, 'interest_expense') or 0)

            # Liquidity Ratios
            metrics['current_ratio'] = self._safe_divide(current_assets, current_liabilities)
            metrics['quick_ratio'] = self._safe_divide(
                current_assets - inventory, current_liabilities
            )
            metrics['cash_ratio'] = self._safe_divide(cash, current_liabilities)

            # Leverage Ratios
            metrics['debt_to_equity'] = self._safe_divide(total_debt, total_equity)
            metrics['debt_to_assets'] = self._safe_divide(total_debt, total_assets)
            metrics['debt_to_ebitda'] = self._safe_divide(total_debt, ttm_ebitda)
            metrics['net_debt_to_ebitda'] = self._safe_divide(total_debt - cash, ttm_ebitda)

            # Interest Coverage
            metrics['interest_coverage_ratio'] = self._safe_divide(
                ttm_operating_income, ttm_interest
            ) if ttm_interest > 0 else None

            # Financial Leverage (Assets / Equity)
            metrics['financial_leverage'] = self._safe_divide(total_assets, total_equity)
            metrics['equity_multiplier'] = metrics['financial_leverage']

            # Long-term Debt to Capital
            long_term_debt = latest.long_term_debt or 0
            total_capital = total_equity + long_term_debt
            metrics['long_term_debt_to_capital'] = self._safe_divide(long_term_debt, total_capital)

            # Working Capital
            working_capital = current_assets - current_liabilities
            metrics['working_capital'] = working_capital
            metrics['working_capital_ratio'] = self._safe_divide(working_capital, total_assets)

            # Altman Z-Score (for non-financial companies)
            if total_assets > 0 and len(fundamentals) >= 4:
                market_cap = getattr(latest, 'market_cap', None)
                ttm_revenue = self._get_ttm(fundamentals, 'revenue')
                retained_earnings = latest.retained_earnings or 0

                if market_cap and ttm_revenue:
                    a = working_capital / total_assets
                    b = retained_earnings / total_assets
                    c = ttm_operating_income / total_assets if ttm_operating_income else 0
                    d = market_cap / (latest.total_liabilities or 1)
                    e = ttm_revenue / total_assets

                    z_score = 1.2*a + 1.4*b + 3.3*c + 0.6*d + 1.0*e
                    metrics['altman_z_score'] = z_score

            # Piotroski F-Score (9-point quality score)
            metrics['piotroski_f_score'] = self._calculate_piotroski_score(fundamentals)

            # Beneish M-Score (earnings manipulation detection)
            metrics['beneish_m_score'] = self._calculate_beneish_score(fundamentals)

        except Exception as e:
            self.logger.error(f"Error computing financial health metrics: {e}")

        return metrics

    def _compute_efficiency_metrics(self, fundamentals: List[Any]) -> Dict[str, float]:
        """Compute efficiency/turnover metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            ttm_revenue = self._get_ttm(fundamentals, 'revenue')
            ttm_cogs = self._get_ttm(fundamentals, 'cost_of_revenue')

            avg_assets = self._get_average(fundamentals, 'total_assets', periods=4)
            avg_inventory = self._get_average(fundamentals, 'inventory', periods=4)
            avg_receivables = self._get_average(fundamentals, 'accounts_receivable', periods=4)
            avg_payables = self._get_average(fundamentals, 'accounts_payable', periods=4)
            avg_fixed_assets = self._get_average(fundamentals, 'fixed_assets', periods=4)

            # Asset Turnover
            metrics['asset_turnover'] = self._safe_divide(ttm_revenue, avg_assets)

            # Inventory Turnover
            metrics['inventory_turnover'] = self._safe_divide(ttm_cogs, avg_inventory)

            # Days Inventory Outstanding (DIO)
            if metrics.get('inventory_turnover'):
                metrics['days_inventory_outstanding'] = 365 / metrics['inventory_turnover']

            # Receivables Turnover
            metrics['receivables_turnover'] = self._safe_divide(ttm_revenue, avg_receivables)

            # Days Sales Outstanding (DSO)
            if metrics.get('receivables_turnover'):
                metrics['days_sales_outstanding'] = 365 / metrics['receivables_turnover']

            # Payables Turnover
            metrics['payables_turnover'] = self._safe_divide(ttm_cogs, avg_payables)

            # Days Payables Outstanding (DPO)
            if metrics.get('payables_turnover'):
                metrics['days_payables_outstanding'] = 365 / metrics['payables_turnover']

            # Cash Conversion Cycle = DIO + DSO - DPO
            dio = metrics.get('days_inventory_outstanding', 0)
            dso = metrics.get('days_sales_outstanding', 0)
            dpo = metrics.get('days_payables_outstanding', 0)
            if dio and dso:
                metrics['cash_conversion_cycle'] = dio + dso - (dpo or 0)

            # Operating Cycle = DIO + DSO
            if dio and dso:
                metrics['operating_cycle'] = dio + dso

            # Fixed Asset Turnover
            if avg_fixed_assets:
                metrics['fixed_asset_turnover'] = self._safe_divide(ttm_revenue, avg_fixed_assets)

            # Working Capital Turnover
            avg_working_capital = self._get_average_working_capital(fundamentals, periods=4)
            if avg_working_capital:
                metrics['working_capital_turnover'] = self._safe_divide(
                    ttm_revenue, avg_working_capital
                )

            # Capital Intensity (inverse of asset turnover)
            metrics['capital_intensity'] = self._safe_divide(avg_assets, ttm_revenue)

        except Exception as e:
            self.logger.error(f"Error computing efficiency metrics: {e}")

        return metrics

    def _compute_cash_flow_metrics(self, fundamentals: List[Any]) -> Dict[str, float]:
        """Compute cash flow metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            ttm_ocf = self._get_ttm(fundamentals, 'operating_cash_flow')
            ttm_fcf = self._get_ttm(fundamentals, 'free_cash_flow')
            ttm_net_income = self._get_ttm(fundamentals, 'net_income')
            ttm_revenue = self._get_ttm(fundamentals, 'revenue')
            ttm_capex = abs(self._get_ttm(fundamentals, 'capital_expenditures') or 0)

            total_debt = latest.total_debt or 0
            current_liabilities = latest.current_liabilities or 0

            # Operating Cash Flow metrics
            metrics['operating_cash_flow_ttm'] = ttm_ocf

            # Free Cash Flow metrics
            metrics['free_cash_flow_ttm'] = ttm_fcf

            # FCF Yield (computed in valuation with price)

            # Cash Flow Margins
            metrics['cash_flow_margin'] = self._safe_divide(ttm_ocf, ttm_revenue)
            metrics['fcf_margin'] = self._safe_divide(ttm_fcf, ttm_revenue)

            # Cash Flow to Debt
            metrics['cash_flow_to_debt'] = self._safe_divide(ttm_ocf, total_debt)

            # Cash Flow Coverage of Current Liabilities
            metrics['ocf_to_current_liabilities'] = self._safe_divide(ttm_ocf, current_liabilities)

            # CapEx Metrics
            metrics['capex_to_revenue'] = self._safe_divide(ttm_capex, ttm_revenue)

            # CapEx to Depreciation (>1 means growing asset base)
            ttm_depreciation = self._get_ttm(fundamentals, 'depreciation') or 0
            if not ttm_depreciation:
                # Estimate depreciation from difference between EBITDA and EBIT
                ttm_ebitda = self._get_ttm(fundamentals, 'ebitda') or 0
                ttm_ebit = self._get_ttm(fundamentals, 'operating_income') or 0
                ttm_depreciation = ttm_ebitda - ttm_ebit

            metrics['capex_to_depreciation'] = self._safe_divide(ttm_capex, ttm_depreciation)

            # Cash Conversion Ratio (OCF / Net Income)
            # >1 indicates high quality earnings
            metrics['cash_conversion_ratio'] = self._safe_divide(ttm_ocf, ttm_net_income)

            # Quality of Earnings (OCF / Net Income)
            metrics['quality_of_earnings'] = metrics['cash_conversion_ratio']

            # Accruals Ratio (lower is better - less accounting manipulation)
            # Accruals = Net Income - OCF
            if ttm_net_income and ttm_ocf:
                accruals = ttm_net_income - ttm_ocf
                avg_assets = self._get_average(fundamentals, 'total_assets', periods=4)
                metrics['accruals_ratio'] = self._safe_divide(accruals, avg_assets)

        except Exception as e:
            self.logger.error(f"Error computing cash flow metrics: {e}")

        return metrics

    def _compute_dividend_metrics(
        self,
        fundamentals: List[Any],
        current_price: Optional[float]
    ) -> Dict[str, float]:
        """Compute dividend metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            ttm_dividends = abs(self._get_ttm(fundamentals, 'dividends_paid') or 0)
            ttm_net_income = self._get_ttm(fundamentals, 'net_income')
            ttm_fcf = self._get_ttm(fundamentals, 'free_cash_flow')
            shares = latest.shares_outstanding or latest.shares_diluted

            if ttm_dividends and shares:
                dps = ttm_dividends / shares
                metrics['dividends_per_share'] = dps

                if current_price:
                    metrics['dividend_yield'] = dps / current_price

            # Payout Ratios
            if ttm_net_income and ttm_net_income > 0:
                metrics['dividend_payout_ratio'] = self._safe_divide(ttm_dividends, ttm_net_income)
                metrics['retention_ratio'] = 1 - metrics.get('dividend_payout_ratio', 0)

            if ttm_fcf and ttm_fcf > 0:
                metrics['fcf_payout_ratio'] = self._safe_divide(ttm_dividends, ttm_fcf)

            # Dividend Coverage
            if ttm_dividends > 0:
                metrics['dividend_coverage'] = self._safe_divide(ttm_net_income, ttm_dividends)

            # Buyback Yield
            ttm_buybacks = abs(self._get_ttm(fundamentals, 'share_repurchases') or 0)
            market_cap = current_price * shares if current_price and shares else None
            if market_cap:
                metrics['buyback_yield'] = self._safe_divide(ttm_buybacks, market_cap)

            # Shareholder Yield (Dividend Yield + Buyback Yield)
            div_yield = metrics.get('dividend_yield', 0)
            buyback_yield = metrics.get('buyback_yield', 0)
            metrics['shareholder_yield'] = div_yield + buyback_yield

            # Dividend Growth (5-year)
            if len(fundamentals) >= 20:
                old_dividends = sum(
                    abs(f.dividends_paid or 0)
                    for f in fundamentals[16:20]
                )
                recent_dividends = sum(
                    abs(f.dividends_paid or 0)
                    for f in fundamentals[0:4]
                )
                if old_dividends > 0:
                    metrics['dividend_growth_5y'] = self._calculate_cagr_from_values(
                        old_dividends, recent_dividends, 4
                    )

        except Exception as e:
            self.logger.error(f"Error computing dividend metrics: {e}")

        return metrics

    def _compute_per_share_metrics(
        self,
        fundamentals: List[Any],
        current_price: Optional[float]
    ) -> Dict[str, float]:
        """Compute per-share metrics"""
        metrics = {}
        latest = fundamentals[0]

        try:
            shares = latest.shares_outstanding or latest.shares_diluted
            if not shares:
                return metrics

            ttm_revenue = self._get_ttm(fundamentals, 'revenue')
            ttm_net_income = self._get_ttm(fundamentals, 'net_income')
            ttm_ocf = self._get_ttm(fundamentals, 'operating_cash_flow')
            ttm_fcf = self._get_ttm(fundamentals, 'free_cash_flow')
            ttm_ebitda = self._get_ttm(fundamentals, 'ebitda')

            metrics['eps_basic'] = latest.eps_basic
            metrics['eps_diluted'] = latest.eps_diluted
            metrics['eps_ttm'] = self._safe_divide(ttm_net_income, shares)

            metrics['revenue_per_share'] = self._safe_divide(ttm_revenue, shares)
            metrics['book_value_per_share'] = self._safe_divide(latest.total_equity, shares)
            metrics['cash_per_share'] = self._safe_divide(latest.cash_and_equivalents, shares)
            metrics['fcf_per_share'] = self._safe_divide(ttm_fcf, shares)
            metrics['ocf_per_share'] = self._safe_divide(ttm_ocf, shares)
            metrics['ebitda_per_share'] = self._safe_divide(ttm_ebitda, shares)

        except Exception as e:
            self.logger.error(f"Error computing per-share metrics: {e}")

        return metrics

    def _compute_quality_scores(self, result: MetricsResult) -> Dict[str, float]:
        """Compute quality scores based on fundamental metrics"""
        scores = {}

        try:
            # Profitability Quality Score (0-100)
            prof_score = 0
            if result.profitability.get('return_on_equity', 0) > 0.15:
                prof_score += 20
            elif result.profitability.get('return_on_equity', 0) > 0.10:
                prof_score += 10
            if result.profitability.get('return_on_invested_capital', 0) > 0.12:
                prof_score += 20
            if result.profitability.get('gross_margin', 0) > 0.40:
                prof_score += 20
            elif result.profitability.get('gross_margin', 0) > 0.25:
                prof_score += 10
            if result.profitability.get('operating_margin', 0) > 0.15:
                prof_score += 20
            if result.profitability.get('net_margin', 0) > 0.10:
                prof_score += 20
            scores['profitability_quality'] = min(prof_score, 100)

            # Financial Strength Score (0-100)
            fin_score = 0
            if result.financial_health.get('current_ratio', 0) > 1.5:
                fin_score += 20
            if result.financial_health.get('debt_to_equity', float('inf')) < 0.5:
                fin_score += 25
            elif result.financial_health.get('debt_to_equity', float('inf')) < 1.0:
                fin_score += 15
            if result.financial_health.get('interest_coverage_ratio', 0) > 5:
                fin_score += 25
            if (result.financial_health.get('altman_z_score', 0) or 0) > 3:
                fin_score += 15
            if (result.financial_health.get('piotroski_f_score', 0) or 0) >= 7:
                fin_score += 15
            scores['financial_strength'] = min(fin_score, 100)

            # Growth Quality Score (0-100)
            growth_score = 0
            rev_growth = result.growth.get('revenue_growth_3y_cagr', 0) or 0
            earn_growth = result.growth.get('earnings_growth_3y_cagr', 0) or 0
            if rev_growth > 0.15:
                growth_score += 30
            elif rev_growth > 0.08:
                growth_score += 15
            if earn_growth > 0.15:
                growth_score += 30
            elif earn_growth > 0.08:
                growth_score += 15
            if result.growth.get('sustainable_growth_rate', 0) > 0.10:
                growth_score += 20
            # Consistency (earnings growth > revenue growth suggests margin expansion)
            if earn_growth > rev_growth:
                growth_score += 20
            scores['growth_quality'] = min(growth_score, 100)

            # Cash Flow Quality Score (0-100)
            cf_score = 0
            if (result.cash_flow.get('quality_of_earnings', 0) or 0) > 1.0:
                cf_score += 30
            if (result.cash_flow.get('fcf_margin', 0) or 0) > 0.10:
                cf_score += 25
            if (result.cash_flow.get('accruals_ratio', 0) or 0) < 0.05:
                cf_score += 25
            if result.cash_flow.get('free_cash_flow_ttm', 0) > 0:
                cf_score += 20
            scores['cash_flow_quality'] = min(cf_score, 100)

            # Valuation Score (0-100, lower valuation = higher score)
            val_score = 0
            pe = result.valuation.get('pe_ratio', float('inf')) or float('inf')
            pb = result.valuation.get('price_to_book', float('inf')) or float('inf')
            ev_ebitda = result.valuation.get('ev_to_ebitda', float('inf')) or float('inf')

            if pe < 15:
                val_score += 30
            elif pe < 25:
                val_score += 15
            if pb < 2:
                val_score += 25
            elif pb < 4:
                val_score += 10
            if ev_ebitda < 10:
                val_score += 25
            elif ev_ebitda < 15:
                val_score += 10
            if (result.valuation.get('earnings_yield', 0) or 0) > 0.08:
                val_score += 20
            scores['valuation_score'] = min(val_score, 100)

        except Exception as e:
            self.logger.error(f"Error computing quality scores: {e}")

        return scores

    def _compute_composite_scores(self, result: MetricsResult) -> Dict[str, float]:
        """Compute composite investment scores"""
        scores = {}

        try:
            quality_scores = result.quality_scores

            # Overall Quality Score (weighted average)
            weights = {
                'profitability_quality': 0.25,
                'financial_strength': 0.20,
                'growth_quality': 0.20,
                'cash_flow_quality': 0.20,
                'valuation_score': 0.15
            }

            total_score = sum(
                quality_scores.get(k, 0) * v
                for k, v in weights.items()
            )
            scores['overall_quality_score'] = total_score

            # Magic Formula Rank (Greenblatt)
            # Combines earnings yield and ROIC
            earnings_yield = result.valuation.get('earnings_yield', 0) or 0
            roic = result.profitability.get('return_on_invested_capital', 0) or 0
            scores['magic_formula_score'] = (earnings_yield * 50 + roic * 50) * 100

            # Value-Quality Score (Buffett-style)
            # High quality at reasonable price
            val_score = quality_scores.get('valuation_score', 0)
            qual_avg = (
                quality_scores.get('profitability_quality', 0) +
                quality_scores.get('financial_strength', 0) +
                quality_scores.get('cash_flow_quality', 0)
            ) / 3
            scores['value_quality_score'] = (val_score * 0.4 + qual_avg * 0.6)

            # Momentum-Value Score
            # Would need price momentum data - placeholder
            scores['momentum_adjusted_score'] = total_score  # Placeholder

            # Investment Grade (A+ to F)
            if total_score >= 85:
                scores['investment_grade'] = 'A+'
            elif total_score >= 75:
                scores['investment_grade'] = 'A'
            elif total_score >= 65:
                scores['investment_grade'] = 'B+'
            elif total_score >= 55:
                scores['investment_grade'] = 'B'
            elif total_score >= 45:
                scores['investment_grade'] = 'C+'
            elif total_score >= 35:
                scores['investment_grade'] = 'C'
            else:
                scores['investment_grade'] = 'D'

        except Exception as e:
            self.logger.error(f"Error computing composite scores: {e}")

        return scores

    # Helper Methods

    def _get_ttm(self, fundamentals: List[Any], field: str) -> Optional[float]:
        """Get trailing twelve months (sum of last 4 quarters)"""
        values = []
        for f in fundamentals[:4]:
            val = getattr(f, field, None)
            if val is not None:
                values.append(val)

        return sum(values) if len(values) == 4 else None

    def _get_average(
        self,
        fundamentals: List[Any],
        field: str,
        periods: int = 4
    ) -> Optional[float]:
        """Get average of field over periods"""
        values = []
        for f in fundamentals[:periods]:
            val = getattr(f, field, None)
            if val is not None:
                values.append(val)

        return np.mean(values) if values else None

    def _get_average_working_capital(
        self,
        fundamentals: List[Any],
        periods: int = 4
    ) -> Optional[float]:
        """Get average working capital"""
        values = []
        for f in fundamentals[:periods]:
            ca = f.current_assets or 0
            cl = f.current_liabilities or 0
            values.append(ca - cl)

        return np.mean(values) if values else None

    def _safe_divide(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safe division handling None and zero"""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    def _calculate_growth(self, current: Optional[float], previous: Optional[float]) -> Optional[float]:
        """Calculate simple growth rate"""
        if current is None or previous is None or previous == 0:
            return None
        return (current - previous) / abs(previous)

    def _calculate_growth_rate(self, values: List[float]) -> Optional[float]:
        """Calculate growth rate from list of values"""
        if len(values) < 2:
            return None
        return (values[0] - values[-1]) / abs(values[-1]) if values[-1] != 0 else None

    def _calculate_cagr(self, values: List[float], years: int) -> Optional[float]:
        """Calculate CAGR from quarterly values"""
        if len(values) < 2 or values[-1] is None or values[0] is None:
            return None
        if values[-1] <= 0 or values[0] <= 0:
            return None
        return (values[0] / values[-1]) ** (1 / years) - 1

    def _calculate_cagr_from_values(
        self,
        start_value: float,
        end_value: float,
        years: int
    ) -> Optional[float]:
        """Calculate CAGR from start and end values"""
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return None
        return (end_value / start_value) ** (1 / years) - 1

    def _estimate_tax_rate(self, fundamentals: List[Any]) -> float:
        """Estimate effective tax rate"""
        for f in fundamentals[:4]:
            if f.pretax_income and f.income_tax and f.pretax_income > 0:
                rate = f.income_tax / f.pretax_income
                if 0 < rate < 0.5:
                    return rate
        return 0.25  # Default corporate tax rate

    def _calculate_piotroski_score(self, fundamentals: List[Any]) -> int:
        """
        Calculate Piotroski F-Score (0-9)
        Higher is better - indicates financial strength
        """
        if len(fundamentals) < 5:
            return 0

        score = 0
        current = fundamentals[0]
        previous = fundamentals[4]  # Year ago

        # Profitability Signals (4 points)
        # 1. Positive ROA
        if current.net_income and current.total_assets:
            roa = current.net_income / current.total_assets
            if roa > 0:
                score += 1

        # 2. Positive Operating Cash Flow
        if current.operating_cash_flow and current.operating_cash_flow > 0:
            score += 1

        # 3. ROA improving
        if previous.net_income and previous.total_assets and current.net_income and current.total_assets:
            prev_roa = previous.net_income / previous.total_assets
            curr_roa = current.net_income / current.total_assets
            if curr_roa > prev_roa:
                score += 1

        # 4. Quality of earnings (OCF > Net Income)
        if current.operating_cash_flow and current.net_income:
            if current.operating_cash_flow > current.net_income:
                score += 1

        # Leverage Signals (3 points)
        # 5. Decrease in leverage
        if current.total_debt and previous.total_debt and current.total_assets and previous.total_assets:
            curr_leverage = current.total_debt / current.total_assets
            prev_leverage = previous.total_debt / previous.total_assets
            if curr_leverage < prev_leverage:
                score += 1

        # 6. Improvement in current ratio
        if current.current_assets and current.current_liabilities and previous.current_assets and previous.current_liabilities:
            curr_cr = current.current_assets / current.current_liabilities
            prev_cr = previous.current_assets / previous.current_liabilities
            if curr_cr > prev_cr:
                score += 1

        # 7. No new shares issued
        if current.shares_outstanding and previous.shares_outstanding:
            if current.shares_outstanding <= previous.shares_outstanding:
                score += 1

        # Efficiency Signals (2 points)
        # 8. Improving gross margin
        if current.gross_profit and current.revenue and previous.gross_profit and previous.revenue:
            curr_gm = current.gross_profit / current.revenue
            prev_gm = previous.gross_profit / previous.revenue
            if curr_gm > prev_gm:
                score += 1

        # 9. Improving asset turnover
        if current.revenue and current.total_assets and previous.revenue and previous.total_assets:
            curr_at = current.revenue / current.total_assets
            prev_at = previous.revenue / previous.total_assets
            if curr_at > prev_at:
                score += 1

        return score

    def _calculate_beneish_score(self, fundamentals: List[Any]) -> Optional[float]:
        """
        Calculate Beneish M-Score
        Score > -1.78 indicates potential earnings manipulation
        """
        if len(fundamentals) < 5:
            return None

        try:
            current = fundamentals[0]
            previous = fundamentals[4]

            # Days Sales in Receivables Index (DSRI)
            if (current.revenue and current.accounts_receivable and
                previous.revenue and previous.accounts_receivable):
                curr_dsr = current.accounts_receivable / (current.revenue / 365)
                prev_dsr = previous.accounts_receivable / (previous.revenue / 365)
                dsri = curr_dsr / prev_dsr if prev_dsr > 0 else 1

            # Gross Margin Index (GMI)
            if (current.revenue and current.gross_profit and
                previous.revenue and previous.gross_profit):
                curr_gm = current.gross_profit / current.revenue
                prev_gm = previous.gross_profit / previous.revenue
                gmi = prev_gm / curr_gm if curr_gm > 0 else 1

            # Asset Quality Index (AQI)
            if (current.total_assets and current.current_assets and
                previous.total_assets and previous.current_assets):
                curr_aq = 1 - (current.current_assets / current.total_assets)
                prev_aq = 1 - (previous.current_assets / previous.total_assets)
                aqi = curr_aq / prev_aq if prev_aq > 0 else 1

            # Sales Growth Index (SGI)
            if current.revenue and previous.revenue:
                sgi = current.revenue / previous.revenue if previous.revenue > 0 else 1

            # Depreciation Index (DEPI) - simplified
            depi = 1  # Placeholder

            # SGA Index (SGAI)
            sgai = 1  # Placeholder

            # Leverage Index (LVGI)
            if (current.total_debt and current.total_assets and
                previous.total_debt and previous.total_assets):
                curr_lv = current.total_debt / current.total_assets
                prev_lv = previous.total_debt / previous.total_assets
                lvgi = curr_lv / prev_lv if prev_lv > 0 else 1

            # Total Accruals to Total Assets (TATA)
            if current.net_income and current.operating_cash_flow and current.total_assets:
                tata = (current.net_income - current.operating_cash_flow) / current.total_assets
            else:
                tata = 0

            # M-Score calculation
            m_score = (
                -4.84 +
                0.92 * dsri +
                0.528 * gmi +
                0.404 * aqi +
                0.892 * sgi +
                0.115 * depi +
                -0.172 * sgai +
                4.679 * tata +
                -0.327 * lvgi
            )

            return m_score

        except Exception:
            return None
