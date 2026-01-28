"""
Risk Management and Analytics Engine
Comprehensive risk metrics, monitoring, and control systems
Inspired by Medallion Fund's rigorous risk management approach
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as t_dist
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RiskMetricType(Enum):
    RETURN = "return"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    VAR = "var"
    TAIL_RISK = "tail_risk"
    FACTOR_RISK = "factor_risk"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    # Return Metrics
    total_return: float
    annualized_return: float
    cagr: float
    monthly_returns: List[float]
    rolling_returns: Dict[str, float]

    # Volatility Metrics
    volatility: float
    annualized_volatility: float
    downside_volatility: float
    upside_volatility: float
    semi_variance: float
    rolling_volatility: pd.Series

    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    treynor_ratio: Optional[float]
    information_ratio: Optional[float]
    omega_ratio: float
    gain_loss_ratio: float
    sterling_ratio: float
    burke_ratio: float

    # Drawdown Metrics
    max_drawdown: float
    max_drawdown_duration: int
    average_drawdown: float
    current_drawdown: float
    drawdown_series: pd.Series
    underwater_curve: pd.Series

    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    marginal_var: Optional[Dict[str, float]]
    component_var: Optional[Dict[str, float]]

    # Tail Risk
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    tail_ratio: float
    left_tail_ratio: float
    right_tail_ratio: float

    # Factor Risk
    beta: Optional[float]
    alpha: Optional[float]
    r_squared: Optional[float]
    tracking_error: Optional[float]
    active_return: Optional[float]

    # Additional Metrics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    ulcer_index: float
    pain_index: float

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, pd.Series):
                result[key] = value.to_dict()
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


class RiskEngine:
    """
    Comprehensive risk management and analytics engine
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        target_return: float = 0.0,
        benchmark_returns: Optional[pd.Series] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.benchmark_returns = benchmark_returns
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        weights: Optional[Dict[str, float]] = None,
        asset_returns: Optional[pd.DataFrame] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics

        Args:
            returns: Portfolio returns series
            weights: Optional portfolio weights for decomposition
            asset_returns: Optional individual asset returns for decomposition
        """
        returns = returns.dropna()

        # Return Metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = self._annualize_return(returns)
        cagr = self._calculate_cagr(returns)
        monthly_returns = returns.resample('M').apply(lambda x: (1+x).prod()-1).tolist()
        rolling_returns = self._calculate_rolling_returns(returns)

        # Volatility Metrics
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(252)
        downside_returns = returns[returns < self.target_return]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        upside_returns = returns[returns > self.target_return]
        upside_volatility = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
        semi_variance = np.mean(np.minimum(returns - self.target_return, 0) ** 2) * 252
        rolling_volatility = returns.rolling(20).std() * np.sqrt(252)

        # Risk-Adjusted Returns
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        calmar = self._calculate_calmar_ratio(returns)
        omega = self._calculate_omega_ratio(returns)
        gain_loss = self._calculate_gain_loss_ratio(returns)
        sterling = self._calculate_sterling_ratio(returns)
        burke = self._calculate_burke_ratio(returns)

        # Benchmark-relative metrics
        treynor = None
        information_ratio = None
        beta = None
        alpha = None
        r_squared = None
        tracking_error = None
        active_return = None

        if self.benchmark_returns is not None:
            aligned_returns, aligned_benchmark = self._align_series(
                returns, self.benchmark_returns
            )
            beta = self._calculate_beta(aligned_returns, aligned_benchmark)
            alpha = self._calculate_alpha(aligned_returns, aligned_benchmark, beta)
            r_squared = self._calculate_r_squared(aligned_returns, aligned_benchmark)
            tracking_error = self._calculate_tracking_error(aligned_returns, aligned_benchmark)
            active_return = annualized_return - self._annualize_return(aligned_benchmark)
            treynor = self._calculate_treynor_ratio(returns, beta)
            information_ratio = active_return / tracking_error if tracking_error > 0 else None

        # Drawdown Metrics
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
        average_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        current_drawdown = drawdown.iloc[-1]

        # VaR Metrics
        var_95 = self._calculate_var(returns, 0.05)
        var_99 = self._calculate_var(returns, 0.01)
        cvar_95 = self._calculate_cvar(returns, 0.05)
        cvar_99 = self._calculate_cvar(returns, 0.01)

        # Component VaR (if weights and asset returns provided)
        marginal_var = None
        component_var = None
        if weights is not None and asset_returns is not None:
            marginal_var, component_var = self._calculate_component_var(
                weights, asset_returns, 0.05
            )

        # Tail Risk Metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        jb_stat, _ = stats.jarque_bera(returns)
        tail_ratio = self._calculate_tail_ratio(returns)
        left_tail = np.percentile(returns, 5)
        right_tail = np.percentile(returns, 95)
        left_tail_ratio = abs(left_tail / np.median(returns)) if np.median(returns) != 0 else 0
        right_tail_ratio = right_tail / np.median(returns) if np.median(returns) != 0 else 0

        # Win/Loss Metrics
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        profit_factor = abs(winning_days.sum() / losing_days.sum()) if losing_days.sum() != 0 else np.inf
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Ulcer and Pain Index
        ulcer_index = self._calculate_ulcer_index(drawdown)
        pain_index = abs(average_drawdown)

        return RiskMetrics(
            # Returns
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            monthly_returns=monthly_returns,
            rolling_returns=rolling_returns,

            # Volatility
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            downside_volatility=downside_volatility,
            upside_volatility=upside_volatility,
            semi_variance=semi_variance,
            rolling_volatility=rolling_volatility,

            # Risk-Adjusted
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            treynor_ratio=treynor,
            information_ratio=information_ratio,
            omega_ratio=omega,
            gain_loss_ratio=gain_loss,
            sterling_ratio=sterling,
            burke_ratio=burke,

            # Drawdown
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            average_drawdown=average_drawdown,
            current_drawdown=current_drawdown,
            drawdown_series=drawdown,
            underwater_curve=drawdown,

            # VaR
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            marginal_var=marginal_var,
            component_var=component_var,

            # Tail Risk
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_stat=jb_stat,
            tail_ratio=tail_ratio,
            left_tail_ratio=left_tail_ratio,
            right_tail_ratio=right_tail_ratio,

            # Factor Risk
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            tracking_error=tracking_error,
            active_return=active_return,

            # Win/Loss
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            ulcer_index=ulcer_index,
            pain_index=pain_index
        )

    def _annualize_return(self, returns: pd.Series) -> float:
        """Annualize returns"""
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    def _calculate_cagr(self, returns: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    def _calculate_rolling_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate rolling returns for different periods"""
        result = {}

        periods = {
            '1m': 21,
            '3m': 63,
            '6m': 126,
            '1y': 252,
            '3y': 756,
            '5y': 1260
        }

        for name, days in periods.items():
            if len(returns) >= days:
                period_return = (1 + returns.iloc[-days:]).prod() - 1
                result[name] = period_return

        return result

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < self.target_return]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0

        downside_std = downside_returns.std() * np.sqrt(252)
        return (self._annualize_return(returns) - self.risk_free_rate) / downside_std

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        if max_dd == 0:
            return np.inf if self._annualize_return(returns) > 0 else 0

        return self._annualize_return(returns) / max_dd

    def _calculate_treynor_ratio(self, returns: pd.Series, beta: float) -> Optional[float]:
        """Calculate Treynor ratio"""
        if beta is None or beta == 0:
            return None
        return (self._annualize_return(returns) - self.risk_free_rate) / beta

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() == 0:
            return np.inf if gains.sum() > 0 else 0

        return gains.sum() / losses.sum()

    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Calculate average gain to average loss ratio"""
        gains = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0 or losses.mean() == 0:
            return np.inf if len(gains) > 0 and gains.mean() > 0 else 0

        return abs(gains.mean() / losses.mean())

    def _calculate_sterling_ratio(self, returns: pd.Series) -> float:
        """Calculate Sterling ratio"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max

        # Average of worst drawdowns
        worst_drawdowns = drawdown.nsmallest(5).abs().mean()

        if worst_drawdowns == 0:
            return np.inf if self._annualize_return(returns) > 0 else 0

        return (self._annualize_return(returns) - self.risk_free_rate) / worst_drawdowns

    def _calculate_burke_ratio(self, returns: pd.Series) -> float:
        """Calculate Burke ratio"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max

        # Square root of sum of squared drawdowns
        burke_denom = np.sqrt((drawdown ** 2).sum())

        if burke_denom == 0:
            return np.inf if self._annualize_return(returns) > 0 else 0

        return (self._annualize_return(returns) - self.risk_free_rate) / burke_denom

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_drawdown = drawdown < 0
        durations = []
        current_duration = 0

        for is_down in in_drawdown:
            if is_down:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return max(durations) if durations else 0

    def _calculate_var(
        self,
        returns: pd.Series,
        alpha: float,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk

        Args:
            returns: Return series
            alpha: Confidence level (e.g., 0.05 for 95% VaR)
            method: 'historical', 'parametric', or 'cornish_fisher'
        """
        if method == "historical":
            return np.percentile(returns, alpha * 100)
        elif method == "parametric":
            return norm.ppf(alpha, returns.mean(), returns.std())
        elif method == "cornish_fisher":
            # Cornish-Fisher expansion for non-normal distributions
            z = norm.ppf(alpha)
            s = stats.skew(returns)
            k = stats.kurtosis(returns)

            z_cf = (
                z +
                (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * (k - 3) / 24 -
                (2*z**3 - 5*z) * (s**2) / 36
            )

            return returns.mean() + z_cf * returns.std()

        return np.percentile(returns, alpha * 100)

    def _calculate_cvar(self, returns: pd.Series, alpha: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, alpha)
        return returns[returns <= var].mean()

    def _calculate_component_var(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        alpha: float = 0.05
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate Marginal and Component VaR

        Marginal VaR: Change in portfolio VaR for small change in position
        Component VaR: Asset's contribution to portfolio VaR
        """
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])
        aligned_returns = asset_returns[assets]

        # Portfolio returns
        port_returns = aligned_returns @ w

        # Portfolio VaR
        port_var = self._calculate_var(pd.Series(port_returns), alpha)

        # Covariance matrix
        cov = aligned_returns.cov().values

        # Portfolio volatility
        port_vol = np.sqrt(w @ cov @ w)

        # Marginal VaR (partial derivative)
        marginal_var = {}
        for i, asset in enumerate(assets):
            # Approximate by small change
            dw = 0.01
            w_up = w.copy()
            w_up[i] += dw
            w_up = w_up / np.sum(np.abs(w_up))

            port_var_up = self._calculate_var(
                pd.Series(aligned_returns.values @ w_up), alpha
            )
            marginal_var[asset] = (port_var_up - port_var) / dw

        # Component VaR = weight * marginal_var
        component_var = {
            asset: weights[asset] * marginal_var[asset]
            for asset in assets
        }

        return marginal_var, component_var

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (right tail / left tail)"""
        left_tail = abs(np.percentile(returns, 5))
        right_tail = np.percentile(returns, 95)

        if left_tail == 0:
            return np.inf if right_tail > 0 else 0

        return right_tail / left_tail

    def _calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate beta relative to benchmark"""
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 0

        return covariance / benchmark_variance

    def _calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        beta: float
    ) -> float:
        """Calculate Jensen's alpha"""
        excess_return = self._annualize_return(returns) - self.risk_free_rate
        benchmark_excess = self._annualize_return(benchmark_returns) - self.risk_free_rate

        return excess_return - beta * benchmark_excess

    def _calculate_r_squared(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate R-squared (coefficient of determination)"""
        correlation = returns.corr(benchmark_returns)
        return correlation ** 2

    def _calculate_tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate tracking error"""
        excess_returns = returns - benchmark_returns
        return excess_returns.std() * np.sqrt(252)

    def _calculate_ulcer_index(self, drawdown: pd.Series) -> float:
        """Calculate Ulcer Index (quadratic mean of drawdowns)"""
        squared_drawdown = drawdown ** 2
        return np.sqrt(squared_drawdown.mean())

    def _align_series(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Align two series by common index"""
        common_idx = series1.index.intersection(series2.index)
        return series1.loc[common_idx], series2.loc[common_idx]


class RiskMonitor:
    """
    Real-time risk monitoring and alert system
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.20,
        max_var_limit: float = 0.05,
        max_volatility_limit: float = 0.30,
        max_leverage_limit: float = 2.0
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.max_var_limit = max_var_limit
        self.max_volatility_limit = max_volatility_limit
        self.max_leverage_limit = max_leverage_limit
        self.alerts: List[Dict] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_risk_limits(
        self,
        metrics: RiskMetrics,
        leverage: float
    ) -> List[Dict]:
        """
        Check if any risk limits are breached

        Returns list of alerts
        """
        alerts = []

        # Drawdown check
        if abs(metrics.current_drawdown) > self.max_drawdown_limit:
            alerts.append({
                'type': 'DRAWDOWN_BREACH',
                'severity': 'HIGH',
                'message': f'Current drawdown {metrics.current_drawdown:.2%} exceeds limit {self.max_drawdown_limit:.2%}',
                'value': metrics.current_drawdown,
                'limit': self.max_drawdown_limit,
                'timestamp': datetime.now()
            })

        # VaR check
        if abs(metrics.var_95) > self.max_var_limit:
            alerts.append({
                'type': 'VAR_BREACH',
                'severity': 'MEDIUM',
                'message': f'VaR(95%) {metrics.var_95:.2%} exceeds limit {self.max_var_limit:.2%}',
                'value': metrics.var_95,
                'limit': self.max_var_limit,
                'timestamp': datetime.now()
            })

        # Volatility check
        if metrics.annualized_volatility > self.max_volatility_limit:
            alerts.append({
                'type': 'VOLATILITY_BREACH',
                'severity': 'MEDIUM',
                'message': f'Volatility {metrics.annualized_volatility:.2%} exceeds limit {self.max_volatility_limit:.2%}',
                'value': metrics.annualized_volatility,
                'limit': self.max_volatility_limit,
                'timestamp': datetime.now()
            })

        # Leverage check
        if leverage > self.max_leverage_limit:
            alerts.append({
                'type': 'LEVERAGE_BREACH',
                'severity': 'HIGH',
                'message': f'Leverage {leverage:.2f}x exceeds limit {self.max_leverage_limit:.2f}x',
                'value': leverage,
                'limit': self.max_leverage_limit,
                'timestamp': datetime.now()
            })

        # Sharpe ratio degradation warning
        if metrics.sharpe_ratio < 0.5:
            alerts.append({
                'type': 'SHARPE_WARNING',
                'severity': 'LOW',
                'message': f'Sharpe ratio {metrics.sharpe_ratio:.2f} is below acceptable threshold',
                'value': metrics.sharpe_ratio,
                'limit': 0.5,
                'timestamp': datetime.now()
            })

        self.alerts.extend(alerts)
        return alerts

    def calculate_position_limits(
        self,
        portfolio_value: float,
        metrics: RiskMetrics,
        target_var: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate position limits based on risk metrics
        """
        # Kelly-based position sizing
        if metrics.win_rate > 0 and metrics.avg_loss != 0:
            kelly = metrics.win_rate - (1 - metrics.win_rate) / abs(metrics.avg_win / metrics.avg_loss)
            kelly_fraction = max(0, min(kelly * 0.25, 0.25))  # Quarter Kelly, capped
        else:
            kelly_fraction = 0.10

        # VaR-based position limit
        if metrics.var_95 != 0:
            var_position_limit = target_var / abs(metrics.var_95)
        else:
            var_position_limit = 1.0

        # Volatility-based position limit
        if metrics.annualized_volatility > 0:
            vol_position_limit = 0.15 / metrics.annualized_volatility  # Target 15% contribution
        else:
            vol_position_limit = 1.0

        return {
            'kelly_position_limit': kelly_fraction,
            'var_position_limit': min(var_position_limit, 0.20),
            'vol_position_limit': min(vol_position_limit, 0.20),
            'recommended_max_position': min(kelly_fraction, var_position_limit, vol_position_limit, 0.20),
            'portfolio_value': portfolio_value,
            'max_position_value': portfolio_value * min(kelly_fraction, var_position_limit, vol_position_limit, 0.20)
        }


class StressTestEngine:
    """
    Stress testing and scenario analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Pre-defined stress scenarios
        self.scenarios = {
            '2008_financial_crisis': {
                'equity': -0.50,
                'bond': 0.05,
                'commodity': -0.30,
                'reit': -0.60,
                'volatility_multiplier': 3.0
            },
            '2020_covid_crash': {
                'equity': -0.35,
                'bond': 0.08,
                'commodity': -0.40,
                'reit': -0.40,
                'volatility_multiplier': 4.0
            },
            '2022_rate_hikes': {
                'equity': -0.25,
                'bond': -0.15,
                'commodity': 0.20,
                'reit': -0.30,
                'volatility_multiplier': 1.5
            },
            'flash_crash': {
                'equity': -0.10,
                'bond': 0.02,
                'commodity': -0.05,
                'reit': -0.08,
                'volatility_multiplier': 5.0
            },
            'stagflation': {
                'equity': -0.20,
                'bond': -0.10,
                'commodity': 0.30,
                'reit': -0.15,
                'volatility_multiplier': 1.5
            }
        }

    def run_stress_test(
        self,
        weights: Dict[str, float],
        asset_classes: Dict[str, str],
        scenario: str
    ) -> Dict[str, Any]:
        """
        Run stress test for a given scenario

        Args:
            weights: Portfolio weights
            asset_classes: Mapping of assets to asset classes
            scenario: Scenario name

        Returns:
            Stress test results
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        scenario_params = self.scenarios[scenario]

        # Calculate portfolio impact
        portfolio_loss = 0
        asset_impacts = {}

        for asset, weight in weights.items():
            asset_class = asset_classes.get(asset, 'equity')
            shock = scenario_params.get(asset_class, scenario_params.get('equity', -0.20))

            asset_loss = weight * shock
            portfolio_loss += asset_loss
            asset_impacts[asset] = {
                'weight': weight,
                'shock': shock,
                'impact': asset_loss
            }

        return {
            'scenario': scenario,
            'portfolio_loss': portfolio_loss,
            'asset_impacts': asset_impacts,
            'volatility_multiplier': scenario_params['volatility_multiplier'],
            'scenario_description': scenario
        }

    def run_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        n_simulations: int = 10000,
        horizon: int = 252
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio

        Args:
            returns: Historical returns
            weights: Portfolio weights
            n_simulations: Number of simulations
            horizon: Forecast horizon in days

        Returns:
            Monte Carlo simulation results
        """
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])

        # Align returns
        aligned_returns = returns[assets]

        # Calculate mean and covariance
        mu = aligned_returns.mean().values
        cov = aligned_returns.cov().values

        # Cholesky decomposition for correlated random variables
        L = np.linalg.cholesky(cov)

        # Simulate paths
        simulated_returns = np.zeros((n_simulations, horizon))

        for sim in range(n_simulations):
            for t in range(horizon):
                z = np.random.standard_normal(len(assets))
                asset_returns = mu + L @ z
                port_return = asset_returns @ w
                simulated_returns[sim, t] = port_return

        # Calculate final values (starting from 1)
        final_values = np.prod(1 + simulated_returns, axis=1)

        # Calculate statistics
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {
            f'p{p}': np.percentile(final_values, p) - 1
            for p in percentiles
        }

        return {
            'mean_return': np.mean(final_values) - 1,
            'median_return': np.median(final_values) - 1,
            'std_return': np.std(final_values),
            'percentiles': percentile_values,
            'prob_loss': np.mean(final_values < 1),
            'expected_shortfall_5': np.mean(final_values[final_values < np.percentile(final_values, 5)]) - 1,
            'max_return': np.max(final_values) - 1,
            'min_return': np.min(final_values) - 1,
            'n_simulations': n_simulations,
            'horizon_days': horizon
        }
