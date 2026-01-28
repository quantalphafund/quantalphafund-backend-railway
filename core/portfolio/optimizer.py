"""
Portfolio Optimization Engine
Implements multiple optimization strategies including:
- Mean-Variance Optimization (Markowitz)
- Maximum Sharpe Ratio
- Risk Parity
- Hierarchical Risk Parity (HRP)
- Black-Litterman
- Kelly Criterion
- CVaR Optimization

Designed for high Sharpe ratio portfolio construction inspired by Medallion Fund
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hrp"
    BLACK_LITTERMAN = "black_litterman"
    KELLY_CRITERION = "kelly"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CVAR = "min_cvar"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 0.20  # Maximum 20% in single position
    min_positions: int = 10
    max_positions: int = 50
    max_sector_weight: float = 0.30
    max_country_weight: float = 0.40
    long_only: bool = False  # Allow short positions
    max_leverage: float = 2.0  # Maximum gross exposure
    target_volatility: Optional[float] = None
    max_turnover: Optional[float] = None

@dataclass
class PortfolioResult:
    """Result of portfolio optimization"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    cvar_95: float
    diversification_ratio: float
    effective_n: float  # Effective number of positions
    sector_weights: Dict[str, float]
    country_weights: Dict[str, float]
    leverage: float
    optimization_method: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization engine
    Supports multiple optimization methodologies
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        constraints: Optional[PortfolioConstraints] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or PortfolioConstraints()
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        expected_returns: Optional[pd.Series] = None,
        views: Optional[Dict] = None,
        alpha_scores: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        country_map: Optional[Dict[str, str]] = None
    ) -> PortfolioResult:
        """
        Optimize portfolio using specified method

        Args:
            returns: DataFrame of asset returns (assets as columns)
            method: Optimization method to use
            expected_returns: Optional expected returns (if None, use historical)
            views: Views for Black-Litterman (dict of asset: view)
            alpha_scores: Alpha scores for Kelly sizing
            sector_map: Mapping of assets to sectors
            country_map: Mapping of assets to countries

        Returns:
            PortfolioResult with optimal weights and metrics
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # Calculate covariance matrix (using exponential weighting for recent data emphasis)
        cov_matrix = self._calculate_covariance(returns)

        # Calculate expected returns if not provided
        if expected_returns is None:
            expected_returns = self._calculate_expected_returns(returns, alpha_scores)

        # Select optimization method
        optimization_methods = {
            OptimizationMethod.MEAN_VARIANCE: self._optimize_mean_variance,
            OptimizationMethod.MIN_VARIANCE: self._optimize_min_variance,
            OptimizationMethod.MAX_SHARPE: self._optimize_max_sharpe,
            OptimizationMethod.RISK_PARITY: self._optimize_risk_parity,
            OptimizationMethod.HIERARCHICAL_RISK_PARITY: self._optimize_hrp,
            OptimizationMethod.BLACK_LITTERMAN: lambda er, cov: self._optimize_black_litterman(
                er, cov, views
            ),
            OptimizationMethod.KELLY_CRITERION: lambda er, cov: self._optimize_kelly(
                er, cov, alpha_scores
            ),
            OptimizationMethod.MAX_DIVERSIFICATION: self._optimize_max_diversification,
            OptimizationMethod.MIN_CVAR: lambda er, cov: self._optimize_min_cvar(
                returns, er
            ),
            OptimizationMethod.EQUAL_WEIGHT: self._optimize_equal_weight,
        }

        # Run optimization
        weights = optimization_methods[method](expected_returns, cov_matrix)

        # Apply constraints
        weights = self._apply_constraints(
            weights, assets, sector_map, country_map
        )

        # Calculate portfolio metrics
        result = self._calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix, returns,
            method, sector_map, country_map
        )

        return result

    def _calculate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = "exponential"
    ) -> np.ndarray:
        """
        Calculate covariance matrix with options for different estimators
        """
        if method == "exponential":
            # Exponentially weighted covariance (more weight on recent data)
            span = min(60, len(returns) // 2)
            cov = returns.ewm(span=span).cov().iloc[-len(returns.columns):]
            return cov.values
        elif method == "shrinkage":
            # Ledoit-Wolf shrinkage estimator
            return self._ledoit_wolf_shrinkage(returns)
        else:
            # Standard sample covariance
            return returns.cov().values

    def _ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance estimator"""
        X = returns.values
        n, p = X.shape

        # Sample covariance
        sample_cov = np.cov(X.T)

        # Shrinkage target (scaled identity)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)

        # Calculate optimal shrinkage intensity
        delta = sample_cov - target
        delta_sq = delta @ delta

        # Shrinkage intensity calculation
        X_centered = X - X.mean(axis=0)
        sum_sq = 0
        for i in range(n):
            xi = X_centered[i].reshape(-1, 1)
            sum_sq += np.sum((xi @ xi.T - sample_cov) ** 2)

        gamma = sum_sq / (n ** 2)
        kappa = (np.sum(delta_sq) - gamma) / n

        shrinkage = max(0, min(1, kappa / np.sum(delta_sq))) if np.sum(delta_sq) > 0 else 0

        return shrinkage * target + (1 - shrinkage) * sample_cov

    def _calculate_expected_returns(
        self,
        returns: pd.DataFrame,
        alpha_scores: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate expected returns using multiple methods
        """
        # Historical mean (annualized)
        historical_mean = returns.mean() * 252

        # Exponentially weighted mean (more weight on recent)
        ewm_mean = returns.ewm(span=60).mean().iloc[-1] * 252

        # Combine methods
        expected = 0.5 * historical_mean + 0.5 * ewm_mean

        # Adjust with alpha scores if provided
        if alpha_scores is not None:
            # Scale alpha scores to reasonable expected return adjustment
            alpha_adjustment = alpha_scores * 0.10  # +-10% adjustment
            expected = expected + alpha_adjustment.reindex(expected.index).fillna(0)

        return expected

    def _optimize_mean_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        target_return: Optional[float] = None
    ) -> pd.Series:
        """Mean-variance optimization (Markowitz)"""
        n = len(expected_returns)
        init_weights = np.ones(n) / n

        def portfolio_variance(w):
            return w @ cov_matrix @ w

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns.values - target_return
            })

        bounds = self._get_weight_bounds(n)

        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = pd.Series(result.x, index=expected_returns.index)
        return self._clean_weights(weights)

    def _optimize_min_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray
    ) -> pd.Series:
        """Minimum variance portfolio"""
        return self._optimize_mean_variance(expected_returns, cov_matrix, target_return=None)

    def _optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray
    ) -> pd.Series:
        """Maximum Sharpe ratio optimization"""
        n = len(expected_returns)
        init_weights = np.ones(n) / n

        def neg_sharpe(w):
            port_return = w @ expected_returns.values
            port_vol = np.sqrt(w @ cov_matrix @ w)
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = self._get_weight_bounds(n)

        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = pd.Series(result.x, index=expected_returns.index)
        return self._clean_weights(weights)

    def _optimize_risk_parity(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray
    ) -> pd.Series:
        """Risk parity optimization - equal risk contribution"""
        n = len(expected_returns)
        init_weights = np.ones(n) / n

        def risk_parity_objective(w):
            port_var = w @ cov_matrix @ w
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib

            # Target: equal risk contribution
            target_risk = port_var / n
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0.01, self.constraints.max_weight) for _ in range(n)]

        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = pd.Series(result.x, index=expected_returns.index)
        return self._clean_weights(weights)

    def _optimize_hrp(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray
    ) -> pd.Series:
        """
        Hierarchical Risk Parity (HRP)
        Uses hierarchical clustering for more robust allocation
        """
        n = len(expected_returns)
        assets = expected_returns.index.tolist()

        # Calculate correlation matrix
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std, std)
        corr_matrix = np.clip(corr_matrix, -1, 1)

        # Distance matrix
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method='single')

        # Sort assets by cluster order
        sorted_idx = self._get_quasi_diag(linkage_matrix)
        sorted_assets = [assets[i] for i in sorted_idx]

        # Recursive bisection for weights
        weights = self._recursive_bisection(
            cov_matrix, sorted_idx
        )

        weights = pd.Series(weights, index=sorted_assets)
        return self._clean_weights(weights.reindex(assets))

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal matrix ordering from hierarchical clustering"""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df1])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])

        return sort_idx.tolist()

    def _recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sorted_idx: List[int]
    ) -> np.ndarray:
        """Recursive bisection for HRP weights"""
        n = len(sorted_idx)
        weights = np.ones(n)
        cluster_items = [sorted_idx]

        while len(cluster_items) > 0:
            cluster_items = [
                item[start:end]
                for item in cluster_items
                for start, end in ((0, len(item) // 2), (len(item) // 2, len(item)))
                if len(item) > 1
            ]

            for i in range(0, len(cluster_items), 2):
                if i + 1 < len(cluster_items):
                    left_cluster = cluster_items[i]
                    right_cluster = cluster_items[i + 1]

                    left_var = self._cluster_variance(cov_matrix, left_cluster)
                    right_var = self._cluster_variance(cov_matrix, right_cluster)

                    alloc_factor = 1 - left_var / (left_var + right_var)

                    weights[left_cluster] *= alloc_factor
                    weights[right_cluster] *= 1 - alloc_factor

        return weights

    def _cluster_variance(
        self,
        cov_matrix: np.ndarray,
        cluster_items: List[int]
    ) -> float:
        """Calculate variance of a cluster"""
        cov_slice = cov_matrix[np.ix_(cluster_items, cluster_items)]
        w = np.ones(len(cluster_items)) / len(cluster_items)
        return w @ cov_slice @ w

    def _optimize_black_litterman(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        views: Optional[Dict] = None
    ) -> pd.Series:
        """
        Black-Litterman model with investor views
        """
        n = len(expected_returns)
        assets = expected_returns.index.tolist()

        # Risk aversion parameter
        delta = (expected_returns.mean() - self.risk_free_rate) / np.trace(cov_matrix)

        # Implied equilibrium returns
        implied_returns = delta * cov_matrix @ np.ones(n) / n

        if views is None or len(views) == 0:
            # No views - use implied equilibrium returns
            bl_returns = pd.Series(implied_returns, index=assets)
        else:
            # Incorporate views
            P = np.zeros((len(views), n))  # Pick matrix
            Q = np.zeros(len(views))  # View returns

            for i, (asset, view) in enumerate(views.items()):
                if asset in assets:
                    P[i, assets.index(asset)] = 1
                    Q[i] = view

            # Uncertainty of views (tau)
            tau = 0.05

            # Omega - uncertainty of views
            omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

            # Black-Litterman formula
            tau_cov = tau * cov_matrix
            M_inverse = np.linalg.inv(
                np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P
            )
            bl_returns = M_inverse @ (
                np.linalg.inv(tau_cov) @ implied_returns +
                P.T @ np.linalg.inv(omega) @ Q
            )
            bl_returns = pd.Series(bl_returns, index=assets)

        # Optimize with BL returns
        return self._optimize_max_sharpe(bl_returns, cov_matrix)

    def _optimize_kelly(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        alpha_scores: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Kelly criterion portfolio optimization
        Maximizes expected log utility
        """
        n = len(expected_returns)

        # Kelly weights = Σ^(-1) * (μ - r)
        excess_returns = expected_returns.values - self.risk_free_rate

        try:
            cov_inv = np.linalg.inv(cov_matrix)
            kelly_weights = cov_inv @ excess_returns
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            cov_inv = np.linalg.pinv(cov_matrix)
            kelly_weights = cov_inv @ excess_returns

        # Scale to sum to 1 (or target leverage)
        kelly_weights = kelly_weights / np.sum(np.abs(kelly_weights))

        # Apply fractional Kelly (safer)
        kelly_fraction = 0.25  # Quarter Kelly
        kelly_weights = kelly_weights * kelly_fraction

        # Renormalize
        if np.sum(np.abs(kelly_weights)) > 0:
            kelly_weights = kelly_weights / np.sum(np.abs(kelly_weights))

        weights = pd.Series(kelly_weights, index=expected_returns.index)
        return self._clean_weights(weights)

    def _optimize_max_diversification(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray
    ) -> pd.Series:
        """Maximum diversification portfolio"""
        n = len(expected_returns)
        init_weights = np.ones(n) / n
        std = np.sqrt(np.diag(cov_matrix))

        def neg_diversification_ratio(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            weighted_vol = w @ std
            return -weighted_vol / port_vol if port_vol > 0 else 0

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = self._get_weight_bounds(n)

        result = minimize(
            neg_diversification_ratio,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = pd.Series(result.x, index=expected_returns.index)
        return self._clean_weights(weights)

    def _optimize_min_cvar(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        alpha: float = 0.05
    ) -> pd.Series:
        """
        Minimum Conditional Value at Risk (CVaR) optimization
        """
        n = len(expected_returns)
        T = len(returns)

        # Variables: weights (n) + VaR (1) + auxiliary (T)
        init_x = np.zeros(n + 1 + T)
        init_x[:n] = np.ones(n) / n

        def cvar_objective(x):
            weights = x[:n]
            var = x[n]
            u = x[n+1:]
            return var + np.sum(u) / (alpha * T)

        # Portfolio returns constraint
        port_returns = returns.values @ np.eye(n)

        def constraint_weights(x):
            return np.sum(x[:n]) - 1

        def constraint_cvar(x, t):
            weights = x[:n]
            var = x[n]
            u_t = x[n + 1 + t]
            port_ret = returns.iloc[t].values @ weights
            return u_t + port_ret + var

        constraints = [{'type': 'eq', 'fun': constraint_weights}]
        for t in range(T):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: constraint_cvar(x, t)
            })

        bounds = (
            [(self.constraints.min_weight, self.constraints.max_weight) for _ in range(n)] +
            [(None, None)] +  # VaR
            [(0, None) for _ in range(T)]  # Auxiliary vars
        )

        result = minimize(
            cvar_objective,
            init_x,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = pd.Series(result.x[:n], index=expected_returns.index)
        return self._clean_weights(weights)

    def _optimize_equal_weight(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray
    ) -> pd.Series:
        """Simple equal weight portfolio"""
        n = len(expected_returns)
        weights = pd.Series(np.ones(n) / n, index=expected_returns.index)
        return weights

    def _get_weight_bounds(self, n: int) -> List[Tuple[float, float]]:
        """Get weight bounds based on constraints"""
        if self.constraints.long_only:
            return [(self.constraints.min_weight, self.constraints.max_weight) for _ in range(n)]
        else:
            # Allow short positions
            return [(-self.constraints.max_weight, self.constraints.max_weight) for _ in range(n)]

    def _clean_weights(self, weights: pd.Series, threshold: float = 0.001) -> pd.Series:
        """Clean up weights - remove very small positions"""
        weights[np.abs(weights) < threshold] = 0
        # Renormalize
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        return weights

    def _apply_constraints(
        self,
        weights: pd.Series,
        assets: List[str],
        sector_map: Optional[Dict[str, str]] = None,
        country_map: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """Apply portfolio constraints"""
        # Clip individual weights
        weights = weights.clip(
            -self.constraints.max_weight if not self.constraints.long_only else 0,
            self.constraints.max_weight
        )

        # Apply sector constraints
        if sector_map and self.constraints.max_sector_weight:
            weights = self._apply_sector_constraint(
                weights, sector_map, self.constraints.max_sector_weight
            )

        # Apply country constraints
        if country_map and self.constraints.max_country_weight:
            weights = self._apply_country_constraint(
                weights, country_map, self.constraints.max_country_weight
            )

        # Apply leverage constraint
        gross_exposure = np.sum(np.abs(weights))
        if gross_exposure > self.constraints.max_leverage:
            weights = weights * self.constraints.max_leverage / gross_exposure

        # Renormalize
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))

        return weights

    def _apply_sector_constraint(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        max_sector_weight: float
    ) -> pd.Series:
        """Apply maximum sector weight constraint"""
        sector_weights = {}
        for asset, weight in weights.items():
            sector = sector_map.get(asset, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)

        for sector, total_weight in sector_weights.items():
            if total_weight > max_sector_weight:
                scale = max_sector_weight / total_weight
                for asset, weight in weights.items():
                    if sector_map.get(asset, 'Other') == sector:
                        weights[asset] = weight * scale

        return weights

    def _apply_country_constraint(
        self,
        weights: pd.Series,
        country_map: Dict[str, str],
        max_country_weight: float
    ) -> pd.Series:
        """Apply maximum country weight constraint"""
        country_weights = {}
        for asset, weight in weights.items():
            country = country_map.get(asset, 'Other')
            country_weights[country] = country_weights.get(country, 0) + abs(weight)

        for country, total_weight in country_weights.items():
            if total_weight > max_country_weight:
                scale = max_country_weight / total_weight
                for asset, weight in weights.items():
                    if country_map.get(asset, 'Other') == country:
                        weights[asset] = weight * scale

        return weights

    def _calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        returns: pd.DataFrame,
        method: OptimizationMethod,
        sector_map: Optional[Dict[str, str]] = None,
        country_map: Optional[Dict[str, str]] = None
    ) -> PortfolioResult:
        """Calculate all portfolio metrics"""
        w = weights.values

        # Expected return
        port_return = w @ expected_returns.values

        # Volatility
        port_vol = np.sqrt(w @ cov_matrix @ w)

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Calculate historical portfolio returns for drawdown and CVaR
        port_returns = returns @ weights
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # CVaR (95%)
        sorted_returns = port_returns.sort_values()
        var_95_idx = int(0.05 * len(sorted_returns))
        cvar_95 = sorted_returns.iloc[:var_95_idx].mean() if var_95_idx > 0 else sorted_returns.iloc[0]

        # Diversification ratio
        std = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.abs(w) @ std
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1

        # Effective N (number of positions)
        effective_n = 1 / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0

        # Sector weights
        sector_weights = {}
        if sector_map:
            for asset, weight in weights.items():
                sector = sector_map.get(asset, 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Country weights
        country_weights = {}
        if country_map:
            for asset, weight in weights.items():
                country = country_map.get(asset, 'Other')
                country_weights[country] = country_weights.get(country, 0) + weight

        # Leverage (gross exposure)
        leverage = np.sum(np.abs(w))

        return PortfolioResult(
            weights=weights.to_dict(),
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            cvar_95=cvar_95,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            sector_weights=sector_weights,
            country_weights=country_weights,
            leverage=leverage,
            optimization_method=method.value,
            timestamp=datetime.now(),
            metadata={
                'n_assets': len(weights),
                'n_long': sum(1 for w in weights if w > 0),
                'n_short': sum(1 for w in weights if w < 0),
                'risk_free_rate': self.risk_free_rate
            }
        )


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine
    Handles transaction costs and tax optimization
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,  # 10 bps
        tax_rate_short: float = 0.37,
        tax_rate_long: float = 0.20
    ):
        self.transaction_cost = transaction_cost
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long

    def calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        cost_basis: Optional[Dict[str, float]] = None,
        holding_periods: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate trades needed to rebalance portfolio

        Returns trades, costs, and tax implications
        """
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        trades = {}
        total_cost = 0
        total_tax = 0

        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            trade_weight = target - current

            if abs(trade_weight) > 0.001:  # Minimum trade threshold
                trade_value = trade_weight * portfolio_value
                trades[asset] = {
                    'weight_change': trade_weight,
                    'value': trade_value,
                    'action': 'buy' if trade_weight > 0 else 'sell'
                }

                # Transaction cost
                trade_cost = abs(trade_value) * self.transaction_cost
                total_cost += trade_cost

                # Tax implications for sells
                if trade_weight < 0 and cost_basis and holding_periods:
                    basis = cost_basis.get(asset, 0)
                    current_value = current * portfolio_value
                    gain = current_value - basis

                    if gain > 0:
                        days_held = holding_periods.get(asset, 0)
                        tax_rate = self.tax_rate_long if days_held > 365 else self.tax_rate_short
                        tax = gain * abs(trade_weight / current) * tax_rate
                        total_tax += tax
                        trades[asset]['tax_impact'] = tax

        return {
            'trades': trades,
            'total_transaction_cost': total_cost,
            'total_tax_impact': total_tax,
            'total_cost': total_cost + total_tax,
            'turnover': sum(abs(t['weight_change']) for t in trades.values()) / 2
        }
