"""
Unified Data Pipeline
Combines Intrinio, Quandl, and Technical Factors into a single 100+ factor prediction engine

This is the main orchestrator that:
1. Fetches 30 years of historical prices from Intrinio
2. Calculates 60+ technical factors
3. Fetches fundamentals and calculates 30 fundamental factors
4. Fetches macro data and calculates 15 macro/sentiment factors
5. Combines everything into a unified prediction

Total: 105+ factors for ML predictions
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .intrinio_client import IntrinioClient, FundamentalFactors
from .quandl_client import QuandlClient, MacroRegimeClassifier, MacroSentimentFactors
from ..ml_training.factor_engine import FactorEngine, MacroFactors, calculate_composite_score

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Complete prediction result with all factors"""
    symbol: str
    prediction_1m: float
    prediction_3m: float
    prediction_6m: float
    prediction_12m: float
    confidence: float
    direction_accuracy: float
    composite_score: float
    signal: str  # BUY, SELL, HOLD
    factors_count: int
    technical_factors: Dict[str, float]
    fundamental_factors: Dict[str, float]
    macro_factors: Dict[str, float]
    regime: Dict[str, str]


class DataPipeline:
    """
    Unified data pipeline for 100+ factor predictions

    Integrates:
    - Intrinio: Historical prices, fundamentals
    - Quandl: Macro data, economic indicators
    - Factor Engine: Technical analysis
    """

    def __init__(
        self,
        intrinio_api_key: Optional[str] = None,
        quandl_api_key: Optional[str] = None
    ):
        self.intrinio = IntrinioClient(api_key=intrinio_api_key)
        self.quandl = QuandlClient(api_key=quandl_api_key)
        self._cache = {}
        self._macro_cache = None
        self._macro_cache_time = None

    def get_historical_prices_array(self, symbol: str, years: int = 30) -> np.ndarray:
        """
        Get historical prices as numpy array

        Args:
            symbol: Stock ticker
            years: Years of history

        Returns:
            Numpy array of monthly closing prices
        """
        cache_key = f"prices_{symbol}_{years}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
        prices = self.intrinio.get_historical_prices(
            symbol,
            start_date=start_date,
            frequency='monthly'
        )

        if not prices:
            logger.warning(f"No price data for {symbol}")
            return np.array([])

        price_array = np.array([p.adj_close for p in prices])
        self._cache[cache_key] = price_array
        return price_array

    def get_technical_factors(self, symbol: str) -> Dict[str, float]:
        """Calculate all technical factors for a symbol"""
        prices = self.get_historical_prices_array(symbol)
        if len(prices) < 24:
            logger.warning(f"Insufficient price data for {symbol}")
            return {}

        engine = FactorEngine(prices)
        return engine.get_all_factors()

    def get_fundamental_factors(self, symbol: str) -> Dict[str, float]:
        """Get all fundamental factors for a symbol"""
        cache_key = f"fundamentals_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch data from Intrinio
        ratios = self.intrinio.get_ratios(symbol)
        income = self.intrinio.get_income_statement(symbol, years=10)
        balance = self.intrinio.get_balance_sheet(symbol, years=10)
        cash_flow = self.intrinio.get_cash_flow(symbol, years=10)

        if not ratios:
            logger.warning(f"No fundamental data for {symbol}")
            return {}

        factors = FundamentalFactors(ratios, income, balance, cash_flow)
        result = factors.get_all_fundamental_factors()

        self._cache[cache_key] = result
        return result

    def get_macro_factors(self) -> Dict[str, float]:
        """Get current macro factors (cached for 1 hour)"""
        now = datetime.now()
        if (self._macro_cache is not None and
            self._macro_cache_time is not None and
            (now - self._macro_cache_time).seconds < 3600):
            return self._macro_cache

        # Fetch from Quandl
        macro_data = self.quandl.get_current_macro_factors()

        # Add regime classification
        classifier = MacroRegimeClassifier(self.quandl)
        regimes = classifier.get_all_regimes()
        regime_score = classifier.get_regime_score()

        # Add sentiment factors
        sentiment = MacroSentimentFactors(self.quandl)
        sentiment_factors = sentiment.get_all_sentiment_factors()

        # Combine all macro factors
        result = {
            **macro_data,
            'regime_score': regime_score,
            **{f'regime_{k}': 1 if v == 'expansion' or v == 'risk_on' else 0
               for k, v in regimes.items()},
            **sentiment_factors,
        }

        self._macro_cache = result
        self._macro_cache_time = now
        return result

    def get_all_factors(self, symbol: str) -> Tuple[Dict[str, float], int]:
        """
        Get all 100+ factors for a symbol

        Returns:
            Tuple of (factors dict, total factor count)
        """
        all_factors = {}

        # Technical factors (60)
        technical = self.get_technical_factors(symbol)
        for k, v in technical.items():
            all_factors[f'tech_{k}'] = v

        # Fundamental factors (30)
        fundamental = self.get_fundamental_factors(symbol)
        for k, v in fundamental.items():
            all_factors[f'fund_{k}'] = v

        # Macro factors (15)
        macro = self.get_macro_factors()
        for k, v in macro.items():
            if isinstance(v, (int, float)):
                all_factors[f'macro_{k}'] = v

        return all_factors, len(all_factors)

    def backtest_prediction(
        self,
        symbol: str,
        lookback_months: int = 36
    ) -> Tuple[float, float]:
        """
        Backtest predictions using walk-forward validation

        Returns:
            Tuple of (direction_accuracy, mean_absolute_error)
        """
        prices = self.get_historical_prices_array(symbol)
        if len(prices) < lookback_months + 12:
            return 50.0, 10.0

        correct = 0
        total = 0
        errors = []

        # Walk-forward validation
        for t in range(len(prices) - lookback_months - 1, len(prices) - 1):
            # Get factors at time t
            historical_prices = prices[:t+1]
            if len(historical_prices) < 24:
                continue

            engine = FactorEngine(historical_prices)
            factors = engine.get_all_factors()

            # Simple prediction based on key factors
            pred = (
                factors.get('momentum_3m', 0) * 0.3 +
                factors.get('ma_cross_5_20', 0) * 0.2 +
                factors.get('rsi', 50) - 50 * 0.1 +
                factors.get('trend_intensity', 0) * 0.2
            ) / 10

            # Actual return
            actual = (prices[t+1] - prices[t]) / prices[t] * 100

            # Check direction
            if (pred > 0 and actual > 0) or (pred <= 0 and actual <= 0):
                correct += 1
            total += 1
            errors.append(abs(pred - actual))

        accuracy = (correct / total * 100) if total > 0 else 50.0
        mae = np.mean(errors) if errors else 10.0

        return accuracy, mae

    def generate_prediction(self, symbol: str) -> PredictionResult:
        """
        Generate complete prediction with all factors

        Returns:
            PredictionResult with predictions and all factors
        """
        # Get all factors
        all_factors, factor_count = self.get_all_factors(symbol)

        # Get individual factor groups
        technical = self.get_technical_factors(symbol)
        fundamental = self.get_fundamental_factors(symbol)
        macro = self.get_macro_factors()

        # Calculate composite score
        composite = calculate_composite_score(technical)

        # Backtest for accuracy
        accuracy, mae = self.backtest_prediction(symbol)

        # Generate predictions
        # Multi-factor model prediction
        momentum_signal = (
            technical.get('momentum_3m', 0) * 0.3 +
            technical.get('momentum_12m_skip_1m', 0) * 0.2 +
            technical.get('momentum_quality', 0) * 0.1
        )

        trend_signal = (
            technical.get('trend_intensity', 0) * 0.2 +
            technical.get('ma_alignment', 50) - 50 * 0.1 +
            technical.get('breakout', 0) * 0.1
        )

        mean_rev_signal = technical.get('mean_rev_signal', 0) * 0.1

        value_signal = 0
        if fundamental:
            pe = fundamental.get('pe_ratio', 20)
            if pe > 0 and pe < 15:
                value_signal = 10
            elif pe > 30:
                value_signal = -5

        macro_signal = (macro.get('regime_score', 50) - 50) * 0.1

        # Combine signals
        pred_1m = (momentum_signal + trend_signal + mean_rev_signal +
                   value_signal + macro_signal) / 5

        # Scale to longer timeframes
        pred_3m = pred_1m * 1.8
        pred_6m = pred_1m * 2.5
        pred_12m = pred_1m * 3.2

        # Confidence based on factor agreement and backtest accuracy
        confidence = min(90, max(40, accuracy * 0.7 + composite * 0.3))

        # Signal determination
        if pred_1m > 3 and composite > 60:
            signal = 'STRONG BUY'
        elif pred_1m > 1 and composite > 55:
            signal = 'BUY'
        elif pred_1m < -3 and composite < 40:
            signal = 'STRONG SELL'
        elif pred_1m < -1 and composite < 45:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # Get regime
        classifier = MacroRegimeClassifier(self.quandl)
        regime = classifier.get_all_regimes()

        return PredictionResult(
            symbol=symbol,
            prediction_1m=round(pred_1m, 2),
            prediction_3m=round(pred_3m, 2),
            prediction_6m=round(pred_6m, 2),
            prediction_12m=round(pred_12m, 2),
            confidence=round(confidence, 1),
            direction_accuracy=round(accuracy, 1),
            composite_score=round(composite, 1),
            signal=signal,
            factors_count=factor_count,
            technical_factors=technical,
            fundamental_factors=fundamental,
            macro_factors=macro,
            regime=regime,
        )

    def get_batch_predictions(self, symbols: List[str]) -> List[PredictionResult]:
        """Generate predictions for multiple symbols"""
        results = []
        for symbol in symbols:
            try:
                result = self.generate_prediction(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")
        return results


# ==========================================================================
# FACTOR WEIGHTS OPTIMIZER
# ==========================================================================

class FactorWeightsOptimizer:
    """
    Optimize factor weights using historical data
    """

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def optimize_weights(
        self,
        symbols: List[str],
        validation_months: int = 12
    ) -> Dict[str, float]:
        """
        Find optimal factor weights through backtesting

        Returns:
            Dictionary of optimized factor weights
        """
        # Default weights to start
        weights = {
            'momentum_3m': 0.15,
            'momentum_12m_skip_1m': 0.10,
            'momentum_quality': 0.10,
            'trend_intensity': 0.10,
            'ma_alignment': 0.08,
            'rsi': 0.07,
            'mean_rev_signal': 0.05,
            'pe_ratio': 0.05,
            'roe': 0.05,
            'regime_score': 0.05,
            'breakout': 0.05,
            'vol_ratio': 0.05,
            'composite': 0.10,
        }

        # Would implement grid search or gradient descent here
        # For now, return default weights

        return weights

    def calculate_information_coefficient(
        self,
        symbol: str,
        factor_name: str,
        lookback_months: int = 36
    ) -> float:
        """
        Calculate Information Coefficient for a factor
        (Spearman correlation between factor and forward returns)
        """
        prices = self.pipeline.get_historical_prices_array(symbol)
        if len(prices) < lookback_months + 1:
            return 0.0

        factor_values = []
        forward_returns = []

        for t in range(24, len(prices) - 1):
            historical = prices[:t+1]
            engine = FactorEngine(historical)
            factors = engine.get_all_factors()

            if factor_name in factors:
                factor_values.append(factors[factor_name])
                fwd_ret = (prices[t+1] - prices[t]) / prices[t] * 100
                forward_returns.append(fwd_ret)

        if len(factor_values) < 12:
            return 0.0

        # Spearman rank correlation
        from scipy.stats import spearmanr
        try:
            ic, _ = spearmanr(factor_values, forward_returns)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0

    def get_factor_ic_report(self, symbols: List[str]) -> Dict[str, float]:
        """
        Generate IC report for all factors across multiple symbols
        """
        # Key factors to analyze
        key_factors = [
            'momentum_3m', 'momentum_6m', 'momentum_12m',
            'rsi', 'ma_cross_5_20', 'trend_strength',
            'mean_rev_signal', 'vol_ratio', 'breakout',
        ]

        ic_report = {}
        for factor in key_factors:
            ics = []
            for symbol in symbols:
                ic = self.calculate_information_coefficient(symbol, factor)
                ics.append(ic)
            ic_report[factor] = np.mean(ics)

        return ic_report
