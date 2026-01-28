"""
Alpha Signal Generator
ML-based alpha generation inspired by Medallion Fund's quantitative approach
Combines fundamental, technical, and alternative data for signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SignalType(Enum):
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    GROWTH = "growth"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    ALTERNATIVE = "alternative"
    COMPOSITE = "composite"

class SignalDirection(Enum):
    LONG = 1
    NEUTRAL = 0
    SHORT = -1

@dataclass
class AlphaSignal:
    """Individual alpha signal"""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    signal_name: str
    raw_value: float
    normalized_value: float  # Z-score normalized
    direction: SignalDirection
    confidence: float  # 0-1 confidence score
    decay_rate: float  # Signal decay in days
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompositeAlpha:
    """Combined alpha signal from multiple sources"""
    symbol: str
    timestamp: datetime
    composite_score: float  # Final combined score
    direction: SignalDirection
    confidence: float
    component_signals: List[AlphaSignal]
    expected_return: float  # Expected return estimate
    position_size_suggestion: float  # Kelly-based position suggestion
    risk_score: float  # Risk assessment

class BaseAlphaModel(ABC):
    """Base class for alpha models"""

    def __init__(self, name: str, signal_type: SignalType, decay_days: float = 20):
        self.name = name
        self.signal_type = signal_type
        self.decay_days = decay_days
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[AlphaSignal]:
        """Generate alpha signal for a symbol"""
        pass

    def normalize_signal(
        self,
        value: float,
        historical_values: np.ndarray,
        method: str = "z_score"
    ) -> float:
        """Normalize signal to standard scale"""
        if method == "z_score":
            mean = np.nanmean(historical_values)
            std = np.nanstd(historical_values)
            if std > 0:
                return (value - mean) / std
            return 0.0
        elif method == "percentile":
            return (np.sum(historical_values < value) / len(historical_values)) * 2 - 1
        elif method == "min_max":
            min_val = np.nanmin(historical_values)
            max_val = np.nanmax(historical_values)
            if max_val > min_val:
                return 2 * (value - min_val) / (max_val - min_val) - 1
            return 0.0
        return value


class MomentumAlphaModel(BaseAlphaModel):
    """
    Momentum-based alpha signals
    Includes price momentum, earnings momentum, and revenue momentum
    """

    def __init__(self):
        super().__init__("price_momentum", SignalType.MOMENTUM, decay_days=20)

    def generate_signal(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[AlphaSignal]:
        """Generate momentum signal"""
        try:
            price_data = data.get('price_history')
            if price_data is None or len(price_data) < 252:
                return None

            # Calculate various momentum metrics
            returns = price_data['close'].pct_change()

            # 12-1 month momentum (skip most recent month)
            momentum_12_1 = (
                price_data['close'].iloc[-21] / price_data['close'].iloc[-252] - 1
            )

            # 6-month momentum
            momentum_6m = price_data['close'].iloc[-1] / price_data['close'].iloc[-126] - 1

            # 3-month momentum
            momentum_3m = price_data['close'].iloc[-1] / price_data['close'].iloc[-63] - 1

            # 52-week high ratio
            high_52w = price_data['high'].iloc[-252:].max()
            current_price = price_data['close'].iloc[-1]
            high_ratio = current_price / high_52w

            # Combine momentum signals
            combined_momentum = (
                0.4 * momentum_12_1 +
                0.3 * momentum_6m +
                0.2 * momentum_3m +
                0.1 * (high_ratio - 0.5) * 2
            )

            # Calculate historical momentum for normalization
            hist_momentum = []
            for i in range(252, len(price_data) - 21, 21):
                m = price_data['close'].iloc[-(i-21)] / price_data['close'].iloc[-i] - 1
                hist_momentum.append(m)

            normalized = self.normalize_signal(
                combined_momentum,
                np.array(hist_momentum)
            )

            # Determine direction and confidence
            direction = SignalDirection.LONG if normalized > 0 else SignalDirection.SHORT
            if abs(normalized) < 0.5:
                direction = SignalDirection.NEUTRAL

            confidence = min(abs(normalized) / 3, 1.0)

            return AlphaSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=SignalType.MOMENTUM,
                signal_name="price_momentum_12_1",
                raw_value=combined_momentum,
                normalized_value=normalized,
                direction=direction,
                confidence=confidence,
                decay_rate=self.decay_days,
                metadata={
                    "momentum_12_1": momentum_12_1,
                    "momentum_6m": momentum_6m,
                    "momentum_3m": momentum_3m,
                    "high_52w_ratio": high_ratio
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating momentum signal: {e}")
            return None


class ValueAlphaModel(BaseAlphaModel):
    """
    Value-based alpha signals
    Combines multiple value metrics for signal generation
    """

    def __init__(self):
        super().__init__("composite_value", SignalType.VALUE, decay_days=60)

    def generate_signal(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[AlphaSignal]:
        """Generate value signal"""
        try:
            metrics = data.get('fundamental_metrics')
            if metrics is None:
                return None

            valuation = metrics.valuation
            profitability = metrics.profitability

            # Value factors (inverted - lower is better)
            pe_signal = -self._score_metric(valuation.get('pe_ratio'), 5, 30)
            pb_signal = -self._score_metric(valuation.get('price_to_book'), 0.5, 5)
            ps_signal = -self._score_metric(valuation.get('price_to_sales'), 0.5, 5)
            ev_ebitda_signal = -self._score_metric(valuation.get('ev_to_ebitda'), 5, 20)

            # Earnings yield (higher is better)
            ey_signal = self._score_metric(valuation.get('earnings_yield'), 0.02, 0.15)

            # FCF yield (higher is better)
            fcf_yield = None
            if valuation.get('price_to_free_cash_flow'):
                fcf_yield = 1 / valuation.get('price_to_free_cash_flow')
            fcf_signal = self._score_metric(fcf_yield, 0.02, 0.15) if fcf_yield else 0

            # ROIC (quality adjustment for value)
            roic = profitability.get('return_on_invested_capital', 0.10)
            quality_adj = self._score_metric(roic, 0.05, 0.25)

            # Combined value signal
            combined_value = (
                0.20 * pe_signal +
                0.15 * pb_signal +
                0.10 * ps_signal +
                0.20 * ev_ebitda_signal +
                0.15 * ey_signal +
                0.10 * fcf_signal +
                0.10 * quality_adj
            )

            # Normalize to -1 to 1 range
            normalized = np.clip(combined_value, -1, 1)

            direction = SignalDirection.LONG if normalized > 0.2 else (
                SignalDirection.SHORT if normalized < -0.2 else SignalDirection.NEUTRAL
            )

            confidence = min(abs(normalized), 1.0)

            return AlphaSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=SignalType.VALUE,
                signal_name="composite_value",
                raw_value=combined_value,
                normalized_value=normalized,
                direction=direction,
                confidence=confidence,
                decay_rate=self.decay_days,
                metadata={
                    "pe_signal": pe_signal,
                    "pb_signal": pb_signal,
                    "ev_ebitda_signal": ev_ebitda_signal,
                    "earnings_yield_signal": ey_signal,
                    "quality_adjustment": quality_adj
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating value signal: {e}")
            return None

    def _score_metric(
        self,
        value: Optional[float],
        low_threshold: float,
        high_threshold: float
    ) -> float:
        """Score a metric between -1 and 1"""
        if value is None:
            return 0

        if value <= low_threshold:
            return 1.0
        elif value >= high_threshold:
            return -1.0
        else:
            # Linear interpolation
            return 1 - 2 * (value - low_threshold) / (high_threshold - low_threshold)


class QualityAlphaModel(BaseAlphaModel):
    """
    Quality-based alpha signals
    Based on profitability, earnings quality, and financial health
    """

    def __init__(self):
        super().__init__("quality_factor", SignalType.QUALITY, decay_days=40)

    def generate_signal(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[AlphaSignal]:
        """Generate quality signal"""
        try:
            metrics = data.get('fundamental_metrics')
            if metrics is None:
                return None

            quality_scores = metrics.quality_scores
            profitability = metrics.profitability
            cash_flow = metrics.cash_flow
            financial_health = metrics.financial_health

            # Profitability quality
            prof_quality = quality_scores.get('profitability_quality', 50) / 100

            # Cash flow quality
            cf_quality = quality_scores.get('cash_flow_quality', 50) / 100

            # Financial strength
            fin_strength = quality_scores.get('financial_strength', 50) / 100

            # Specific quality metrics
            roe = profitability.get('return_on_equity', 0.10)
            roic = profitability.get('return_on_invested_capital', 0.10)
            gross_profit_to_assets = profitability.get('gross_profit_to_assets', 0.20)

            # Earnings quality
            accruals = cash_flow.get('accruals_ratio', 0)
            quality_of_earnings = cash_flow.get('quality_of_earnings', 1)

            # Piotroski F-Score
            piotroski = financial_health.get('piotroski_f_score', 5)

            # Combine quality signals
            combined_quality = (
                0.20 * prof_quality +
                0.15 * cf_quality +
                0.15 * fin_strength +
                0.10 * min(roe / 0.20, 1) +
                0.10 * min(roic / 0.15, 1) +
                0.10 * min(gross_profit_to_assets / 0.30, 1) +
                0.10 * (1 - min(abs(accruals) / 0.10, 1)) +
                0.10 * (piotroski / 9)
            )

            # Convert to -1 to 1 scale (centered around 0.5)
            normalized = (combined_quality - 0.5) * 2

            direction = SignalDirection.LONG if normalized > 0.2 else (
                SignalDirection.SHORT if normalized < -0.2 else SignalDirection.NEUTRAL
            )

            confidence = min(abs(normalized), 1.0)

            return AlphaSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=SignalType.QUALITY,
                signal_name="quality_factor",
                raw_value=combined_quality,
                normalized_value=normalized,
                direction=direction,
                confidence=confidence,
                decay_rate=self.decay_days,
                metadata={
                    "profitability_quality": prof_quality,
                    "cash_flow_quality": cf_quality,
                    "financial_strength": fin_strength,
                    "piotroski_score": piotroski,
                    "quality_of_earnings": quality_of_earnings
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating quality signal: {e}")
            return None


class GrowthAlphaModel(BaseAlphaModel):
    """
    Growth-based alpha signals
    Focuses on revenue growth, earnings growth, and growth sustainability
    """

    def __init__(self):
        super().__init__("growth_factor", SignalType.GROWTH, decay_days=30)

    def generate_signal(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[AlphaSignal]:
        """Generate growth signal"""
        try:
            metrics = data.get('fundamental_metrics')
            if metrics is None:
                return None

            growth = metrics.growth
            quality_scores = metrics.quality_scores

            # Growth metrics
            rev_growth_yoy = growth.get('revenue_growth_yoy', 0) or 0
            earn_growth_yoy = growth.get('earnings_growth_yoy', 0) or 0
            rev_growth_3y = growth.get('revenue_growth_3y_cagr', 0) or 0
            earn_growth_3y = growth.get('earnings_growth_3y_cagr', 0) or 0
            fcf_growth = growth.get('fcf_growth_yoy', 0) or 0
            sustainable_growth = growth.get('sustainable_growth_rate', 0) or 0

            # Growth quality (from quality scores)
            growth_quality = quality_scores.get('growth_quality', 50) / 100

            # Score individual growth metrics
            rev_yoy_score = np.tanh(rev_growth_yoy * 3)  # Scales ~33% growth to ~1
            earn_yoy_score = np.tanh(earn_growth_yoy * 3)
            rev_3y_score = np.tanh(rev_growth_3y * 5)  # ~20% CAGR to ~1
            earn_3y_score = np.tanh(earn_growth_3y * 5)
            fcf_score = np.tanh(fcf_growth * 3)

            # Growth consistency (earnings growth > revenue growth suggests margin expansion)
            consistency_bonus = 0.1 if earn_growth_yoy > rev_growth_yoy else 0

            # Combined growth signal
            combined_growth = (
                0.20 * rev_yoy_score +
                0.25 * earn_yoy_score +
                0.15 * rev_3y_score +
                0.15 * earn_3y_score +
                0.10 * fcf_score +
                0.05 * (sustainable_growth * 5) +
                0.10 * (growth_quality - 0.5) * 2 +
                consistency_bonus
            )

            normalized = np.clip(combined_growth, -1, 1)

            direction = SignalDirection.LONG if normalized > 0.2 else (
                SignalDirection.SHORT if normalized < -0.2 else SignalDirection.NEUTRAL
            )

            confidence = min(abs(normalized), 1.0)

            return AlphaSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=SignalType.GROWTH,
                signal_name="growth_factor",
                raw_value=combined_growth,
                normalized_value=normalized,
                direction=direction,
                confidence=confidence,
                decay_rate=self.decay_days,
                metadata={
                    "revenue_growth_yoy": rev_growth_yoy,
                    "earnings_growth_yoy": earn_growth_yoy,
                    "revenue_growth_3y_cagr": rev_growth_3y,
                    "earnings_growth_3y_cagr": earn_growth_3y,
                    "growth_quality_score": growth_quality
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating growth signal: {e}")
            return None


class TechnicalAlphaModel(BaseAlphaModel):
    """
    Technical analysis alpha signals
    Combines trend, momentum, and mean-reversion indicators
    """

    def __init__(self):
        super().__init__("technical_composite", SignalType.TECHNICAL, decay_days=5)

    def generate_signal(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[AlphaSignal]:
        """Generate technical signal"""
        try:
            price_data = data.get('price_history')
            if price_data is None or len(price_data) < 200:
                return None

            close = price_data['close']
            high = price_data['high']
            low = price_data['low']
            volume = price_data['volume']

            signals = {}

            # Trend signals
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            current_price = close.iloc[-1]

            # Price vs SMAs
            signals['price_vs_sma20'] = (current_price / sma_20 - 1) * 5
            signals['price_vs_sma50'] = (current_price / sma_50 - 1) * 3
            signals['price_vs_sma200'] = (current_price / sma_200 - 1) * 2

            # Golden/Death cross
            signals['sma_cross'] = 1 if sma_50 > sma_200 else -1

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]

            # RSI signal (mean reversion)
            if rsi_value > 70:
                signals['rsi'] = -(rsi_value - 70) / 30  # Overbought
            elif rsi_value < 30:
                signals['rsi'] = (30 - rsi_value) / 30  # Oversold
            else:
                signals['rsi'] = 0

            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal

            signals['macd'] = np.tanh(macd_hist.iloc[-1] / close.iloc[-1] * 100)

            # Bollinger Bands
            bb_sma = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_sma + 2 * bb_std
            bb_lower = bb_sma - 2 * bb_std

            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            signals['bollinger'] = -(bb_position - 0.5) * 2  # Mean reversion

            # Volume analysis
            vol_sma = volume.rolling(20).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            vol_ratio = current_vol / vol_sma if vol_sma > 0 else 1

            # Combine with price direction for volume confirmation
            price_change = close.iloc[-1] / close.iloc[-2] - 1
            if price_change > 0 and vol_ratio > 1.5:
                signals['volume_confirm'] = 0.5
            elif price_change < 0 and vol_ratio > 1.5:
                signals['volume_confirm'] = -0.5
            else:
                signals['volume_confirm'] = 0

            # ATR for volatility adjustment
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr / current_price

            # Weight signals
            combined_technical = (
                0.15 * np.clip(signals['price_vs_sma20'], -1, 1) +
                0.15 * np.clip(signals['price_vs_sma50'], -1, 1) +
                0.10 * np.clip(signals['price_vs_sma200'], -1, 1) +
                0.10 * signals['sma_cross'] +
                0.15 * signals['rsi'] +
                0.15 * signals['macd'] +
                0.10 * signals['bollinger'] +
                0.10 * signals['volume_confirm']
            )

            # Adjust for volatility (reduce signal in high volatility)
            vol_adjustment = 1 - min(atr_pct / 0.05, 0.5)
            combined_technical *= vol_adjustment

            normalized = np.clip(combined_technical, -1, 1)

            direction = SignalDirection.LONG if normalized > 0.2 else (
                SignalDirection.SHORT if normalized < -0.2 else SignalDirection.NEUTRAL
            )

            confidence = min(abs(normalized) * (1 + vol_ratio * 0.1), 1.0)

            return AlphaSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=SignalType.TECHNICAL,
                signal_name="technical_composite",
                raw_value=combined_technical,
                normalized_value=normalized,
                direction=direction,
                confidence=confidence,
                decay_rate=self.decay_days,
                metadata={
                    "rsi": rsi_value,
                    "macd_histogram": macd_hist.iloc[-1],
                    "sma_cross": signals['sma_cross'],
                    "bb_position": bb_position,
                    "volume_ratio": vol_ratio,
                    "atr_percent": atr_pct
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating technical signal: {e}")
            return None


class AlphaAggregator:
    """
    Combines multiple alpha signals into a composite score
    Implements Medallion-inspired signal combination with dynamic weighting
    """

    def __init__(self):
        self.models = {
            SignalType.MOMENTUM: MomentumAlphaModel(),
            SignalType.VALUE: ValueAlphaModel(),
            SignalType.QUALITY: QualityAlphaModel(),
            SignalType.GROWTH: GrowthAlphaModel(),
            SignalType.TECHNICAL: TechnicalAlphaModel(),
        }

        # Default weights (can be dynamically adjusted based on regime)
        self.weights = {
            SignalType.MOMENTUM: 0.20,
            SignalType.VALUE: 0.20,
            SignalType.QUALITY: 0.25,
            SignalType.GROWTH: 0.15,
            SignalType.TECHNICAL: 0.10,
            SignalType.SENTIMENT: 0.05,
            SignalType.ALTERNATIVE: 0.05,
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_composite_alpha(
        self,
        symbol: str,
        data: Dict[str, Any],
        risk_free_rate: float = 0.05
    ) -> Optional[CompositeAlpha]:
        """
        Generate composite alpha signal from all models

        Args:
            symbol: Security symbol
            data: Dictionary containing price_history, fundamental_metrics, etc.
            risk_free_rate: Current risk-free rate for position sizing

        Returns:
            CompositeAlpha with combined signal and position suggestion
        """
        try:
            signals = []

            # Generate signals from each model
            for signal_type, model in self.models.items():
                signal = model.generate_signal(symbol, data)
                if signal:
                    signals.append(signal)
                    self.logger.debug(
                        f"{symbol} {signal_type.value}: {signal.normalized_value:.3f}"
                    )

            if not signals:
                self.logger.warning(f"No signals generated for {symbol}")
                return None

            # Calculate weighted composite score
            weighted_sum = 0
            total_weight = 0
            confidence_weighted_sum = 0

            for signal in signals:
                weight = self.weights.get(signal.signal_type, 0.1)

                # Adjust weight by confidence
                effective_weight = weight * signal.confidence

                weighted_sum += signal.normalized_value * effective_weight
                confidence_weighted_sum += signal.confidence * weight
                total_weight += effective_weight

            composite_score = weighted_sum / total_weight if total_weight > 0 else 0
            avg_confidence = confidence_weighted_sum / sum(
                self.weights.get(s.signal_type, 0.1) for s in signals
            )

            # Determine direction
            if composite_score > 0.15:
                direction = SignalDirection.LONG
            elif composite_score < -0.15:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            # Estimate expected return (simplified)
            # In production, this would use historical signal performance
            expected_return = composite_score * 0.10  # Scale to realistic return

            # Calculate risk score (inverse of quality signals)
            risk_signals = [s for s in signals if s.signal_type in [
                SignalType.QUALITY, SignalType.VALUE
            ]]
            if risk_signals:
                risk_score = 1 - np.mean([
                    (s.normalized_value + 1) / 2 for s in risk_signals
                ])
            else:
                risk_score = 0.5

            # Kelly criterion position sizing
            # f* = (p*b - q) / b where p=win prob, q=lose prob, b=odds
            # Simplified: f* = expected_return / variance
            if direction != SignalDirection.NEUTRAL and expected_return != 0:
                # Estimate win probability from confidence
                win_prob = 0.5 + avg_confidence * 0.15  # 50% to 65%

                # Simplified Kelly
                kelly_fraction = (win_prob * 2 - 1) / 1  # Assuming 1:1 odds
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

                # Adjust for risk
                position_size = kelly_fraction * (1 - risk_score * 0.5)
            else:
                position_size = 0

            return CompositeAlpha(
                symbol=symbol,
                timestamp=datetime.now(),
                composite_score=composite_score,
                direction=direction,
                confidence=avg_confidence,
                component_signals=signals,
                expected_return=expected_return,
                position_size_suggestion=position_size,
                risk_score=risk_score
            )

        except Exception as e:
            self.logger.error(f"Error generating composite alpha for {symbol}: {e}")
            return None

    def rank_securities(
        self,
        alphas: List[CompositeAlpha]
    ) -> pd.DataFrame:
        """
        Rank securities by composite alpha score

        Returns DataFrame with rankings for portfolio construction
        """
        if not alphas:
            return pd.DataFrame()

        data = []
        for alpha in alphas:
            data.append({
                'symbol': alpha.symbol,
                'composite_score': alpha.composite_score,
                'direction': alpha.direction.value,
                'confidence': alpha.confidence,
                'expected_return': alpha.expected_return,
                'position_size': alpha.position_size_suggestion,
                'risk_score': alpha.risk_score,
                'timestamp': alpha.timestamp
            })

        df = pd.DataFrame(data)

        # Rank by composite score
        df['alpha_rank'] = df['composite_score'].rank(ascending=False)

        # Separate long and short rankings
        df['long_rank'] = df[df['direction'] == 1]['composite_score'].rank(ascending=False)
        df['short_rank'] = df[df['direction'] == -1]['composite_score'].rank(ascending=True)

        return df.sort_values('composite_score', ascending=False)

    def adjust_weights_for_regime(
        self,
        market_regime: str
    ):
        """
        Dynamically adjust factor weights based on market regime

        Args:
            market_regime: One of 'bull', 'bear', 'high_volatility', 'low_volatility'
        """
        regime_weights = {
            'bull': {
                SignalType.MOMENTUM: 0.30,
                SignalType.VALUE: 0.10,
                SignalType.QUALITY: 0.20,
                SignalType.GROWTH: 0.25,
                SignalType.TECHNICAL: 0.10,
            },
            'bear': {
                SignalType.MOMENTUM: 0.10,
                SignalType.VALUE: 0.30,
                SignalType.QUALITY: 0.35,
                SignalType.GROWTH: 0.10,
                SignalType.TECHNICAL: 0.10,
            },
            'high_volatility': {
                SignalType.MOMENTUM: 0.10,
                SignalType.VALUE: 0.25,
                SignalType.QUALITY: 0.35,
                SignalType.GROWTH: 0.10,
                SignalType.TECHNICAL: 0.15,
            },
            'low_volatility': {
                SignalType.MOMENTUM: 0.25,
                SignalType.VALUE: 0.20,
                SignalType.QUALITY: 0.20,
                SignalType.GROWTH: 0.20,
                SignalType.TECHNICAL: 0.10,
            }
        }

        if market_regime in regime_weights:
            self.weights.update(regime_weights[market_regime])
            self.logger.info(f"Adjusted weights for {market_regime} regime")
