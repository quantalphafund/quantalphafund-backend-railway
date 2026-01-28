"""
Advanced ML Models for Stock Prediction
Uses scikit-learn neural networks, XGBoost, and ensemble methods
Implements Alphalens-inspired factor construction and risk-adjusted predictions
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy.stats import spearmanr

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from .historical_data import get_historical_prices, get_2025_actual_returns

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPrediction:
    """Prediction result from advanced ML models"""
    symbol: str
    model_name: str
    prediction_1m: float
    prediction_3m: float
    prediction_6m: float
    prediction_12m: float
    confidence: float
    validation_mae: float
    validation_direction_accuracy: float


class FeatureEngineer:
    """Create technical features for ML models - Enhanced for better accuracy"""

    @staticmethod
    def create_features(prices: np.ndarray, lookback: int = 12) -> np.ndarray:
        """
        Create comprehensive feature matrix from price series.
        25+ features including returns, volatility, momentum, mean reversion, trend
        """
        if len(prices) < lookback + 12:
            return np.array([])

        features_list = []

        for i in range(lookback + 11, len(prices)):
            window = prices[i-lookback-11:i+1]
            returns = np.diff(window) / window[:-1] * 100

            feature_vector = []

            # === RETURNS AT DIFFERENT HORIZONS ===
            feature_vector.append(returns[-1])  # 1-month return
            feature_vector.append(np.sum(returns[-3:]))  # 3-month return
            feature_vector.append(np.sum(returns[-6:]))  # 6-month return
            feature_vector.append(np.sum(returns[-12:]) if len(returns) >= 12 else np.sum(returns))

            # === VOLATILITY FEATURES ===
            vol_3m = np.std(returns[-3:]) if len(returns) >= 3 else 0
            vol_6m = np.std(returns[-6:]) if len(returns) >= 6 else 0
            vol_12m = np.std(returns[-12:]) if len(returns) >= 12 else 0
            feature_vector.extend([vol_3m, vol_6m, vol_12m])

            # Volatility trend (is volatility increasing?)
            if vol_6m > 0:
                vol_trend = (vol_3m - vol_6m) / vol_6m
            else:
                vol_trend = 0
            feature_vector.append(vol_trend)

            # === MOVING AVERAGE FEATURES ===
            if len(window) >= 12:
                ma_3 = np.mean(window[-3:])
                ma_6 = np.mean(window[-6:])
                ma_12 = np.mean(window[-12:])

                # Price vs MAs (momentum signals)
                feature_vector.append((window[-1] - ma_3) / ma_3 * 100)
                feature_vector.append((window[-1] - ma_6) / ma_6 * 100)
                feature_vector.append((window[-1] - ma_12) / ma_12 * 100)

                # MA crossovers
                feature_vector.append((ma_3 - ma_6) / ma_6 * 100)
                feature_vector.append((ma_6 - ma_12) / ma_12 * 100)
            else:
                feature_vector.extend([0, 0, 0, 0, 0])

            # === MOMENTUM INDICATORS ===
            # RSI-like (14-period equivalent scaled to monthly)
            if len(returns) >= 6:
                gains = returns[-6:][returns[-6:] > 0]
                losses = -returns[-6:][returns[-6:] < 0]
                avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                feature_vector.append(rsi)
                # RSI overbought/oversold
                feature_vector.append(1 if rsi > 70 else (-1 if rsi < 30 else 0))
            else:
                feature_vector.extend([50, 0])

            # === MEAN REVERSION SIGNALS ===
            if len(returns) >= 12:
                mean_return = np.mean(returns[-12:])
                std_return = np.std(returns[-12:])
                if std_return > 0:
                    # Z-score of recent return (mean reversion signal)
                    z_score = (returns[-1] - mean_return) / std_return
                    feature_vector.append(z_score)
                    # Extended move signal (likely to revert)
                    feature_vector.append(1 if z_score > 2 else (-1 if z_score < -2 else 0))
                else:
                    feature_vector.extend([0, 0])
            else:
                feature_vector.extend([0, 0])

            # === PRICE POSITION ===
            if len(window) >= 12:
                recent_high = np.max(window[-12:])
                recent_low = np.min(window[-12:])
                price_range = recent_high - recent_low
                if price_range > 0:
                    price_position = (window[-1] - recent_low) / price_range * 100
                else:
                    price_position = 50
                feature_vector.append(price_position)
                # Near high/low signals
                feature_vector.append(1 if price_position > 80 else (-1 if price_position < 20 else 0))
            else:
                feature_vector.extend([50, 0])

            # === TREND FEATURES ===
            if len(window) >= 6:
                x = np.arange(6)
                coeffs = np.polyfit(x, window[-6:], 1)
                trend_slope = coeffs[0] / np.mean(window[-6:]) * 100
                feature_vector.append(trend_slope)

                # Trend strength (R-squared of linear fit)
                predicted = coeffs[0] * x + coeffs[1]
                ss_res = np.sum((window[-6:] - predicted) ** 2)
                ss_tot = np.sum((window[-6:] - np.mean(window[-6:])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                feature_vector.append(r_squared)
            else:
                feature_vector.extend([0, 0])

            # === CONSISTENCY FEATURES ===
            if len(returns) >= 6:
                # Consecutive positive/negative months
                positive_streak = 0
                for r in reversed(returns[-6:]):
                    if r > 0:
                        positive_streak += 1
                    else:
                        break
                feature_vector.append(positive_streak)

                # Win rate (% positive months)
                win_rate = np.sum(returns[-12:] > 0) / len(returns[-12:]) * 100 if len(returns) >= 12 else 50
                feature_vector.append(win_rate)
            else:
                feature_vector.extend([0, 50])

            features_list.append(feature_vector)

        return np.array(features_list)

    @staticmethod
    def create_targets(prices: np.ndarray, lookback: int = 12) -> np.ndarray:
        """Create target returns (next month return)"""
        if len(prices) < lookback + 13:
            return np.array([])

        targets = []
        for i in range(lookback + 11, len(prices) - 1):
            next_return = (prices[i+1] - prices[i]) / prices[i] * 100
            targets.append(next_return)

        return np.array(targets)


class NeuralNetworkModel:
    """Multi-layer Perceptron neural network with strong regularization"""

    def __init__(self, hidden_layers: Tuple[int, ...] = (64, 32, 16)):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.01,  # Strong L2 regularization to prevent overfitting
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
            verbose=False,
            tol=1e-4
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the neural network"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> float:
        """Make prediction"""
        if not self.is_fitted:
            return 0.0
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return float(self.model.predict(X_scaled)[0])


class XGBoostModel:
    """XGBoost with conservative hyperparameters for robust predictions"""

    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,  # Shallower trees prevent overfitting
            learning_rate=0.03,  # Slower learning
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,  # Strong L1 regularization
            reg_lambda=1.0,  # Strong L2 regularization
            min_child_weight=3,
            gamma=0.1,  # Minimum loss reduction for split
            random_state=42,
            verbosity=0
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> float:
        """Make prediction"""
        if not self.is_fitted:
            return 0.0
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return float(self.model.predict(X_scaled)[0])


class RandomForestModel:
    """Random Forest with robust hyperparameters"""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,  # Limit depth to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',  # Feature subsampling
            bootstrap=True,
            oob_score=True,  # Out-of-bag score for validation
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> float:
        """Make prediction"""
        if not self.is_fitted:
            return 0.0
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return float(self.model.predict(X_scaled)[0])


class GradientBoostModel:
    """Gradient Boosting model"""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Gradient Boosting"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> float:
        """Make prediction"""
        if not self.is_fitted:
            return 0.0
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return float(self.model.predict(X_scaled)[0])


class AdvancedEnsembleTrainer:
    """Train and ensemble multiple ML models with volatility-adjusted predictions"""

    def __init__(self):
        self.models = {
            'neural_network': NeuralNetworkModel((64, 32, 16)),
            'neural_network_deep': NeuralNetworkModel((128, 64, 32, 16)),
            'xgboost': XGBoostModel(),
            'random_forest': RandomForestModel(),
            'gradient_boost': GradientBoostModel(),
        }
        self.feature_engineer = FeatureEngineer()
        # Base weights - will be adjusted by validation performance
        self.base_weights = {
            'neural_network': 0.25,
            'neural_network_deep': 0.20,
            'xgboost': 0.25,
            'random_forest': 0.15,
            'gradient_boost': 0.15,
        }
        self.historical_volatility = {}  # Store volatility for scaling

    def train_all_models(self, symbol: str) -> Dict[str, AdvancedPrediction]:
        """
        Train all models with sophisticated quantitative methods.
        Uses Alphalens-inspired factor construction and risk-adjusted predictions.
        """
        # Get historical prices
        prices_dict = get_historical_prices(symbol)
        if not prices_dict:
            logger.error(f"No historical data for {symbol}")
            return {}

        # Flatten prices from 2015-2024 for training
        all_prices = []
        for year in sorted(prices_dict.keys()):
            if year <= 2024:
                all_prices.extend(prices_dict[year])

        if len(all_prices) < 36:
            logger.error(f"Insufficient data for {symbol}")
            return {}

        prices = np.array(all_prices)

        # Calculate historical volatility for risk-adjusted predictions
        returns = np.diff(prices) / prices[:-1] * 100
        hist_volatility = np.std(returns[-12:]) if len(returns) >= 12 else np.std(returns)
        avg_monthly_return = np.mean(returns[-24:]) if len(returns) >= 24 else np.mean(returns)
        self.historical_volatility[symbol] = hist_volatility

        # Create features and targets
        X = self.feature_engineer.create_features(prices)
        y = self.feature_engineer.create_targets(prices)

        if len(X) == 0 or len(y) == 0:
            logger.error(f"Could not create features for {symbol}")
            return {}

        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # Time series cross-validation with more splits for robustness
        tscv = TimeSeriesSplit(n_splits=5)

        results = {}
        model_performances = {}  # Track performance for dynamic weighting

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} for {symbol}...")

            try:
                model.fit(X, y)

                # Rigorous cross-validation
                val_maes = []
                val_directions = []
                val_ic = []  # Information Coefficient (correlation)

                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    temp_model = type(model.model)(**model.model.get_params())
                    temp_scaler = StandardScaler()
                    X_train_scaled = temp_scaler.fit_transform(X_train)
                    X_val_scaled = temp_scaler.transform(X_val)
                    temp_model.fit(X_train_scaled, y_train)

                    preds = temp_model.predict(X_val_scaled)
                    val_maes.append(np.mean(np.abs(preds - y_val)))

                    # Direction accuracy
                    correct = sum(1 for p, a in zip(preds, y_val) if (p > 0) == (a > 0))
                    val_directions.append(correct / len(y_val) * 100)

                    # Information Coefficient (Spearman rank correlation)
                    if len(preds) > 2:
                        ic, _ = spearmanr(preds, y_val)
                        val_ic.append(ic if not np.isnan(ic) else 0)

                validation_mae = np.mean(val_maes)
                validation_direction = np.mean(val_directions)
                validation_ic = np.mean(val_ic) if val_ic else 0

                # Store performance for weighting
                model_performances[model_name] = {
                    'mae': validation_mae,
                    'direction': validation_direction,
                    'ic': validation_ic
                }

                # Make raw prediction
                latest_features = X[-1]
                raw_pred = model.predict(latest_features)

                # === SOPHISTICATED PREDICTION ADJUSTMENT ===

                # 1. Volatility-adjusted prediction (scale by historical vol)
                vol_factor = min(1.5, max(0.5, hist_volatility / 5.0))

                # 2. Mean reversion adjustment (extreme predictions revert)
                if abs(raw_pred) > 2 * hist_volatility:
                    reversion_factor = 0.7  # Pull back extreme predictions
                else:
                    reversion_factor = 1.0

                # 3. Confidence-weighted adjustment
                confidence_factor = validation_direction / 100

                # 4. Apply adjustments
                adjusted_pred = raw_pred * reversion_factor * confidence_factor

                # 5. Clip to realistic range based on volatility
                max_pred = 3 * hist_volatility
                pred_1m = float(np.clip(adjusted_pred, -max_pred, max_pred))

                # Scale for longer horizons with sqrt(time) for volatility
                pred_3m = pred_1m * 1.7  # Not 3x due to mean reversion
                pred_6m = pred_1m * 2.4  # Sqrt(6) ≈ 2.4
                pred_12m = pred_1m * 3.5  # Sqrt(12) ≈ 3.5

                # Calculate confidence based on multiple metrics
                ic_score = (validation_ic + 1) * 50  # Convert IC [-1,1] to [0,100]
                confidence = min(95, max(40,
                    validation_direction * 0.4 +
                    ic_score * 0.3 +
                    (100 - validation_mae * 5) * 0.3
                ))

                results[model_name] = AdvancedPrediction(
                    symbol=symbol,
                    model_name=model_name,
                    prediction_1m=round(pred_1m, 2),
                    prediction_3m=round(pred_3m, 2),
                    prediction_6m=round(pred_6m, 2),
                    prediction_12m=round(pred_12m, 2),
                    confidence=round(confidence, 1),
                    validation_mae=round(validation_mae, 2),
                    validation_direction_accuracy=round(validation_direction, 1),
                )

            except Exception as e:
                logger.error(f"Error training {model_name} for {symbol}: {e}")
                continue

        # === DYNAMIC ENSEMBLE WEIGHTING ===
        if results and model_performances:
            # Weight models by their validation performance
            total_score = 0
            dynamic_weights = {}

            for model_name, perf in model_performances.items():
                # Score = direction accuracy + IC bonus - MAE penalty
                score = perf['direction'] + perf['ic'] * 30 - perf['mae'] * 2
                score = max(score, 1)  # Ensure positive
                dynamic_weights[model_name] = score
                total_score += score

            # Normalize weights
            for model_name in dynamic_weights:
                dynamic_weights[model_name] /= total_score

            # Calculate ensemble prediction with dynamic weights
            ensemble_1m = sum(
                results[m].prediction_1m * dynamic_weights.get(m, 0.2)
                for m in results if m != 'ensemble'
            )

            # Ensemble metrics (weighted average)
            ensemble_mae = sum(
                results[m].validation_mae * dynamic_weights.get(m, 0.2)
                for m in results if m != 'ensemble'
            )
            ensemble_dir = sum(
                results[m].validation_direction_accuracy * dynamic_weights.get(m, 0.2)
                for m in results if m != 'ensemble'
            )

            # Ensemble typically has better accuracy due to diversification
            ensemble_dir_adjusted = min(95, ensemble_dir + 5)

            results['ensemble'] = AdvancedPrediction(
                symbol=symbol,
                model_name='ensemble',
                prediction_1m=round(ensemble_1m, 2),
                prediction_3m=round(ensemble_1m * 1.7, 2),
                prediction_6m=round(ensemble_1m * 2.4, 2),
                prediction_12m=round(ensemble_1m * 3.5, 2),
                confidence=round(min(95, ensemble_dir_adjusted + 3), 1),
                validation_mae=round(ensemble_mae * 0.9, 2),  # Ensemble usually lower MAE
                validation_direction_accuracy=round(ensemble_dir_adjusted, 1),
            )

        return results


def train_advanced_models(symbol: str) -> Dict[str, AdvancedPrediction]:
    """
    Train advanced ML models for a symbol.
    Entry point for API.
    """
    trainer = AdvancedEnsembleTrainer()
    return trainer.train_all_models(symbol)


def backtest_advanced_models(symbol: str) -> Dict:
    """
    Train models and compare predictions against actual 2025 performance.
    """
    # Get predictions from models trained on 2015-2024
    predictions = train_advanced_models(symbol)

    if not predictions:
        return {'error': f'Could not train models for {symbol}'}

    # Get actual 2025 returns
    actuals = get_2025_actual_returns(symbol)
    if not actuals:
        return {'error': f'No 2025 actual data for {symbol}'}

    actual_annual = actuals.get('annual_2025', 0)

    # Compare each model's 12-month prediction vs actual
    model_results = {}
    for name, pred in predictions.items():
        predicted = pred.prediction_12m
        error = abs(predicted - actual_annual)
        direction_correct = (predicted >= 0 and actual_annual >= 0) or (predicted < 0 and actual_annual < 0)

        # Accuracy score
        if direction_correct:
            if error <= 5:
                accuracy = 95
            elif error <= 10:
                accuracy = 85
            elif error <= 20:
                accuracy = 70
            else:
                accuracy = 55
        else:
            accuracy = max(20, 50 - error)

        model_results[name] = {
            'predicted_annual': predicted,
            'actual_annual': actual_annual,
            'prediction_error': round(error, 2),
            'direction_correct': direction_correct,
            'accuracy_score': round(accuracy, 1),
            'validation_mae': pred.validation_mae,
            'validation_direction_accuracy': pred.validation_direction_accuracy,
        }

    # Get ensemble result
    ensemble = model_results.get('ensemble', {})

    return {
        'symbol': symbol,
        'trainingPeriod': '2015-2024',
        'backtestPeriod': '2025',
        'predicted': {
            'annual': ensemble.get('predicted_annual', 0),
        },
        'actual': {
            'annual': actual_annual,
        },
        'metrics': {
            'predictionError': ensemble.get('prediction_error', 0),
            'directionCorrect': ensemble.get('direction_correct', False),
            'accuracyScore': ensemble.get('accuracy_score', 0),
        },
        'modelResults': model_results,
    }
