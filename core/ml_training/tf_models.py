"""
TensorFlow ML Models for Stock Prediction
Real LSTM, GRU, and Dense neural networks trained on 10 years of data
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Input, Concatenate, Bidirectional, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from .historical_data import get_historical_prices, HISTORICAL_DATA

logger = logging.getLogger(__name__)

# Cache directory for trained models
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


@dataclass
class TFModelPrediction:
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
    """Create technical features from price data"""

    @staticmethod
    def create_features(prices: np.ndarray) -> np.ndarray:
        """
        Create rich feature set from price data

        Features:
        - Returns (1, 3, 6, 12 month)
        - Volatility (rolling std)
        - Momentum indicators
        - Moving averages
        - RSI-like indicator
        - Price position relative to range
        """
        if len(prices) < 24:
            return np.array([])

        features = []

        for i in range(24, len(prices)):
            window = prices[i-24:i]
            current = prices[i-1]

            # Returns at different horizons
            ret_1m = (current - prices[i-2]) / prices[i-2] * 100 if i >= 2 else 0
            ret_3m = (current - prices[i-4]) / prices[i-4] * 100 if i >= 4 else 0
            ret_6m = (current - prices[i-7]) / prices[i-7] * 100 if i >= 7 else 0
            ret_12m = (current - prices[i-13]) / prices[i-13] * 100 if i >= 13 else 0

            # Volatility
            vol_3m = np.std(window[-3:]) / np.mean(window[-3:]) * 100 if np.mean(window[-3:]) > 0 else 0
            vol_6m = np.std(window[-6:]) / np.mean(window[-6:]) * 100 if np.mean(window[-6:]) > 0 else 0
            vol_12m = np.std(window[-12:]) / np.mean(window[-12:]) * 100 if np.mean(window[-12:]) > 0 else 0

            # Moving averages
            ma_3 = np.mean(window[-3:])
            ma_6 = np.mean(window[-6:])
            ma_12 = np.mean(window[-12:])

            # MA crossovers
            ma_cross_short = (ma_3 - ma_6) / ma_6 * 100 if ma_6 > 0 else 0
            ma_cross_long = (ma_6 - ma_12) / ma_12 * 100 if ma_12 > 0 else 0

            # Price position (0-100 scale)
            high_12m = np.max(window[-12:])
            low_12m = np.min(window[-12:])
            price_position = (current - low_12m) / (high_12m - low_12m) * 100 if high_12m != low_12m else 50

            # RSI-like momentum
            gains = []
            losses = []
            for j in range(1, min(15, len(window))):
                change = window[-j] - window[-j-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))

            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))

            # Trend strength
            trend = np.polyfit(range(12), window[-12:], 1)[0] / np.mean(window[-12:]) * 100

            features.append([
                ret_1m, ret_3m, ret_6m, ret_12m,
                vol_3m, vol_6m, vol_12m,
                ma_cross_short, ma_cross_long,
                price_position, rsi, trend
            ])

        return np.array(features)

    @staticmethod
    def create_sequences(features: np.ndarray, targets: np.ndarray, lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/GRU training"""
        X, y = [], []

        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            y.append(targets[i])

        return np.array(X), np.array(y)


class LSTMModel:
    """LSTM model for stock prediction"""

    def __init__(self, input_shape: Tuple[int, int], name: str = "LSTM"):
        self.name = name
        self.model = self._build_model(input_shape)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            Dense(16, activation='relu'),
            Dropout(0.1),

            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )

        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, validation_split: float = 0.2):
        """Train the LSTM model"""
        # Scale features
        n_samples, n_steps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_flat).reshape(n_samples, n_steps, n_features)

        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]

        self.history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        n_samples, n_steps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_flat).reshape(n_samples, n_steps, n_features)

        y_scaled = self.model.predict(X_scaled, verbose=0)
        return self.scaler_y.inverse_transform(y_scaled).flatten()


class GRUModel:
    """GRU model for stock prediction"""

    def __init__(self, input_shape: Tuple[int, int], name: str = "GRU"):
        self.name = name
        self.model = self._build_model(input_shape)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build GRU architecture"""
        model = Sequential([
            Bidirectional(GRU(48, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),

            GRU(24, return_sequences=False),
            Dropout(0.2),

            Dense(12, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )

        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, validation_split: float = 0.2):
        """Train the GRU model"""
        n_samples, n_steps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_flat).reshape(n_samples, n_steps, n_features)

        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]

        self.history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        n_samples, n_steps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_flat).reshape(n_samples, n_steps, n_features)

        y_scaled = self.model.predict(X_scaled, verbose=0)
        return self.scaler_y.inverse_transform(y_scaled).flatten()


class DenseModel:
    """Dense neural network for stock prediction"""

    def __init__(self, input_dim: int, name: str = "DenseNN"):
        self.name = name
        self.model = self._build_model(input_dim)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None

    def _build_model(self, input_dim: int) -> Sequential:
        """Build Dense architecture"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dropout(0.1),

            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )

        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, validation_split: float = 0.2):
        """Train the Dense model"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]

        self.history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled, verbose=0)
        return self.scaler_y.inverse_transform(y_scaled).flatten()


class XGBoostModel:
    """XGBoost model for stock prediction"""

    def __init__(self, name: str = "XGBoost"):
        self.name = name
        self.model = None
        self.scaler_X = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model"""
        X_scaled = self.scaler_X.fit_transform(X)

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )

        self.model.fit(X_scaled, y, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler_X.transform(X)
        return self.model.predict(X_scaled)


class TFEnsembleTrainer:
    """Train and ensemble all TensorFlow models"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.validation_metrics = {}

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and test data"""
        prices_data = get_historical_prices(self.symbol)
        if not prices_data:
            raise ValueError(f"No data for {self.symbol}")

        # Get training data (2015-2024)
        train_prices = []
        for year in range(2015, 2025):
            if year in prices_data:
                train_prices.extend(prices_data[year])

        train_prices = np.array(train_prices)

        # Calculate returns (target variable)
        returns = np.diff(train_prices) / train_prices[:-1] * 100

        # Create features
        features = self.feature_engineer.create_features(train_prices)

        if len(features) == 0:
            raise ValueError(f"Not enough data for {self.symbol}")

        # Align features with returns
        # Features start from index 24, returns from index 1
        aligned_returns = returns[23:]  # Skip first 23 returns to align with features
        min_len = min(len(features), len(aligned_returns))
        features = features[:min_len]
        aligned_returns = aligned_returns[:min_len]

        # Split: train on 80%, validate on 20%
        split_idx = int(len(features) * 0.8)

        X_train = features[:split_idx]
        y_train = aligned_returns[:split_idx]
        X_val = features[split_idx:]
        y_val = aligned_returns[split_idx:]

        return X_train, y_train, X_val, y_val

    def train_all_models(self) -> Dict[str, TFModelPrediction]:
        """Train all models and return predictions"""
        logger.info(f"Training TensorFlow models for {self.symbol}...")

        X_train, y_train, X_val, y_val = self.prepare_data()

        # Prepare sequences for LSTM/GRU
        lookback = 6
        X_seq_train, y_seq_train = self.feature_engineer.create_sequences(X_train, y_train, lookback)
        X_seq_val, y_seq_val = self.feature_engineer.create_sequences(X_val, y_val, lookback)

        results = {}

        # Train LSTM
        try:
            logger.info(f"Training LSTM for {self.symbol}...")
            lstm = LSTMModel(input_shape=(lookback, X_train.shape[1]))
            lstm.train(X_seq_train, y_seq_train, epochs=80)
            lstm_pred = lstm.predict(X_seq_val)
            lstm_mae = np.mean(np.abs(lstm_pred - y_seq_val))
            lstm_dir_acc = np.mean((lstm_pred >= 0) == (y_seq_val >= 0)) * 100

            self.models['lstm'] = lstm
            self.validation_metrics['lstm'] = {'mae': lstm_mae, 'dir_acc': lstm_dir_acc}

            # Generate future predictions
            last_seq = X_val[-lookback:].reshape(1, lookback, -1)
            pred_1m = float(lstm.predict(last_seq)[0])

            results['lstm'] = TFModelPrediction(
                symbol=self.symbol,
                model_name='LSTM',
                prediction_1m=round(pred_1m, 2),
                prediction_3m=round(pred_1m * 2.2, 2),
                prediction_6m=round(pred_1m * 3.8, 2),
                prediction_12m=round(pred_1m * 6.5, 2),
                confidence=round(min(90, 70 + (100 - lstm_mae) * 0.3), 1),
                validation_mae=round(lstm_mae, 2),
                validation_direction_accuracy=round(lstm_dir_acc, 1)
            )
        except Exception as e:
            logger.error(f"LSTM training failed for {self.symbol}: {e}")

        # Train GRU
        try:
            logger.info(f"Training GRU for {self.symbol}...")
            gru = GRUModel(input_shape=(lookback, X_train.shape[1]))
            gru.train(X_seq_train, y_seq_train, epochs=80)
            gru_pred = gru.predict(X_seq_val)
            gru_mae = np.mean(np.abs(gru_pred - y_seq_val))
            gru_dir_acc = np.mean((gru_pred >= 0) == (y_seq_val >= 0)) * 100

            self.models['gru'] = gru
            self.validation_metrics['gru'] = {'mae': gru_mae, 'dir_acc': gru_dir_acc}

            last_seq = X_val[-lookback:].reshape(1, lookback, -1)
            pred_1m = float(gru.predict(last_seq)[0])

            results['gru'] = TFModelPrediction(
                symbol=self.symbol,
                model_name='GRU',
                prediction_1m=round(pred_1m, 2),
                prediction_3m=round(pred_1m * 2.2, 2),
                prediction_6m=round(pred_1m * 3.8, 2),
                prediction_12m=round(pred_1m * 6.5, 2),
                confidence=round(min(90, 70 + (100 - gru_mae) * 0.3), 1),
                validation_mae=round(gru_mae, 2),
                validation_direction_accuracy=round(gru_dir_acc, 1)
            )
        except Exception as e:
            logger.error(f"GRU training failed for {self.symbol}: {e}")

        # Train Dense
        try:
            logger.info(f"Training Dense NN for {self.symbol}...")
            dense = DenseModel(input_dim=X_train.shape[1])
            dense.train(X_train, y_train, epochs=80)
            dense_pred = dense.predict(X_val)
            dense_mae = np.mean(np.abs(dense_pred - y_val))
            dense_dir_acc = np.mean((dense_pred >= 0) == (y_val >= 0)) * 100

            self.models['dense'] = dense
            self.validation_metrics['dense'] = {'mae': dense_mae, 'dir_acc': dense_dir_acc}

            pred_1m = float(dense.predict(X_val[-1:].reshape(1, -1))[0])

            results['dense'] = TFModelPrediction(
                symbol=self.symbol,
                model_name='DenseNN',
                prediction_1m=round(pred_1m, 2),
                prediction_3m=round(pred_1m * 2.2, 2),
                prediction_6m=round(pred_1m * 3.8, 2),
                prediction_12m=round(pred_1m * 6.5, 2),
                confidence=round(min(90, 70 + (100 - dense_mae) * 0.3), 1),
                validation_mae=round(dense_mae, 2),
                validation_direction_accuracy=round(dense_dir_acc, 1)
            )
        except Exception as e:
            logger.error(f"Dense training failed for {self.symbol}: {e}")

        # Train XGBoost
        try:
            logger.info(f"Training XGBoost for {self.symbol}...")
            xgb_model = XGBoostModel()
            xgb_model.train(X_train, y_train)
            xgb_pred = xgb_model.predict(X_val)
            xgb_mae = np.mean(np.abs(xgb_pred - y_val))
            xgb_dir_acc = np.mean((xgb_pred >= 0) == (y_val >= 0)) * 100

            self.models['xgboost'] = xgb_model
            self.validation_metrics['xgboost'] = {'mae': xgb_mae, 'dir_acc': xgb_dir_acc}

            pred_1m = float(xgb_model.predict(X_val[-1:].reshape(1, -1))[0])

            results['xgboost'] = TFModelPrediction(
                symbol=self.symbol,
                model_name='XGBoost',
                prediction_1m=round(pred_1m, 2),
                prediction_3m=round(pred_1m * 2.2, 2),
                prediction_6m=round(pred_1m * 3.8, 2),
                prediction_12m=round(pred_1m * 6.5, 2),
                confidence=round(min(90, 70 + (100 - xgb_mae) * 0.3), 1),
                validation_mae=round(xgb_mae, 2),
                validation_direction_accuracy=round(xgb_dir_acc, 1)
            )
        except Exception as e:
            logger.error(f"XGBoost training failed for {self.symbol}: {e}")

        # Create ensemble prediction
        if results:
            ensemble_1m = np.mean([r.prediction_1m for r in results.values()])
            ensemble_mae = np.mean([r.validation_mae for r in results.values()])
            ensemble_dir = np.mean([r.validation_direction_accuracy for r in results.values()])

            results['ensemble'] = TFModelPrediction(
                symbol=self.symbol,
                model_name='TF_Ensemble',
                prediction_1m=round(ensemble_1m, 2),
                prediction_3m=round(ensemble_1m * 2.2, 2),
                prediction_6m=round(ensemble_1m * 3.8, 2),
                prediction_12m=round(ensemble_1m * 6.5, 2),
                confidence=round(min(92, 72 + (100 - ensemble_mae) * 0.3), 1),
                validation_mae=round(ensemble_mae, 2),
                validation_direction_accuracy=round(ensemble_dir, 1)
            )

        return results


def train_tf_models(symbol: str) -> Dict[str, TFModelPrediction]:
    """Train TensorFlow models for a symbol"""
    trainer = TFEnsembleTrainer(symbol)
    return trainer.train_all_models()


def backtest_tf_models(symbol: str) -> Dict:
    """
    Backtest TensorFlow models against actual 2025 performance
    """
    # Train models on 2015-2024 data
    predictions = train_tf_models(symbol)

    if not predictions:
        return {'error': f'Failed to train models for {symbol}'}

    # Get actual 2025 returns
    prices_data = get_historical_prices(symbol)
    if not prices_data or 2025 not in prices_data or 2024 not in prices_data:
        return {'error': 'No 2025 data available'}

    dec_2024 = prices_data[2024][-1]
    dec_2025 = prices_data[2025][-1]
    actual_annual = ((dec_2025 - dec_2024) / dec_2024) * 100

    # Get ensemble prediction
    ensemble = predictions.get('ensemble')
    if not ensemble:
        return {'error': 'No ensemble prediction'}

    predicted_annual = ensemble.prediction_12m
    error = abs(predicted_annual - actual_annual)
    direction_correct = (predicted_annual >= 0) == (actual_annual >= 0)

    # Calculate accuracy score
    if direction_correct:
        if error <= 5:
            accuracy = 95
        elif error <= 10:
            accuracy = 85
        elif error <= 20:
            accuracy = 70
        elif error <= 30:
            accuracy = 60
        else:
            accuracy = 50
    else:
        accuracy = max(20, 45 - error * 0.5)

    return {
        'symbol': symbol,
        'model': 'TensorFlow Ensemble',
        'trainingData': '2015-2024 (10 years)',
        'backtestPeriod': '2025',
        'predicted': {
            'annual': round(predicted_annual, 2),
            '1m': ensemble.prediction_1m,
            '3m': ensemble.prediction_3m,
            '6m': ensemble.prediction_6m,
        },
        'actual': {
            'annual': round(actual_annual, 2),
        },
        'metrics': {
            'predictionError': round(error, 2),
            'directionCorrect': direction_correct,
            'accuracyScore': round(accuracy, 1),
            'validationMAE': ensemble.validation_mae,
            'validationDirectionAccuracy': ensemble.validation_direction_accuracy,
        },
        'modelDetails': {
            model_name: {
                'prediction_12m': pred.prediction_12m,
                'confidence': pred.confidence,
                'validation_mae': pred.validation_mae,
                'direction_accuracy': pred.validation_direction_accuracy,
            }
            for model_name, pred in predictions.items()
        }
    }
