"""
Advanced Deep Learning Models for Financial Prediction
LSTM, GRU, Transformer, and Temporal Fusion Transformer architectures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Note: These models require PyTorch. Install with: pip install torch
# For production, uncomment the torch imports and model implementations

@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    input_size: int = 50  # Number of features
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1  # Predict return
    dropout: float = 0.2
    sequence_length: int = 60  # 60 days lookback
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10

@dataclass
class TransformerConfig:
    """Configuration for Transformer models"""
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    sequence_length: int = 60
    output_size: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100

class FeatureEngineer:
    """
    Feature engineering for deep learning models
    Creates technical, fundamental, and derived features
    """

    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 60]):
        self.lookback_periods = lookback_periods

    def create_features(
        self,
        price_data: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models

        Args:
            price_data: DataFrame with OHLCV columns
            fundamental_data: Optional fundamental metrics dict

        Returns:
            DataFrame with engineered features
        """
        df = price_data.copy()
        features = pd.DataFrame(index=df.index)

        # Returns
        features['return_1d'] = df['close'].pct_change()
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)

        # Log returns
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility features
        for period in self.lookback_periods:
            features[f'volatility_{period}d'] = features['return_1d'].rolling(period).std()
            features[f'volatility_ratio_{period}d'] = (
                features['return_1d'].rolling(period).std() /
                features['return_1d'].rolling(period * 2).std()
            )

        # Price relative to moving averages
        for period in self.lookback_periods:
            ma = df['close'].rolling(period).mean()
            features[f'price_ma_ratio_{period}d'] = df['close'] / ma
            features[f'ma_slope_{period}d'] = ma.pct_change(5)

        # Momentum features
        for period in self.lookback_periods:
            features[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
            features[f'roc_{period}d'] = df['close'].diff(period) / df['close'].shift(period)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_normalized'] = (features['rsi_14'] - 50) / 50

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema_12 - ema_26) / df['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_upper'] = (bb_ma + 2 * bb_std - df['close']) / df['close']
        features['bb_lower'] = (df['close'] - (bb_ma - 2 * bb_std)) / df['close']
        features['bb_width'] = (4 * bb_std) / bb_ma
        features['bb_position'] = (df['close'] - bb_ma) / (2 * bb_std)

        # Volume features
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv_ma'] = features['obv'] / features['obv'].rolling(20).mean()

        # VWAP-related
        if 'vwap' in df.columns:
            features['vwap_ratio'] = df['close'] / df['vwap']

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean() / df['close']
        features['atr_ratio'] = features['atr'] / features['atr'].rolling(60).mean()

        # Price patterns
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        features['inside_bar'] = (
            (df['high'] < df['high'].shift(1)) &
            (df['low'] > df['low'].shift(1))
        ).astype(int)

        # Candle patterns
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
        features['is_bullish'] = (df['close'] > df['open']).astype(int)

        # 52-week metrics
        features['high_52w_ratio'] = df['close'] / df['high'].rolling(252).max()
        features['low_52w_ratio'] = df['close'] / df['low'].rolling(252).min()
        features['range_position'] = (
            (df['close'] - df['low'].rolling(252).min()) /
            (df['high'].rolling(252).max() - df['low'].rolling(252).min())
        )

        # Day of week encoding
        features['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        features['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)

        # Month encoding
        features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # Add fundamental features if provided
        if fundamental_data:
            for key, value in fundamental_data.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    features[f'fund_{key}'] = value

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)

        return features

    def create_sequences(
        self,
        features: pd.DataFrame,
        target_column: str = 'return_1d',
        sequence_length: int = 60,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer training

        Returns:
            X: Shape (samples, sequence_length, features)
            y: Shape (samples,)
        """
        data = features.values
        target_idx = features.columns.get_loc(target_column) if target_column in features.columns else 0

        X, y = [], []

        for i in range(sequence_length, len(data) - forecast_horizon):
            X.append(data[i-sequence_length:i])
            y.append(data[i + forecast_horizon - 1, target_idx])

        return np.array(X), np.array(y)


class BaseDeepModel(ABC):
    """Base class for deep learning models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def prepare_data(
        self,
        price_data: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        features = self.feature_engineer.create_features(price_data, fundamental_data)
        X, y = self.feature_engineer.create_sequences(
            features,
            sequence_length=self.config.sequence_length
        )
        return X, y


class LSTMModel(BaseDeepModel):
    """
    LSTM-based model for financial time series prediction
    Captures long-term dependencies in sequential data
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = None

    def build_model(self):
        """Build LSTM architecture"""
        # PyTorch implementation placeholder
        # Uncomment and use when PyTorch is installed

        """
        import torch
        import torch.nn as nn

        class LSTMNetwork(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    batch_first=True,
                    dropout=config.dropout if config.num_layers > 1 else 0,
                    bidirectional=True
                )
                self.attention = nn.MultiheadAttention(
                    config.hidden_size * 2,
                    num_heads=4,
                    dropout=config.dropout
                )
                self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
                self.fc2 = nn.Linear(config.hidden_size, config.output_size)
                self.dropout = nn.Dropout(config.dropout)
                self.layer_norm = nn.LayerNorm(config.hidden_size * 2)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)

                # Self-attention on LSTM outputs
                lstm_out = lstm_out.permute(1, 0, 2)  # (seq, batch, features)
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                attn_out = attn_out.permute(1, 0, 2)  # (batch, seq, features)

                # Take last timestep
                out = self.layer_norm(attn_out[:, -1, :])
                out = self.dropout(torch.relu(self.fc1(out)))
                out = self.fc2(out)
                return out

        self.model = LSTMNetwork(self.config)
        """

        self.logger.info("LSTM model built (placeholder - requires PyTorch)")
        return self

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train the LSTM model"""
        # Placeholder for PyTorch training loop
        self.logger.info(f"Training LSTM on {len(X)} samples...")

        # Normalize data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        # Reshape for scaling
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_flat)
        X = X_scaled.reshape(original_shape)

        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(torch.FloatTensor(X_val))
                val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_val))

            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss.item()
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        """

        self.is_trained = True
        self.logger.info("LSTM training complete (placeholder)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.scaler:
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X = X_scaled.reshape(original_shape)

        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X))
        return predictions.numpy().flatten()
        """

        # Placeholder: return simple prediction
        return np.zeros(len(X))


class TransformerModel(BaseDeepModel):
    """
    Transformer-based model for financial prediction
    Uses attention mechanisms to capture complex patterns
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config

    def build_model(self):
        """Build Transformer architecture"""
        """
        import torch
        import torch.nn as nn

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() *
                    (-np.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class FinancialTransformer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config

                # Input projection
                self.input_projection = nn.Linear(config.input_size, config.d_model)

                # Positional encoding
                self.pos_encoder = PositionalEncoding(config.d_model)

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=config.num_encoder_layers
                )

                # Output layers
                self.fc1 = nn.Linear(config.d_model, config.d_model // 2)
                self.fc2 = nn.Linear(config.d_model // 2, config.output_size)
                self.dropout = nn.Dropout(config.dropout)
                self.layer_norm = nn.LayerNorm(config.d_model)

            def forward(self, x):
                # Project input
                x = self.input_projection(x)
                x = self.pos_encoder(x)

                # Transformer encoding
                x = self.transformer_encoder(x)

                # Global average pooling + last timestep
                x = x.mean(dim=1) + x[:, -1, :]
                x = self.layer_norm(x)

                # Output
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                return x

        self.model = FinancialTransformer(self.config)
        """

        self.logger.info("Transformer model built (placeholder - requires PyTorch)")
        return self

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train the Transformer model"""
        self.logger.info(f"Training Transformer on {len(X)} samples...")

        # Similar training loop as LSTM
        # Placeholder implementation

        self.is_trained = True
        self.logger.info("Transformer training complete (placeholder)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.zeros(len(X))


class TemporalFusionTransformer(BaseDeepModel):
    """
    Temporal Fusion Transformer (TFT)
    State-of-the-art architecture for interpretable time series forecasting
    Combines LSTM for local patterns with attention for long-range dependencies
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config

    def build_model(self):
        """Build TFT architecture"""
        """
        Key components:
        1. Variable Selection Networks - identify important features
        2. LSTM Encoder/Decoder - capture local temporal patterns
        3. Static Enrichment - incorporate static features
        4. Temporal Self-Attention - long-range dependencies
        5. Gating Mechanisms - control information flow
        """

        self.logger.info("TFT model built (placeholder - requires PyTorch)")
        return self

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train the TFT model"""
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.zeros(len(X))

    def get_attention_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get attention weights for interpretability"""
        return {}


class EnsembleDeepModel:
    """
    Ensemble of deep learning models
    Combines LSTM, Transformer, and TFT for robust predictions
    """

    def __init__(self):
        self.models: Dict[str, BaseDeepModel] = {}
        self.weights: Dict[str, float] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_model(self, name: str, model: BaseDeepModel, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[name] = model
        self.weights[name] = weight

    def train_all(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train all models"""
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            model.build_model()
            model.train(X, y, validation_split)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted ensemble prediction

        Returns:
            Weighted average of all model predictions
        """
        predictions = []
        total_weight = 0

        for name, model in self.models.items():
            if model.is_trained:
                pred = model.predict(X)
                weight = self.weights.get(name, 1.0)
                predictions.append(pred * weight)
                total_weight += weight

        if not predictions:
            return np.zeros(len(X))

        return np.sum(predictions, axis=0) / total_weight

    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual model predictions"""
        return {
            name: model.predict(X)
            for name, model in self.models.items()
            if model.is_trained
        }
