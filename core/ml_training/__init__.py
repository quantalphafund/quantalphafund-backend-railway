"""ML Training Module"""
from .historical_data import (
    get_historical_prices,
    get_training_data,
    get_2025_actual_returns,
    HISTORICAL_DATA,
)
from .model_trainer import (
    train_and_predict,
    backtest_2025,
    train_all_models,
    get_trained_predictions,
    ModelPrediction,
    BacktestResult,
)

# Advanced ML models (sklearn neural networks + XGBoost + ensemble)
from .advanced_models import (
    train_advanced_models,
    backtest_advanced_models,
    AdvancedPrediction,
    AdvancedEnsembleTrainer,
)

# For backward compatibility
TF_AVAILABLE = True  # We now use sklearn which is always available
train_tf_models = train_advanced_models
backtest_tf_models = backtest_advanced_models

__all__ = [
    'get_historical_prices',
    'get_training_data',
    'get_2025_actual_returns',
    'HISTORICAL_DATA',
    'train_and_predict',
    'backtest_2025',
    'train_all_models',
    'get_trained_predictions',
    'ModelPrediction',
    'BacktestResult',
    'train_advanced_models',
    'backtest_advanced_models',
    'AdvancedPrediction',
    'AdvancedEnsembleTrainer',
    'train_tf_models',
    'backtest_tf_models',
    'TF_AVAILABLE',
]
