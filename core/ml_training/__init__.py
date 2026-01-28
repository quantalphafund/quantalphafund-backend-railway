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
]
