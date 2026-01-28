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

# TensorFlow models (imported conditionally to handle missing dependencies)
try:
    from .tf_models import (
        train_tf_models,
        backtest_tf_models,
        TFModelPrediction,
        TFEnsembleTrainer,
    )
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    train_tf_models = None
    backtest_tf_models = None
    TFModelPrediction = None
    TFEnsembleTrainer = None

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
    'train_tf_models',
    'backtest_tf_models',
    'TFModelPrediction',
    'TFEnsembleTrainer',
    'TF_AVAILABLE',
]
