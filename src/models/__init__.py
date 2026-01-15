"""
Models module for Google Stock ML project.

Tree-based models:
- xgboost_model: XGBoost regressor
- lightgbm_model: LightGBM regressor

Neural network models:
- lstm_gru: LSTM and GRU networks (single loop)
- hybrid: Hybrid Sequential and Parallel (single loop)

Ensemble:
- ensemble: Combines predictions from all models
"""

# Tree-based models
from .xgboost_model import train_final_model, split_valid_es_score
from .lightgbm_model import train_final_model_lgb, split_valid_es_score_lgb, get_default_lgb_params

# Neural network utilities
from .neural_utils import (
    make_sequences_eod_nn,
    make_sequences_pred_nn,
    split_valid_es_score as split_valid_es_score_nn,
    train_single_nn_model,
    predict_single_nn_model
)

# Neural network models
from .lstm_gru import train_lstm_gru, predict_lstm_gru
from .hybrid import train_hybrid, predict_hybrid

# Ensemble
from .ensemble import (
    run_ensemble,
    load_model_predictions,
    load_model_metrics,
    simple_average,
    weighted_average,
    stacking,
    rank_average
)

__all__ = [
    # XGBoost
    "train_final_model",
    "split_valid_es_score",
    # LightGBM
    "train_final_model_lgb",
    "split_valid_es_score_lgb",
    "get_default_lgb_params",
    # Neural utilities
    "make_sequences_eod_nn",
    "make_sequences_pred_nn",
    "split_valid_es_score_nn",
    "train_single_nn_model",
    "predict_single_nn_model",
    # LSTM + GRU
    "train_lstm_gru",
    "predict_lstm_gru",
    # Hybrid (Sequential + Parallel)
    "train_hybrid",
    "predict_hybrid",
    # Ensemble
    "run_ensemble",
    "load_model_predictions",
    "load_model_metrics",
    "simple_average",
    "weighted_average",
    "stacking",
    "rank_average",
]
