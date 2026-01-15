"""
LSTM and GRU models module for Google Stock ML project.
Matches notebook Cell 67 (BLOCK 26) and Cell 69 (BLOCK 27) exactly.
Both models run in a single loop.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from .neural_utils import train_single_nn_model, predict_single_nn_model


def _build_lstm_model(inp, n_features, config):
    """Build LSTM architecture."""
    from tensorflow.keras import layers
    
    UNITS_1 = int(config["units_1"])
    UNITS_2 = int(config["units_2"])
    DENSE_UNITS = int(config["dense_units"])
    DROPOUT = float(config["dropout"])
    DENSE_ACT = config["dense_activation"]
    OUTPUT_ACT = config["output_activation"]
    
    x = layers.LSTM(UNITS_1, return_sequences=True, dropout=DROPOUT)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(UNITS_2, return_sequences=False, dropout=DROPOUT)(x)
    x = layers.Dense(DENSE_UNITS, activation=DENSE_ACT)(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation=OUTPUT_ACT)(x)
    
    return out


def _build_gru_model(inp, n_features, config):
    """Build GRU architecture."""
    from tensorflow.keras import layers
    
    UNITS_1 = int(config["units_1"])
    UNITS_2 = int(config["units_2"])
    DENSE_UNITS = int(config["dense_units"])
    DROPOUT = float(config["dropout"])
    DENSE_ACT = config["dense_activation"]
    OUTPUT_ACT = config["output_activation"]
    
    x = layers.GRU(UNITS_1, return_sequences=True, dropout=DROPOUT)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.GRU(UNITS_2, return_sequences=False, dropout=DROPOUT)(x)
    x = layers.Dense(DENSE_UNITS, activation=DENSE_ACT)(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation=OUTPUT_ACT)(x)
    
    return out


# Model builders mapping
_MODEL_BUILDERS = {
    "lstm": _build_lstm_model,
    "gru": _build_gru_model,
}


def train_lstm_gru(
    X_train_dict: Dict[str, pd.DataFrame],
    X_valid_dict: Dict[str, pd.DataFrame],
    X_test_dict: Dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    w_test: np.ndarray,
    model_types: List[str],
    config: Dict,
    hpo_config: Dict,
    models_out_local: Optional[Path] = None,
    models_out_drive: Optional[Path] = None,
    pred_out_local: Optional[Path] = None,
    pred_out_drive: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Train LSTM and GRU models in a single loop.
    Matches notebook Cell 67 (BLOCK 26) exactly.
    
    Args:
        X_train_dict, X_valid_dict, X_test_dict: {feature_set: DataFrame}
        y_train, y_valid, y_test: Target arrays
        w_train, w_valid, w_test: Weight arrays
        model_types: List of models to train (e.g., ["lstm", "gru"])
        config: RUN_PARAMS containing lstm/gru configs
        hpo_config: RUN_PARAMS["hpo"]
        models_out_local/drive: Model output directories
        pred_out_local/drive: Predictions output directories
    
    Returns:
        Dict with results for each model type
    """
    all_results = {}
    
    for model_type in model_types:
        if model_type not in _MODEL_BUILDERS:
            print(f"[WARN] Unknown model type: {model_type}, skipping...")
            continue
        
        nn_config = config.get(model_type)
        if nn_config is None:
            print(f"[WARN] No config for {model_type}, skipping...")
            continue
        
        result = train_single_nn_model(
            model_type=model_type,
            X_train_dict=X_train_dict,
            X_valid_dict=X_valid_dict,
            X_test_dict=X_test_dict,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            w_train=w_train,
            w_valid=w_valid,
            w_test=w_test,
            nn_config=nn_config,
            hpo_config=hpo_config,
            build_model_fn=_MODEL_BUILDERS[model_type],
            models_out_local=models_out_local,
            models_out_drive=models_out_drive,
            pred_out_local=pred_out_local,
            pred_out_drive=pred_out_drive
        )
        
        if result is not None:
            all_results[model_type] = result
    
    print("[OK] BLOCK 26 complete.")
    return all_results if all_results else None


def predict_lstm_gru(
    X_test_dict: Dict[str, pd.DataFrame],
    y_test: np.ndarray,
    w_test: np.ndarray,
    model_types: List[str],
    config: Dict,
    plot_config: Dict,
    models_dir_local: Path,
    pred_out_local: Optional[Path] = None,
    pred_out_drive: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate predictions for LSTM and GRU models.
    Matches notebook Cell 69 (BLOCK 27) exactly.
    
    Returns:
        Dict with prediction results for each model type
    """
    all_results = {}
    
    for model_type in model_types:
        nn_config = config.get(model_type)
        if nn_config is None:
            print(f"[WARN] No config for {model_type}, skipping...")
            continue
        
        result = predict_single_nn_model(
            model_type=model_type,
            X_test_dict=X_test_dict,
            y_test=y_test,
            w_test=w_test,
            nn_config=nn_config,
            plot_config=plot_config,
            models_dir_local=models_dir_local,
            pred_out_local=pred_out_local,
            pred_out_drive=pred_out_drive
        )
        
        if result is not None:
            all_results[model_type] = result
    
    print("[OK] BLOCK 27 complete.")
    return all_results if all_results else None
