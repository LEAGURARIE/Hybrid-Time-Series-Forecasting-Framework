"""
Hybrid models module for Google Stock ML project.
Matches notebook Cell 72 (BLOCK 28) and Cell 74 (BLOCK 29) exactly.
Both models (Sequential and Parallel) run in a single loop.

Architectures:
- hybrid_seq: Input → LSTM → LayerNorm → GRU → Dense → Output
- hybrid_par: Input → [LSTM, GRU] → Concat → Dense → Output
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from .neural_utils import train_single_nn_model, predict_single_nn_model


def _build_hybrid_seq_model(inp, n_features, config):
    """Build Hybrid Sequential architecture: LSTM → GRU."""
    from tensorflow.keras import layers
    
    LSTM_UNITS = int(config["lstm_units"])
    GRU_UNITS = int(config["gru_units"])
    DENSE_UNITS = int(config["dense_units"])
    DROPOUT = float(config["dropout"])
    DENSE_ACT = config["dense_activation"]
    OUTPUT_ACT = config["output_activation"]
    
    x = layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=DROPOUT)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.GRU(GRU_UNITS, return_sequences=False, dropout=DROPOUT)(x)
    x = layers.Dense(DENSE_UNITS, activation=DENSE_ACT)(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation=OUTPUT_ACT)(x)
    
    return out


def _build_hybrid_par_model(inp, n_features, config):
    """Build Hybrid Parallel architecture: LSTM ∥ GRU → Concat."""
    from tensorflow.keras import layers
    
    LSTM_UNITS = int(config["lstm_units"])
    GRU_UNITS = int(config["gru_units"])
    DENSE_UNITS = int(config["dense_units"])
    DROPOUT = float(config["dropout"])
    DENSE_ACT = config["dense_activation"]
    OUTPUT_ACT = config["output_activation"]
    
    lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=False, dropout=DROPOUT)(inp)
    gru_out = layers.GRU(GRU_UNITS, return_sequences=False, dropout=DROPOUT)(inp)
    x = layers.Concatenate()([lstm_out, gru_out])
    x = layers.Dense(DENSE_UNITS, activation=DENSE_ACT)(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation=OUTPUT_ACT)(x)
    
    return out


# Model builders mapping
_MODEL_BUILDERS = {
    "hybrid_seq": _build_hybrid_seq_model,
    "hybrid_par": _build_hybrid_par_model,
}


def train_hybrid(
    X_train_dict: Dict[str, pd.DataFrame],
    X_valid_dict: Dict[str, pd.DataFrame],
    X_test_dict: Dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    w_test: np.ndarray,
    hybrid_types: List[str],
    config: Dict,
    hpo_config: Dict,
    models_out_local: Optional[Path] = None,
    models_out_drive: Optional[Path] = None,
    pred_out_local: Optional[Path] = None,
    pred_out_drive: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Train Hybrid Sequential and Parallel models in a single loop.
    Matches notebook Cell 72 (BLOCK 28) exactly.
    
    Args:
        X_train_dict, X_valid_dict, X_test_dict: {feature_set: DataFrame}
        y_train, y_valid, y_test: Target arrays
        w_train, w_valid, w_test: Weight arrays
        hybrid_types: List of models to train (e.g., ["hybrid_seq", "hybrid_par"])
        config: RUN_PARAMS containing hybrid_seq/hybrid_par configs
        hpo_config: RUN_PARAMS["hpo"]
        models_out_local/drive: Model output directories
        pred_out_local/drive: Predictions output directories
    
    Returns:
        Dict with results for each hybrid type
    """
    all_results = {}
    
    for hybrid_type in hybrid_types:
        if hybrid_type not in _MODEL_BUILDERS:
            print(f"[WARN] Unknown hybrid type: {hybrid_type}, skipping...")
            continue
        
        hyb_config = config.get(hybrid_type)
        if hyb_config is None:
            print(f"[WARN] No config for {hybrid_type}, skipping...")
            continue
        
        arch_name = "Sequential (LSTM→GRU)" if hybrid_type == "hybrid_seq" else "Parallel (LSTM∥GRU)"
        print(f"\n[INFO] Training {hybrid_type.upper()} — {arch_name}")
        
        result = train_single_nn_model(
            model_type=hybrid_type,
            X_train_dict=X_train_dict,
            X_valid_dict=X_valid_dict,
            X_test_dict=X_test_dict,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            w_train=w_train,
            w_valid=w_valid,
            w_test=w_test,
            nn_config=hyb_config,
            hpo_config=hpo_config,
            build_model_fn=_MODEL_BUILDERS[hybrid_type],
            models_out_local=models_out_local,
            models_out_drive=models_out_drive,
            pred_out_local=pred_out_local,
            pred_out_drive=pred_out_drive
        )
        
        if result is not None:
            all_results[hybrid_type] = result
    
    print("[OK] BLOCK 28 complete.")
    return all_results if all_results else None


def predict_hybrid(
    X_test_dict: Dict[str, pd.DataFrame],
    y_test: np.ndarray,
    w_test: np.ndarray,
    hybrid_types: List[str],
    config: Dict,
    plot_config: Dict,
    models_dir_local: Path,
    pred_out_local: Optional[Path] = None,
    pred_out_drive: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate predictions for Hybrid models.
    Matches notebook Cell 74 (BLOCK 29) exactly.
    
    Returns:
        Dict with prediction results for each hybrid type
    """
    all_results = {}
    
    for hybrid_type in hybrid_types:
        hyb_config = config.get(hybrid_type)
        if hyb_config is None:
            print(f"[WARN] No config for {hybrid_type}, skipping...")
            continue
        
        arch_name = "Sequential (LSTM→GRU)" if hybrid_type == "hybrid_seq" else "Parallel (LSTM∥GRU)"
        print(f"\n[INFO] Predicting with {hybrid_type.upper()} — {arch_name}")
        
        result = predict_single_nn_model(
            model_type=hybrid_type,
            X_test_dict=X_test_dict,
            y_test=y_test,
            w_test=w_test,
            nn_config=hyb_config,
            plot_config=plot_config,
            models_dir_local=models_dir_local,
            pred_out_local=pred_out_local,
            pred_out_drive=pred_out_drive
        )
        
        if result is not None:
            all_results[hybrid_type] = result
    
    print("[OK] BLOCK 29 complete.")
    return all_results if all_results else None
