"""
Configuration module for Google Stock ML project.
Contains all hyperparameters, paths, and run settings.

Usage:
    from src.config import CONFIG
    
    # Access any parameter:
    train_end = CONFIG["data"]["train_end"]
    lstm_epochs = CONFIG["lstm"]["epochs"]
    
    # Modify parameters here, not in CLI
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict


# =============================================================================
# MAIN CONFIGURATION - EDIT HERE
# =============================================================================

CONFIG = {
    "random_state": 42,
    "output_dir": "./output",
    "run_id": None,  # None = auto-generate timestamp
    
    # --- Data ---
    "price_tickers": ["GOOGL", "MSFT", "NVDA", "^IXIC", "SPY", "QQQ", "^VIX", "^TNX", "XLK", "^GDAXI"],
    
    "data": {
        "start_date": "2023-11-20",
        "end_date": "2026-01-15",  # Set fixed date for reproducibility; use None for today's date
        "limit_start_date": "2023-12-31",
        "train_end": "2025-05-20",
        "valid_start": "2025-05-21",
        "valid_end": "2025-09-10",
        "test_start": "2025-09-11",
        "target_src_col": "GOOGL_logret_cc",
        "target_col": "GOOGL_logret_t1",
    },
    
    # --- Features ---
    "features": {
        "rolling_w_short": 5,
        "rolling_w_long": 21,
        "do_volume_rolling": True,
        "cross_asset_base": "GOOGL",
        "cross_asset_peers": ["SPY", "QQQ", "^IXIC", "XLK"],
        "cross_asset_windows": [5, 21],
        "regime_base": "GOOGL",
        "market_vol_ticker": "SPY",
        "exclude_raw_ohlc": ["^VIX", "^TNX", "^GDAXI"],
        "drop_volume_tickers": ["^VIX", "^TNX", "^GDAXI"],  # Drop raw Volume columns for these tickers
        "covid_start": "2020-02-01",
        "covid_end": "2023-05-05",
        "crisis_2008_start": "2007-07-01",
        "crisis_2008_end": "2009-09-01",
    },
    
    "eu_break_close": {
        "enabled": True,
        "eu_ticker": "^GDAXI",
        "gap_days_threshold": 2,
        "apply_to": "next_us_trading_day",
    },
    
    # --- Weights ---
    "weights": {"c": 1.0, "max_w": 4.0},
    
    # --- XGBoost Feature Selection ---
    "xgb_fs": {
        "spearman_thresh": 0.85,
        "gain_cum_thresh": 0.85,
        "min_features": 10,
        "neg_sigma": 1.2,
        "pos_sigma": 0.7,
        "min_gain": 0.001,
        "perm_repeats": 15,
        "n_estimators": 1000,
        "early_stopping_rounds": 40,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 15,
        "gamma": 1,
        "subsample": 0.60,
        "colsample_bytree": 0.60,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "max_delta_step": 1,
        "random_state": 42,
    },
    
    # --- HPO ---
    "hpo": {
        "n_estimators": 1500,
        "early_stopping_rounds": 40,
        "n_trials_stage1": 120,
        "n_trials_stage2": 60,
        "n_trials_stage2_lowlr": 30,
        "print_every_stage1": 10,
        "print_every_stage2": 20,
        "tie_tol": 1e-5,
        "valid_es_start": "2025-05-21",
        "valid_es_end": "2025-07-15",
        "valid_score_start": "2025-07-16",
        "valid_score_end": "2025-09-10",
        "random_state": 42,
        "lookback": 7,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        # Sampling ranges for HPO
        "sampling": {
            "broad": {
                "max_depth": [2, 4],
                "min_child_weight_log": [2.0, 15.0],
                "subsample": [0.6, 0.85],
                "colsample_bytree": [0.6, 0.85],
                "gamma": [0.5, 4.0],
                "reg_alpha_exp": [-6, -2],
                "reg_lambda_exp": [0.0, 2.0],
                "max_delta_step": [0.0, 1.0],
                "lr_high_prob": 0.10,
                "lr_high": [0.03, 0.06],
                "lr_low": [0.005, 0.03],
            },
            "refine": {
                "max_depth_delta": [-1, 1],
                "max_depth_clip": [2, 5],
                "lr_sigma": 0.20,
                "lr_clip": [0.005, 0.08],
                "min_child_weight_sigma": 0.30,
                "min_child_weight_clip": [5.0, 20.0],
                "subsample_sigma": 0.05,
                "subsample_clip": [0.6, 0.9],
                "colsample_sigma": 0.05,
                "colsample_clip": [0.6, 0.9],
                "gamma_sigma": 0.25,
                "gamma_clip": [0.5, 4.0],
                "reg_alpha_sigma": 0.5,
                "reg_alpha_exp_clip": [-7, -1],
                "reg_lambda_sigma": 0.4,
                "reg_lambda_exp_clip": [0.0, 2.0],
                "max_delta_step_sigma": 0.15,
                "max_delta_step_clip": [0.0, 1.0],
            },
            "refine_low_lr": {
                "lr_shift": -0.8,
                "lr_clip": [0.0015, 0.06],
            },
        },
    },
    
    # --- SHAP ---
    "shap": {
        "enabled": True,
        "top_n_features": None,  # None = use xgb_selected, 10 = use top 10 SHAP features
        "max_display": 20,
        "save_values": True,
    },
    
    # --- LSTM ---
    "lstm": {
        "lookback": 7,
        "stride": 1,
        "units_1": 8,
        "units_2": 4,
        "dense_units": 1,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "clipnorm": 1.0,
        "epochs": 50,
        "batch_size": 4,
        "patience": 5,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80"],
    },
    
    # --- GRU ---
    "gru": {
        "lookback": 7,
        "stride": 1,
        "units_1": 8,
        "units_2": 4,
        "dense_units": 1,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "clipnorm": 1.0,
        "epochs": 50,
        "batch_size": 4,
        "patience": 5,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80"],
    },
    
    # --- Hybrid Sequential ---
    "hybrid_seq": {
        "lookback": 7,
        "stride": 1,
        "lstm_units": 8,
        "gru_units": 4,
        "dense_units": 1,
        "dropout": 0.20,
        "learning_rate": 4e-4,
        "clipnorm": 1.0,
        "epochs": 50,
        "batch_size": 4,
        "patience": 6,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80"],
    },
    
    # --- Hybrid Parallel ---
    "hybrid_par": {
        "lookback": 7,
        "stride": 1,
        "lstm_units": 8,
        "gru_units": 8,
        "dense_units": 1,
        "dropout": 0.20,
        "learning_rate": 4e-4,
        "clipnorm": 1.0,
        "epochs": 50,
        "batch_size": 4,
        "patience": 6,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80"],
    },
    
    # --- Ensemble ---
    "ensemble": {
        "method": "weighted_average",  # simple_average, weighted_average, stacking, rank_average
        "models": ["xgb", "lgb", "lstm", "gru", "hybrid_seq", "hybrid_par"],  # or null for all
        "filter": {
            "min_diracc": 0.55,  # remove models with DirAcc < threshold
            "max_wrmse": 0.020,  # remove models with wRMSE > threshold
            "top_n": 4,          # keep only top N models by wRMSE
        },
        "weights": "auto",  # "auto" for inverse_wrmse, or dict of manual weights
        "weight_method": "inverse_wrmse_squared",  # inverse_wrmse_squared gives advantage to stable models
        "meta_model": "ridge",  # for stacking: ridge, lasso, elasticnet, xgb, lgb
        "meta_params": {
            "alpha": 5.0,  # higher regularization for small data
            "l1_ratio": 0.5,  # for elasticnet
            "n_estimators": 100,  # for xgb/lgb
            "max_depth": 3,  # for xgb/lgb
            "learning_rate": 0.1,  # for xgb/lgb
            "random_state": 42,
        },
        "compare_methods": False,  # run comparison of all methods
    },
}


# =============================================================================
# CLI PATHS SETUP
# =============================================================================

def setup_cli_paths(output_dir: str = None, run_id: str = None) -> Dict[str, Any]:
    """Setup paths for CLI (local only, no Drive)."""
    BASE_DIR = Path(output_dir or CONFIG["output_dir"]).resolve()
    RUN_ID = run_id or CONFIG["run_id"] or datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = BASE_DIR / "runs" / RUN_ID
    
    paths = {
        "base": BASE_DIR,
        "run_id": RUN_ID,
        "run_dir": RUN_DIR,
        "raw": BASE_DIR / "data" / "raw",
        "interim": BASE_DIR / "data" / "interim",
        "processed": BASE_DIR / "data" / "processed",
        "results_summary": BASE_DIR / "results_summary",
        "run_config": RUN_DIR / "config",
        "run_outputs": RUN_DIR / "outputs",
        "run_predictions": RUN_DIR / "predictions",
        "run_proc": RUN_DIR / "processed",
        "run_fs": RUN_DIR / "feature_selection",
        "run_ms": RUN_DIR / "model_selection",
        "run_models": RUN_DIR / "models",
    }
    
    for key, path in paths.items():
        if key not in ["base", "run_id"] and isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    
    return paths
