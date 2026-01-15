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
        "start_date": "2004-09-01",
        "end_date": "2026-01-15",  # Set fixed date for reproducibility; use None for today's date
        "limit_start_date": "2005-12-31",
        "train_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2023-12-31",
        "test_start": "2023-01-01",
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
        "spearman_thresh": 0.90,
        "gain_cum_thresh": 0.90,
        "min_features": 15,
        "neg_sigma": 1.0,
        "pos_sigma": 0.5,
        "min_gain": 0.0,
        "perm_repeats": 20,
        "n_estimators": 4000,
        "early_stopping_rounds": 80,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 10,
        "gamma": 0.5,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1e-4,
        "reg_lambda": 5.0,
        "max_delta_step": 1,
        "random_state": 42,
    },
    
    # --- HPO ---
    "hpo": {
        "n_estimators": 4000,
        "early_stopping_rounds": 80,
        "n_trials_stage1": 160,
        "n_trials_stage2": 80,
        "n_trials_stage2_lowlr": 40,
        "print_every_stage1": 20,
        "print_every_stage2": 20,
        "tie_tol": 1e-5,
        "valid_es_start": "2021-01-01",
        "valid_es_end": "2021-12-31",
        "valid_score_start": "2022-01-01",
        "valid_score_end": "2023-12-31",
        "random_state": 42,
        "lookback": 15,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        # Sampling ranges for HPO
        "sampling": {
            "broad": {
                "max_depth": [2, 7],
                "min_child_weight_log": [0.5, 20.0],
                "subsample": [0.6, 1.0],
                "colsample_bytree": [0.55, 1.0],
                "gamma": [0.0, 3.0],
                "reg_alpha_exp": [-9, -2],
                "reg_lambda_exp": [-2, 1.3],
                "max_delta_step": [0.0, 2.0],
                "lr_high_prob": 0.15,
                "lr_high": [0.06, 0.12],
                "lr_low": [0.003, 0.06],
            },
            "refine": {
                "max_depth_delta": [-1, 2],
                "max_depth_clip": [2, 8],
                "lr_sigma": 0.25,
                "lr_clip": [0.002, 0.15],
                "min_child_weight_sigma": 0.40,
                "min_child_weight_clip": [0.3, 30.0],
                "subsample_sigma": 0.06,
                "subsample_clip": [0.5, 1.0],
                "colsample_sigma": 0.06,
                "colsample_clip": [0.5, 1.0],
                "gamma_sigma": 0.30,
                "gamma_clip": [0.0, 5.0],
                "reg_alpha_sigma": 0.7,
                "reg_alpha_exp_clip": [-10, 0],
                "reg_lambda_sigma": 0.5,
                "reg_lambda_exp_clip": [-3, 2],
                "max_delta_step_sigma": 0.20,
                "max_delta_step_clip": [0.0, 4.0],
            },
            "refine_low_lr": {
                "lr_shift": -0.8,
                "lr_clip": [0.0015, 0.06],
            },
        },
    },
    
    # --- LightGBM Feature Selection ---
    "lgb_fs": {
        "spearman_thresh": 0.90,
        "gain_cum_thresh": 0.90,
        "min_features": 15,
        "neg_sigma": 1.0,
        "pos_sigma": 0.5,
        "min_gain": 0.0,
        "perm_repeats": 20,
        "n_estimators": 4000,
        "early_stopping_rounds": 80,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1e-4,
        "reg_lambda": 5.0,
        "random_state": 42,
    },
    
    # --- LightGBM HPO ---
    "lgb_hpo": {
        "n_estimators": 4000,
        "early_stopping_rounds": 80,
        "n_trials_stage1": 160,
        "n_trials_stage2": 80,
        "n_trials_stage2_lowlr": 40,
        "valid_es_start": "2021-01-01",
        "valid_es_end": "2021-12-31",
        "valid_score_start": "2022-01-01",
        "valid_score_end": "2023-12-31",
        "lookback": 15,
        "random_state": 42,
    },
    
    # --- LSTM ---
    "lstm": {
        "lookback": 15,
        "stride": 1,
        "units_1": 32,
        "units_2": 16,
        "dense_units": 16,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "clipnorm": 1.0,
        "epochs": 80,
        "batch_size": 16,
        "patience": 10,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80", "xgb_selected"],
    },
    
    # --- GRU ---
    "gru": {
        "lookback": 15,
        "stride": 1,
        "units_1": 32,
        "units_2": 16,
        "dense_units": 16,
        "dropout": 0.20,
        "learning_rate": 5e-4,
        "clipnorm": 1.0,
        "epochs": 80,
        "batch_size": 16,
        "patience": 10,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80", "xgb_selected"],
    },
    
    # --- Hybrid Sequential ---
    "hybrid_seq": {
        "lookback": 15,
        "stride": 1,
        "lstm_units": 32,
        "gru_units": 16,
        "dense_units": 16,
        "dropout": 0.20,
        "learning_rate": 4e-4,
        "clipnorm": 1.0,
        "epochs": 90,
        "batch_size": 16,
        "patience": 12,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80", "xgb_selected"],
    },
    
    # --- Hybrid Parallel ---
    "hybrid_par": {
        "lookback": 15,
        "stride": 1,
        "lstm_units": 24,
        "gru_units": 24,
        "dense_units": 16,
        "dropout": 0.20,
        "learning_rate": 4e-4,
        "clipnorm": 1.0,
        "epochs": 90,
        "batch_size": 16,
        "patience": 12,
        "loss": "mse",
        "dense_activation": "relu",
        "output_activation": "linear",
        "random_state": 42,
        "feature_sets": ["neural_40", "neural_80", "xgb_selected"],
    },
    
    # --- Ensemble ---
    "ensemble": {
        "method": "weighted_average",  # simple_average, weighted_average, stacking, rank_average
        "models": ["xgb", "lgb", "lstm", "gru", "hybrid_seq", "hybrid_par"],  # or null for all
        "weights": "auto",  # "auto" for inverse_wrmse, or dict of manual weights
        "weight_method": "inverse_wrmse",  # inverse_wrmse, inverse_wrmse_squared
        "meta_model": "ridge",  # for stacking: ridge, lasso, elasticnet, xgb, lgb
        "meta_params": {
            "alpha": 1.0,  # for ridge/lasso/elasticnet
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
