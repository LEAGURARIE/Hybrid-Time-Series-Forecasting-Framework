#!/usr/bin/env python3
"""
CLI for Google Stock ML Pipeline.

Usage:
    python scripts/run_pipeline.py                              # Full pipeline
    python scripts/run_pipeline.py --steps models,ensemble      # Only models + ensemble
    python scripts/run_pipeline.py --steps hpo,models           # HPO + models
    python scripts/run_pipeline.py --models xgb,lstm            # Specific models
    python scripts/run_pipeline.py --steps ensemble             # Only ensemble
    python scripts/run_pipeline.py --ensemble-method stacking   # Specific method
    python scripts/run_pipeline.py --compare-ensembles          # Compare all methods

Steps: load, features, split, fs, hpo, models, ensemble, summary

All settings are in src/config.py
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CONFIG, setup_cli_paths
from src.utils import save_json, load_json, save_pickle, load_pickle


def load_with_fallback(filename: str, run_dir: Path, fallback_dir: Path, use_pandas: bool = False) -> Any:
    """Load file: RUN_ID -> PROJECT level."""
    for path in [run_dir / filename, fallback_dir / filename]:
        if path.exists():
            return pd.read_pickle(path) if use_pandas else load_pickle(path)
    raise FileNotFoundError(f"File not found: {filename}")


def print_header(title: str, char: str = "="):
    print(f"\n{char * 60}\n {title}\n{char * 60}")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_load_data(paths: Dict) -> pd.DataFrame:
    """Step 1: Load raw data."""
    print_header("Step 1/7: Load Data")
    from src.data.loaders import load_all_data
    
    # Use end_date from config, or today if not set
    end_date = CONFIG["data"].get("end_date")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    full_df = load_all_data(
        price_tickers=CONFIG["price_tickers"],
        start=CONFIG["data"]["start_date"],
        end=end_date,
        base_ticker="GOOGL",
        eu_config=CONFIG.get("eu_break_close"),
        load_macro=True,
        paths={"raw": paths["raw"]}
    )
    
    # Note: loaders.py saves prices_raw.pkl
    print(f"[OK] Shape: {full_df.shape}")
    return full_df


def step_features(paths: Dict, full_df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Build features."""
    print_header("Step 2/7: Feature Engineering")
    from src.features.engineering import build_all_features
    
    full_df = build_all_features(full_df, CONFIG)
    
    # Save interim data
    full_df.to_pickle(paths["interim"] / "full_df.pkl")
    print(f"[INFO] Saved full_df.pkl to {paths['interim']}")
    # Note: engineering.py saves full_df.pkl, feature_list, missing_summary, target_meta
    print(f"[OK] Shape: {full_df.shape}")
    return full_df


def step_split(paths: Dict, full_df: pd.DataFrame) -> Dict:
    """Step 3: Split train/valid/test."""
    print_header("Step 3/7: Split Data")
    from src.data.split import split_data
    
    # split_data saves X_train/valid/test_xgb.pkl, y_*.pkl, weights_*.pkl
    splits = split_data(full_df, CONFIG, output_dir=paths["processed"])
    
    return splits


def step_feature_selection(paths: Dict, splits: Dict) -> List[str]:
    """Step 4: Feature selection."""
    print_header("Step 4/7: Feature Selection")
    from src.features.selection import xgb_feature_selection
    
    # xgb_feature_selection saves:
    # - fs_out_local: selected_features_xgb.pkl/csv, feature_importance_*.csv
    # - proc_data_local: X_train/valid/test_xgb_selected.pkl
    selected, _, _ = xgb_feature_selection(
        X_train=splits["X_train"],
        X_valid=splits["X_valid"],
        X_test=splits["X_test"],
        y_train=splits["y_train"],
        y_valid=splits["y_valid"],
        w_train=splits["w_train"],
        w_valid=splits["w_valid"],
        config=CONFIG["xgb_fs"],
        fs_out_local=paths["run_fs"],
        proc_data_local=paths["processed"],
        target_col=CONFIG["data"]["target_col"]
    )
    
    print(f"[OK] Selected {len(selected)} features")
    return selected


def step_hpo(paths: Dict) -> Dict:
    """Step 5: HPO."""
    print_header("Step 5/7: HPO")
    from src.tuning.hpo import run_hpo
    
    proc, run_proc = paths["processed"], paths["run_proc"]
    hpo_cfg = CONFIG["hpo"]
    
    X_train = load_with_fallback("X_train_xgb_selected.pkl", run_proc, proc, use_pandas=True)
    X_valid = load_with_fallback("X_valid_xgb_selected.pkl", run_proc, proc, use_pandas=True)
    y_train = load_with_fallback("y_train.pkl", run_proc, proc)
    y_valid = load_with_fallback("y_valid.pkl", run_proc, proc)
    w_train = load_with_fallback("weights_train.pkl", run_proc, proc)
    w_valid = load_with_fallback("weights_valid.pkl", run_proc, proc)
    
    # Get selected features list
    selected_features = list(X_train.columns)
    
    # run_hpo does its own validation split (ES vs SCORE) internally
    best_params, best_model, results_df = run_hpo(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        w_train=w_train,
        w_valid=w_valid,
        selected_features=selected_features,
        config=hpo_cfg,
        ms_out_local=paths["run_ms"],
        proc_data_local=proc
    )
    
    print(f"[OK] Best params saved")
    return best_params


def step_models(paths: Dict, models_to_run: List[str]):
    """Step 6: Train models."""
    print_header("Step 6/7: Train Models")
    
    proc, run_proc, run_ms = paths["processed"], paths["run_proc"], paths["run_ms"]
    hpo_cfg = CONFIG["hpo"]
    skip = hpo_cfg["lookback"] - 1
    
    X_train = load_with_fallback("X_train_xgb_selected.pkl", run_proc, proc, use_pandas=True)
    X_valid = load_with_fallback("X_valid_xgb_selected.pkl", run_proc, proc, use_pandas=True)
    X_test = load_with_fallback("X_test_xgb_selected.pkl", run_proc, proc, use_pandas=True)
    y_train = load_with_fallback("y_train.pkl", run_proc, proc)
    y_valid = load_with_fallback("y_valid.pkl", run_proc, proc)
    y_test = load_with_fallback("y_test.pkl", run_proc, proc)
    w_train = load_with_fallback("weights_train.pkl", run_proc, proc)
    w_valid = load_with_fallback("weights_valid.pkl", run_proc, proc)
    w_test = load_with_fallback("weights_test.pkl", run_proc, proc)
    
    results = []
    
    if "xgb" in models_to_run:
        print_header("XGBoost", "-")
        from src.models.xgboost_model import train_final_model
        
        try:
            best_params = load_with_fallback("best_params_xgb_reg_t1.pkl", run_ms, proc)
        except FileNotFoundError:
            best_params = CONFIG["xgb_fs"]
        
        # train_final_model does its own validation split internally
        result = train_final_model(
            X_train=X_train,
            X_valid=X_valid,
            X_test=X_test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            w_train=w_train,
            w_valid=w_valid,
            w_test=w_test,
            best_params=best_params,
            config=hpo_cfg,
            models_out_local=paths["run_models"],
            pred_xgb_local=paths["run_dir"] / "predictions" / "xgb"
        )
        
        # Extract metrics from model_results
        test_metrics = [r for r in result["model_results"] if r["split"] == "TEST"]
        if test_metrics:
            metrics = test_metrics[0]
        else:
            metrics = {"wRMSE": 0, "wMAE": 0, "DirAcc": 0}
        
        # Include key params in results for summary comparison
        result_entry = {
            "model": "XGBoost", 
            "run_id": paths["run_id"], 
            "test_wrmse": metrics.get("wRMSE", 0),
            "test_wmae": metrics.get("wMAE", 0),
            "test_diracc": metrics.get("DirAcc", 0),
            # Key XGBoost params
            "n_estimators": best_params.get("n_estimators", 0),
            "max_depth": best_params.get("max_depth", 0),
            "learning_rate": best_params.get("learning_rate", 0),
            "subsample": best_params.get("subsample", 0),
            "colsample_bytree": best_params.get("colsample_bytree", 0),
        }
        results.append(result_entry)
        print(f"[OK] wRMSE: {result_entry['test_wrmse']:.6f}, DirAcc: {result_entry['test_diracc']:.4f}")
    
    if "lgb" in models_to_run:
        print_header("LightGBM", "-")
        from src.models.lightgbm_model import train_final_model_lgb
        
        lgb_hpo_cfg = CONFIG.get("lgb_hpo", CONFIG["hpo"])
        
        try:
            best_params_lgb = load_with_fallback("best_params_lgb_reg_t1.pkl", run_ms, proc)
        except FileNotFoundError:
            best_params_lgb = CONFIG.get("lgb_fs", CONFIG["xgb_fs"])
        
        # train_final_model_lgb does its own validation split internally
        result_lgb = train_final_model_lgb(
            X_train=X_train,
            X_valid=X_valid,
            X_test=X_test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            w_train=w_train,
            w_valid=w_valid,
            w_test=w_test,
            best_params=best_params_lgb,
            config=lgb_hpo_cfg,
            models_out_local=paths["run_models"],
            pred_lgb_local=paths["run_dir"] / "predictions" / "lgb"
        )
        
        if result_lgb:
            # Extract metrics from model_results
            test_metrics = [r for r in result_lgb["model_results"] if r["split"] == "TEST"]
            if test_metrics:
                metrics_lgb = test_metrics[0]
            else:
                metrics_lgb = {"wRMSE": 0, "wMAE": 0, "DirAcc": 0}
            
            # Include key params in results for summary comparison
            result_entry = {
                "model": "LightGBM", 
                "run_id": paths["run_id"], 
                "test_wrmse": metrics_lgb.get("wRMSE", 0),
                "test_wmae": metrics_lgb.get("wMAE", 0),
                "test_diracc": metrics_lgb.get("DirAcc", 0),
                # Key LightGBM params
                "num_leaves": best_params_lgb.get("num_leaves", 0),
                "max_depth": best_params_lgb.get("max_depth", 0),
                "learning_rate": best_params_lgb.get("learning_rate", 0),
                "subsample": best_params_lgb.get("subsample", 0),
                "colsample_bytree": best_params_lgb.get("colsample_bytree", 0),
            }
            results.append(result_entry)
            print(f"[OK] wRMSE: {result_entry['test_wrmse']:.6f}, DirAcc: {result_entry['test_diracc']:.4f}")
        else:
            print("[SKIP] LightGBM not available")
    
    # LSTM and GRU
    nn_models_to_run = [m for m in ["lstm", "gru"] if m in models_to_run]
    if nn_models_to_run:
        print_header("LSTM / GRU Models", "-")
        from src.models.lstm_gru import train_lstm_gru
        from src.data.split import prepare_all_neural_features
        from sklearn.preprocessing import StandardScaler
        
        # Prepare neural features (matching notebook BLOCK 21-22)
        neural_data = prepare_all_neural_features(
            X_train, X_valid.iloc[skip:], X_test.iloc[skip:],
            y_train,
            config=CONFIG,
            output_dir=paths["processed"],
            fs_input_dir=paths["run_fs"],
            fs_output_dir=paths["run_fs"]
        )
        
        # Get arrays
        y_train_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        y_valid_arr = y_valid.iloc[skip:].values if hasattr(y_valid, 'values') else np.asarray(y_valid)[skip:]
        y_test_arr = y_test.iloc[skip:].values if hasattr(y_test, 'values') else np.asarray(y_test)[skip:]
        w_train_arr = np.asarray(w_train)
        w_valid_arr = np.asarray(w_valid)[skip:]
        w_test_arr = np.asarray(w_test)[skip:]
        
        # Scale data for each feature set
        X_train_dict = {}
        X_valid_dict = {}
        X_test_dict = {}
        
        # Collect all feature sets requested by neural models from config
        nn_feature_sets = set()
        for model_type in nn_models_to_run:
            model_cfg = CONFIG.get(model_type, {})
            nn_feature_sets.update(model_cfg.get("feature_sets", []))
        
        print(f"[INFO] Feature sets requested by neural models: {sorted(nn_feature_sets)}")
        
        # Process neural feature sets (neural_40, neural_80) from MI-based selection
        neural_fs_names = [fs for fs in nn_feature_sets if fs.startswith("neural_")]
        for fs_name in neural_fs_names:
            if fs_name not in neural_data:
                print(f"[WARN] Feature set {fs_name} not found in neural_data, skipping...")
                continue
            
            X_tr = neural_data[fs_name]["X_train"]
            X_va = neural_data[fs_name]["X_valid"]
            X_te = neural_data[fs_name]["X_test"]
            
            scaler = StandardScaler()
            X_tr_scaled = pd.DataFrame(
                scaler.fit_transform(X_tr.values),
                index=X_tr.index,
                columns=X_tr.columns
            )
            X_va_scaled = pd.DataFrame(
                scaler.transform(X_va.values),
                index=X_va.index,
                columns=X_va.columns
            )
            X_te_scaled = pd.DataFrame(
                scaler.transform(X_te.values),
                index=X_te.index,
                columns=X_te.columns
            )
            
            X_train_dict[fs_name] = X_tr_scaled
            X_valid_dict[fs_name] = X_va_scaled
            X_test_dict[fs_name] = X_te_scaled
            
            # Save scaler
            save_pickle(scaler, paths["run_models"] / f"scaler_{fs_name}.pkl")
        
        if "xgb_selected" in nn_feature_sets:
            print(f"[INFO] Adding xgb_selected feature set for neural models...")
            
            # Use the already-loaded XGB-selected features (with skip alignment)
            X_tr_xgb = X_train
            X_va_xgb = X_valid.iloc[skip:]
            X_te_xgb = X_test.iloc[skip:]
            
            scaler_xgb = StandardScaler()
            X_tr_xgb_scaled = pd.DataFrame(
                scaler_xgb.fit_transform(X_tr_xgb.values),
                index=X_tr_xgb.index,
                columns=X_tr_xgb.columns
            )
            X_va_xgb_scaled = pd.DataFrame(
                scaler_xgb.transform(X_va_xgb.values),
                index=X_va_xgb.index,
                columns=X_va_xgb.columns
            )
            X_te_xgb_scaled = pd.DataFrame(
                scaler_xgb.transform(X_te_xgb.values),
                index=X_te_xgb.index,
                columns=X_te_xgb.columns
            )
            
            X_train_dict["xgb_selected"] = X_tr_xgb_scaled
            X_valid_dict["xgb_selected"] = X_va_xgb_scaled
            X_test_dict["xgb_selected"] = X_te_xgb_scaled
            
            # Save scaler
            save_pickle(scaler_xgb, paths["run_models"] / "scaler_xgb_selected.pkl")
            print(f"[OK] xgb_selected: TRAIN={X_tr_xgb.shape} | VALID={X_va_xgb.shape} | TEST={X_te_xgb.shape}")
        
        nn_results = train_lstm_gru(
            X_train_dict=X_train_dict,
            X_valid_dict=X_valid_dict,
            X_test_dict=X_test_dict,
            y_train=y_train_arr,
            y_valid=y_valid_arr,
            y_test=y_test_arr,
            w_train=w_train_arr,
            w_valid=w_valid_arr,
            w_test=w_test_arr,
            model_types=nn_models_to_run,
            config=CONFIG,
            hpo_config=hpo_cfg,
            models_out_local=paths["run_models"],
            pred_out_local=paths["run_dir"] / "predictions"
        )
        
        if nn_results:
            for model_name, model_data in nn_results.items():
                if "metrics" in model_data:
                    result_entry = {
                        "model": model_name.upper(),
                        "run_id": paths["run_id"],
                        **model_data["metrics"]
                    }
                    results.append(result_entry)
                    print(f"[OK] {model_name.upper()} wRMSE: {model_data['metrics'].get('test_wrmse', 0):.6f}")
    
    # Hybrid models
    hybrid_models_to_run = [m for m in ["hybrid_seq", "hybrid_par"] if m in models_to_run]
    if hybrid_models_to_run:
        print_header("Hybrid Models", "-")
        from src.models.hybrid import train_hybrid
        from src.data.split import prepare_all_neural_features
        from sklearn.preprocessing import StandardScaler
        
        # Prepare neural features if not already done
        if 'neural_data' not in locals():
            neural_data = prepare_all_neural_features(
                X_train, X_valid.iloc[skip:], X_test.iloc[skip:],
                y_train,
                config=CONFIG,
                output_dir=paths["processed"],
                fs_input_dir=paths["run_fs"],
                fs_output_dir=paths["run_fs"]
            )
            
            y_train_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
            y_valid_arr = y_valid.iloc[skip:].values if hasattr(y_valid, 'values') else np.asarray(y_valid)[skip:]
            y_test_arr = y_test.iloc[skip:].values if hasattr(y_test, 'values') else np.asarray(y_test)[skip:]
            w_train_arr = np.asarray(w_train)
            w_valid_arr = np.asarray(w_valid)[skip:]
            w_test_arr = np.asarray(w_test)[skip:]
        
        # Scale data if not already done
        if 'X_train_dict' not in locals():
            X_train_dict = {}
            X_valid_dict = {}
            X_test_dict = {}
            
            # Collect all feature sets requested by hybrid models from config
            hybrid_feature_sets = set()
            for model_type in hybrid_models_to_run:
                model_cfg = CONFIG.get(model_type, {})
                hybrid_feature_sets.update(model_cfg.get("feature_sets", []))
            
            print(f"[INFO] Feature sets requested by hybrid models: {sorted(hybrid_feature_sets)}")
            
            # Process neural feature sets
            neural_fs_names = [fs for fs in hybrid_feature_sets if fs.startswith("neural_")]
            for fs_name in neural_fs_names:
                if fs_name not in neural_data:
                    print(f"[WARN] Feature set {fs_name} not found in neural_data, skipping...")
                    continue
                
                X_tr = neural_data[fs_name]["X_train"]
                X_va = neural_data[fs_name]["X_valid"]
                X_te = neural_data[fs_name]["X_test"]
                
                scaler = StandardScaler()
                X_tr_scaled = pd.DataFrame(
                    scaler.fit_transform(X_tr.values),
                    index=X_tr.index,
                    columns=X_tr.columns
                )
                X_va_scaled = pd.DataFrame(
                    scaler.transform(X_va.values),
                    index=X_va.index,
                    columns=X_va.columns
                )
                X_te_scaled = pd.DataFrame(
                    scaler.transform(X_te.values),
                    index=X_te.index,
                    columns=X_te.columns
                )
                
                X_train_dict[fs_name] = X_tr_scaled
                X_valid_dict[fs_name] = X_va_scaled
                X_test_dict[fs_name] = X_te_scaled
            
            # Process xgb_selected feature set for hybrid models
            if "xgb_selected" in hybrid_feature_sets:
                print(f"[INFO] Adding xgb_selected feature set for hybrid models...")
                
                X_tr_xgb = X_train
                X_va_xgb = X_valid.iloc[skip:]
                X_te_xgb = X_test.iloc[skip:]
                
                scaler_xgb = StandardScaler()
                X_tr_xgb_scaled = pd.DataFrame(
                    scaler_xgb.fit_transform(X_tr_xgb.values),
                    index=X_tr_xgb.index,
                    columns=X_tr_xgb.columns
                )
                X_va_xgb_scaled = pd.DataFrame(
                    scaler_xgb.transform(X_va_xgb.values),
                    index=X_va_xgb.index,
                    columns=X_va_xgb.columns
                )
                X_te_xgb_scaled = pd.DataFrame(
                    scaler_xgb.transform(X_te_xgb.values),
                    index=X_te_xgb.index,
                    columns=X_te_xgb.columns
                )
                
                X_train_dict["xgb_selected"] = X_tr_xgb_scaled
                X_valid_dict["xgb_selected"] = X_va_xgb_scaled
                X_test_dict["xgb_selected"] = X_te_xgb_scaled
                print(f"[OK] xgb_selected: TRAIN={X_tr_xgb.shape} | VALID={X_va_xgb.shape} | TEST={X_te_xgb.shape}")
        
        hybrid_results = train_hybrid(
            X_train_dict=X_train_dict,
            X_valid_dict=X_valid_dict,
            X_test_dict=X_test_dict,
            y_train=y_train_arr,
            y_valid=y_valid_arr,
            y_test=y_test_arr,
            w_train=w_train_arr,
            w_valid=w_valid_arr,
            w_test=w_test_arr,
            hybrid_types=hybrid_models_to_run,
            config=CONFIG,
            hpo_config=hpo_cfg,
            models_out_local=paths["run_models"],
            pred_out_local=paths["run_dir"] / "predictions"
        )
        
        if hybrid_results:
            for model_name, model_data in hybrid_results.items():
                if "metrics" in model_data:
                    result_entry = {
                        "model": model_name.upper(),
                        "run_id": paths["run_id"],
                        **model_data["metrics"]
                    }
                    results.append(result_entry)
                    print(f"[OK] {model_name.upper()} wRMSE: {model_data['metrics'].get('test_wrmse', 0):.6f}")
    
    for r in results:
        save_json(r, paths["run_outputs"] / f"{r['model'].lower()}_results.json")
    
    return results


def step_ensemble(paths: Dict, config: Dict = None):
    """Step 7: Ensemble - combine model predictions."""
    print_header("Step 7/8: Ensemble")
    from src.models.ensemble import run_ensemble, compare_ensemble_methods
    
    if config is None:
        config = CONFIG.get("ensemble", {})
    
    models_dir = paths["run_models"]
    output_dir = paths["run_outputs"]
    
    # Load weights if available
    valid_weights = None
    test_weights = None
    try:
        valid_weights = load_pickle(paths["processed"] / "weights_valid.pkl")
        test_weights = load_pickle(paths["processed"] / "weights_test.pkl")
    except FileNotFoundError:
        print("[WARN] Sample weights not found, using uniform weights")
    
    # Check if we should compare methods
    if config.get("compare_methods", False):
        print("[INFO] Running ensemble comparison...")
        comparison_df = compare_ensemble_methods(
            models_dir=models_dir,
            output_dir=output_dir / "ensemble_comparison",
            methods=None,  # All methods
            models=config.get("models")
        )
        return {"comparison": comparison_df}
    
    # Run single ensemble method
    result = run_ensemble(
        models_dir=models_dir,
        output_dir=output_dir,
        config=config,
        valid_weights=valid_weights,
        test_weights=test_weights
    )
    
    return result


def step_summary(paths: Dict):
    """Step 8: Summary - accumulate results from ALL runs."""
    print_header("Step 8/8: Summary")
    from datetime import datetime
    
    # Scan all runs
    runs_dir = paths["base"] / "runs"
    if not runs_dir.exists():
        print("[WARN] No runs directory")
        return
    
    run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    if not run_folders:
        print("[WARN] No runs found")
        return
    
    print(f"[INFO] Scanning {len(run_folders)} runs...")
    
    all_results = []
    
    for run_folder in run_folders:
        run_id = run_folder.name
        outputs_dir = run_folder / "outputs"
        config_dir = run_folder / "config"
        models_dir = run_folder / "models"
        
        # Load run config for period info
        run_config = {}
        config_path = config_dir / "run_params.json"
        if config_path.exists():
            run_config = load_json(config_path)
        
        # Extract period info
        data_cfg = run_config.get("data", {})
        train_start = data_cfg.get("limit_start_date", "N/A")
        train_end = data_cfg.get("train_end", "N/A")
        train_period = f"{str(train_start)[:10] if train_start != 'N/A' else 'N/A'} - {train_end}"
        
        valid_start = data_cfg.get("valid_start", "N/A")
        valid_end = data_cfg.get("valid_end", "N/A")
        valid_period = f"{valid_start} - {valid_end}"
        
        test_start = data_cfg.get("test_start", "N/A")
        test_end = data_cfg.get("test_end", "latest")
        test_period = f"{test_start} - {test_end if test_end else 'latest'}"
        
        # Get data range from full_df if available
        data_start, data_end = "N/A", "N/A"
        full_df_path = paths["interim"] / "full_df.pkl"
        if full_df_path.exists():
            try:
                full_df = load_pickle(full_df_path)
                if hasattr(full_df, 'index') and len(full_df.index) > 0:
                    data_start = str(full_df.index.min().date())
                    data_end = str(full_df.index.max().date())
            except:
                pass
        
        # Load results from this run
        if not outputs_dir.exists() and not models_dir.exists():
            continue
        
        # --------------------------
        # Load baselines from final_metrics.csv
        # --------------------------
        if models_dir.exists():
            metrics_path = models_dir / "final_metrics.csv"
            if metrics_path.exists():
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    for baseline_name in ["BASELINE_ZERO", "BASELINE_NAIVE"]:
                        baseline_rows = metrics_df[metrics_df["model"] == baseline_name]
                        for _, row in baseline_rows.iterrows():
                            if row.get("split") == "TEST":
                                all_results.append({
                                    "run_id": run_id,
                                    "model": baseline_name,
                                    "feature_set": "baseline",
                                    "train_period": train_period,
                                    "valid_period": valid_period,
                                    "test_period": test_period,
                                    "data_start": data_start,
                                    "data_end": data_end,
                                    "test_wrmse": float(row["wRMSE"]),
                                    "test_wmae": float(row.get("wMAE", 0)) if "wMAE" in row else None,
                                    "test_diracc": float(row["DirAcc"]),
                                })
                except Exception as e:
                    pass
            
            # --------------------------
            # Load XGBoost from final_metrics.csv
            # --------------------------
            if metrics_path.exists():
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    xgb_test = metrics_df[(metrics_df["model"] == "FINAL_XGB") & (metrics_df["split"] == "TEST")]
                    if len(xgb_test) > 0:
                        row = xgb_test.iloc[0]
                        all_results.append({
                            "run_id": run_id,
                            "model": "XGBoost",
                            "feature_set": "xgb_selected",
                            "train_period": train_period,
                            "valid_period": valid_period,
                            "test_period": test_period,
                            "data_start": data_start,
                            "data_end": data_end,
                            "test_wrmse": float(row["wRMSE"]),
                            "test_wmae": float(row.get("wMAE", 0)) if "wMAE" in row else None,
                            "test_diracc": float(row["DirAcc"]),
                        })
                except:
                    pass
            
            # --------------------------
            # Load LightGBM from final_metrics_lgb.csv
            # --------------------------
            lgb_path = models_dir / "final_metrics_lgb.csv"
            if lgb_path.exists():
                try:
                    lgb_df = pd.read_csv(lgb_path)
                    # Filter for FINAL_LGB model on TEST split
                    lgb_test = lgb_df[(lgb_df["model"] == "FINAL_LGB") & (lgb_df["split"] == "TEST")]
                    if len(lgb_test) > 0:
                        row = lgb_test.iloc[0]
                        all_results.append({
                            "run_id": run_id,
                            "model": "LightGBM",
                            "feature_set": "xgb_selected",
                            "train_period": train_period,
                            "valid_period": valid_period,
                            "test_period": test_period,
                            "data_start": data_start,
                            "data_end": data_end,
                            "test_wrmse": float(row["wRMSE"]),
                            "test_wmae": float(row.get("wMAE", 0)) if "wMAE" in row else None,
                            "test_diracc": float(row["DirAcc"]),
                        })
                except:
                    pass
            
            # --------------------------
            # Load LSTM/GRU from summary files
            # --------------------------
            for model_type in ["lstm", "gru"]:
                summary_path = models_dir / f"{model_type}_summary.csv"
                if summary_path.exists():
                    try:
                        df = pd.read_csv(summary_path)
                        for _, row in df.iterrows():
                            all_results.append({
                                "run_id": run_id,
                                "model": model_type.upper(),
                                "feature_set": row.get("feature_set", "unknown"),
                                "train_period": train_period,
                                "valid_period": valid_period,
                                "test_period": test_period,
                                "data_start": data_start,
                                "data_end": data_end,
                                "test_wrmse": float(row["model_test_wrmse"]),
                                "test_wmae": float(row.get("model_test_wmae", 0)) if "model_test_wmae" in row else None,
                                "test_diracc": float(row["model_test_diracc"]),
                            })
                    except:
                        pass
            
            # --------------------------
            # Load Hybrid from summary files
            # --------------------------
            for hybrid_type in ["hybrid_seq", "hybrid_par"]:
                summary_path = models_dir / f"{hybrid_type}_summary.csv"
                if summary_path.exists():
                    try:
                        df = pd.read_csv(summary_path)
                        model_name = "Hybrid-Seq" if hybrid_type == "hybrid_seq" else "Hybrid-Par"
                        for _, row in df.iterrows():
                            all_results.append({
                                "run_id": run_id,
                                "model": model_name,
                                "feature_set": row.get("feature_set", "unknown"),
                                "train_period": train_period,
                                "valid_period": valid_period,
                                "test_period": test_period,
                                "data_start": data_start,
                                "data_end": data_end,
                                "test_wrmse": float(row["model_test_wrmse"]),
                                "test_wmae": float(row.get("model_test_wmae", 0)) if "model_test_wmae" in row else None,
                                "test_diracc": float(row["model_test_diracc"]),
                            })
                    except:
                        pass
        
        # --------------------------
        # Load Ensemble from outputs
        # --------------------------
        if outputs_dir.exists():
            ensemble_path = outputs_dir / "ensemble_results.json"
            if ensemble_path.exists():
                try:
                    ens_data = load_json(ensemble_path)
                    ens_method = ens_data.get("method", "weighted_average")
                    all_results.append({
                        "run_id": run_id,
                        "model": f"Ensemble-{ens_method}",
                        "feature_set": "all_models",
                        "train_period": train_period,
                        "valid_period": valid_period,
                        "test_period": test_period,
                        "data_start": data_start,
                        "data_end": data_end,
                        "test_wrmse": float(ens_data.get("test_wrmse", 0)),
                        "test_wmae": float(ens_data.get("test_wmae", 0)) if "test_wmae" in ens_data else None,
                        "test_diracc": float(ens_data.get("test_diracc", 0)),
                    })
                except:
                    pass
    
    if not all_results:
        print("[WARN] No results found")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Filter out invalid results (wRMSE=0 is impossible except for perfect predictions)
    # Keep BASELINE_ZERO which has legitimate 0 values for some metrics
    invalid_mask = (
        (results_df["test_wrmse"] == 0) & 
        (results_df["test_diracc"] == 0) &
        (~results_df["model"].str.contains("BASELINE", case=False, na=False))
    )
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"[WARN] Filtering {n_invalid} invalid results (wRMSE=0 and DirAcc=0)")
        results_df = results_df[~invalid_mask]
    
    # Sort and rank
    results_df = results_df.sort_values("test_wrmse").reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    
    # Best per model
    best_per_model = results_df.groupby("model").first().reset_index()
    best_per_model = best_per_model.sort_values("test_wrmse").reset_index(drop=True)
    best_per_model["rank"] = range(1, len(best_per_model) + 1)
    
    # Print summary
    print(f"\n[INFO] Total results: {len(results_df)} from {results_df['run_id'].nunique()} runs")
    print(results_df[["rank", "model", "run_id", "test_wrmse", "test_diracc"]].head(10).to_string(index=False))
    
    # Save to project-level results_summary/
    summary_dir = paths["results_summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. All results
    all_cols = ["rank", "run_id", "model", "feature_set", "train_period", "valid_period", "test_period",
                "data_start", "data_end", "test_wrmse", "test_wmae", "test_diracc"]
    cols_exist = [c for c in all_cols if c in results_df.columns]
    results_df[cols_exist].to_csv(summary_dir / "all_results.csv", index=False)
    print(f"[OK] Saved: all_results.csv")
    
    # 2. Best per model
    best_cols = ["rank", "model", "feature_set", "run_id", "train_period", "valid_period", "test_period",
                 "data_start", "data_end", "test_wrmse", "test_wmae", "test_diracc"]
    best_cols_exist = [c for c in best_cols if c in best_per_model.columns]
    best_per_model[best_cols_exist].to_csv(summary_dir / "best_per_model.csv", index=False)
    print(f"[OK] Saved: best_per_model.csv")
    
    # 3. Full results with params
    results_df.to_csv(summary_dir / "all_results_with_params.csv", index=False)
    print(f"[OK] Saved: all_results_with_params.csv")
    
    # 4. Generate RESULTS.md
    best = results_df.iloc[0]
    n_runs = results_df["run_id"].nunique()
    
    md_lines = [
        "# Results Summary",
        "",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Total Runs:** {n_runs} | **Total Configurations:** {len(results_df)}",
        "",
        "---",
        "",
        "## 🏆 Best Results (Top 10)",
        "",
        "| # | Model | Feature Set | wRMSE | DirAcc | Run |",
        "|---|-------|-------------|-------|--------|-----|",
    ]
    
    for _, row in results_df.head(10).iterrows():
        wrmse = f"{row['test_wrmse']:.6f}" if pd.notna(row.get("test_wrmse")) else "-"
        diracc = f"{row['test_diracc']:.2%}" if pd.notna(row.get("test_diracc")) else "-"
        fs = row.get("feature_set", "-")
        run_short = row['run_id'][-6:] if len(str(row['run_id'])) > 6 else row['run_id']
        md_lines.append(f"| {row['rank']} | {row['model']} | {fs} | {wrmse} | {diracc} | {run_short} |")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## 📊 Best per Model Type",
        "",
        "| # | Model | Feature Set | wRMSE | DirAcc |",
        "|---|-------|-------------|-------|--------|",
    ])
    
    for _, row in best_per_model.iterrows():
        wrmse = f"{row['test_wrmse']:.6f}" if pd.notna(row.get("test_wrmse")) else "-"
        diracc = f"{row['test_diracc']:.2%}" if pd.notna(row.get("test_diracc")) else "-"
        fs = row.get("feature_set", "-")
        md_lines.append(f"| {row['rank']} | {row['model']} | {fs} | {wrmse} | {diracc} |")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## 🥇 Overall Best",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Model | **{best['model']}** |",
        f"| Feature Set | {best.get('feature_set', 'N/A')} |",
        f"| wRMSE | {best.get('test_wrmse', 'N/A'):.6f} |",
        f"| DirAcc | {best.get('test_diracc', 0):.2%} |",
        f"| Run ID | {best['run_id']} |",
        f"| Data Range | {best.get('data_start', 'N/A')} → {best.get('data_end', 'N/A')} |",
        "",
        "---",
        "",
        "## 📖 Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| wRMSE | Weighted Root Mean Squared Error (↓ lower is better) |",
        "| DirAcc | Directional Accuracy (↑ higher is better) |",
        "",
        "*Full details with period configurations available in CSV files.*",
    ])
    
    # =========================================
    # Bootstrap Confidence Intervals
    # =========================================
    print("\n[INFO] Computing Bootstrap Confidence Intervals...")
    
    from src.utils import bootstrap_all_metrics, format_ci
    
    bootstrap_results = []
    N_BOOTSTRAP = 1000
    CONFIDENCE_LEVEL = 0.95
    
    for _, row in best_per_model.iterrows():
        model_name = row["model"]
        run_id = row["run_id"]
        feature_set = row.get("feature_set", "unknown")
        
        run_folder = paths["base"] / "runs" / run_id
        pred_dir = run_folder / "predictions"
        
        # Model-specific prediction files in predictions/{model}/
        # Try new structure first, then fallback to old structure
        pred_paths = {
            "XGBoost": [
                pred_dir / "xgb" / "predictions_test.csv",  # New structure
                pred_dir / "predictions_test.csv",           # Old structure (fallback)
            ],
            "LightGBM": [
                pred_dir / "lgb" / "predictions_test.csv",  # New structure
                pred_dir / "predictions_test.csv",           # Old structure (fallback)
            ],
            "BASELINE_ZERO": [
                pred_dir / "xgb" / "predictions_test.csv",  # New structure
                pred_dir / "predictions_test.csv",           # Old structure (fallback)
            ],
            "BASELINE_NAIVE": [
                pred_dir / "xgb" / "predictions_test.csv",  # New structure
                pred_dir / "predictions_test.csv",           # Old structure (fallback)
            ],
        }
        
        # Neural networks: try different feature_sets (get from config or use all possible)
        all_possible_fs = ["neural_40", "neural_80", "xgb_selected"]
        for model_type in ["LSTM", "GRU"]:
            pred_paths[model_type] = []
            for fs in all_possible_fs:
                pred_paths[model_type].append(pred_dir / f"{model_type.lower()}_{fs}" / "predictions_test.csv")
        
        for model_type, model_key in [("Hybrid-Seq", "hybrid_seq"), ("Hybrid-Par", "hybrid_par")]:
            pred_paths[model_type] = []
            for fs in all_possible_fs:
                pred_paths[model_type].append(pred_dir / f"{model_key}_{fs}" / "predictions_test.csv")
        
        # Find first existing prediction file
        pred_file = None
        if "Ensemble" in model_name or "ensemble" in model_name:
            pred_file = run_folder / "outputs" / "ensemble_predictions_test.csv"
        else:
            # pred_paths contains lists of possible paths
            possible_paths = pred_paths.get(model_name, [])
            for p in possible_paths:
                if p.exists():
                    pred_file = p
                    break
        
        if pred_file and pred_file.exists():
            try:
                pred_df = pd.read_csv(pred_file)
                
                # Handle different column names for y_true
                if "actual" in pred_df.columns:
                    y_true = pred_df["actual"].values
                elif "y_true" in pred_df.columns:
                    y_true = pred_df["y_true"].values
                else:
                    raise ValueError(f"No 'actual' or 'y_true' column in {pred_file}")
                
                # Get predictions based on model type
                if model_name == "BASELINE_ZERO":
                    y_pred = np.zeros(len(y_true))
                elif model_name == "BASELINE_NAIVE":
                    y_pred = np.roll(y_true, 1)
                    y_pred[0] = 0
                elif "predicted" in pred_df.columns:
                    y_pred = pred_df["predicted"].values
                elif "y_pred_model" in pred_df.columns:
                    y_pred = pred_df["y_pred_model"].values
                elif "y_pred" in pred_df.columns:
                    y_pred = pred_df["y_pred"].values
                else:
                    raise ValueError(f"No prediction column in {pred_file}")
                
                # Get weights from CSV or load from pkl
                if "sample_weight" in pred_df.columns:
                    weights = pred_df["sample_weight"].values
                else:
                    weights = None
                    w_path = paths["processed"] / "weights_test.pkl"
                    if w_path.exists():
                        weights = load_pickle(w_path)
                        weights = np.asarray(weights)[:len(y_true)]
                
                if weights is None:
                    weights = np.ones(len(y_true))
                
                # Run bootstrap using bootstrap_all_metrics
                ci_results = bootstrap_all_metrics(
                    y_true, y_pred, weights,
                    n_bootstrap=N_BOOTSTRAP,
                    confidence_level=CONFIDENCE_LEVEL,
                    random_state=42
                )
                
                bootstrap_results.append({
                    "model": model_name,
                    "run_id": run_id,
                    "feature_set": feature_set,
                    "n_samples": len(y_true),
                    "n_bootstrap": N_BOOTSTRAP,
                    "confidence_level": CONFIDENCE_LEVEL,
                    # wRMSE
                    "wrmse": ci_results["wrmse"]["point_estimate"],
                    "wrmse_ci_lower": ci_results["wrmse"]["ci_lower"],
                    "wrmse_ci_upper": ci_results["wrmse"]["ci_upper"],
                    "wrmse_std": ci_results["wrmse"]["std"],
                    # wMAE
                    "wmae": ci_results["wmae"]["point_estimate"],
                    "wmae_ci_lower": ci_results["wmae"]["ci_lower"],
                    "wmae_ci_upper": ci_results["wmae"]["ci_upper"],
                    "wmae_std": ci_results["wmae"]["std"],
                    # DirAcc
                    "diracc": ci_results["diracc"]["point_estimate"],
                    "diracc_ci_lower": ci_results["diracc"]["ci_lower"],
                    "diracc_ci_upper": ci_results["diracc"]["ci_upper"],
                    "diracc_std": ci_results["diracc"]["std"],
                })
                print(f"  [OK] {model_name}: {format_ci(ci_results['wrmse'])}")
                
            except Exception as e:
                print(f"  [WARN] {model_name}: {e}")
        else:
            print(f"  [WARN] {model_name}: No predictions file")
    
    if len(bootstrap_results) > 0:
        bootstrap_df = pd.DataFrame(bootstrap_results)
        bootstrap_df = bootstrap_df.sort_values("wrmse").reset_index(drop=True)
        bootstrap_df.insert(0, "rank", range(1, len(bootstrap_df) + 1))
        bootstrap_df.to_csv(summary_dir / "bootstrap_ci.csv", index=False)
        print(f"[OK] Saved: bootstrap_ci.csv")
        
        # Also save as JSON
        save_json(bootstrap_results, summary_dir / "bootstrap_ci.json")
        print(f"[OK] Saved: bootstrap_ci.json")
        
        # Add to RESULTS.md
        md_lines.extend([
            "",
            "---",
            "",
            "## Bootstrap Confidence Intervals (95%)",
            "",
            "| Model | wRMSE (95% CI) | wMAE (95% CI) | DirAcc (95% CI) | N |",
            "|-------|----------------|---------------|-----------------|---|",
        ])
        
        for _, row in bootstrap_df.iterrows():
            wrmse_ci = f"{row['wrmse']:.6f} [{row['wrmse_ci_lower']:.6f}, {row['wrmse_ci_upper']:.6f}]"
            wmae_ci = f"{row['wmae']:.6f} [{row['wmae_ci_lower']:.6f}, {row['wmae_ci_upper']:.6f}]"
            diracc_ci = f"{row['diracc']:.4f} [{row['diracc_ci_lower']:.4f}, {row['diracc_ci_upper']:.4f}]"
            md_lines.append(f"| {row['model']} | {wrmse_ci} | {wmae_ci} | {diracc_ci} | {row['n_samples']} |")
    
    # =========================================
    # TOMORROW PREDICTIONS SUMMARY
    # =========================================
    print("\n[INFO] Collecting tomorrow predictions...")
    
    tomorrow_results = []
    
    # Use the most recent run for tomorrow predictions
    if len(run_folders) > 0:
        latest_run = run_folders[0]  # Already sorted desc
        latest_run_id = latest_run.name
        pred_dir = latest_run / "predictions"
        outputs_dir = latest_run / "outputs"
        
        # Model paths for tomorrow.csv
        tomorrow_paths = {
            ("XGBoost", "xgb_selected"): pred_dir / "xgb" / "tomorrow.csv",
            ("LightGBM", "xgb_selected"): pred_dir / "lgb" / "tomorrow.csv",
            ("LSTM", "xgb_selected"): pred_dir / "lstm_xgb_selected" / "tomorrow.csv",
            ("LSTM", "neural_40"): pred_dir / "lstm_neural_40" / "tomorrow.csv",
            ("LSTM", "neural_80"): pred_dir / "lstm_neural_80" / "tomorrow.csv",
            ("GRU", "xgb_selected"): pred_dir / "gru_xgb_selected" / "tomorrow.csv",
            ("GRU", "neural_40"): pred_dir / "gru_neural_40" / "tomorrow.csv",
            ("GRU", "neural_80"): pred_dir / "gru_neural_80" / "tomorrow.csv",
            ("Hybrid-Seq", "xgb_selected"): pred_dir / "hybrid_seq_xgb_selected" / "tomorrow.csv",
            ("Hybrid-Seq", "neural_40"): pred_dir / "hybrid_seq_neural_40" / "tomorrow.csv",
            ("Hybrid-Seq", "neural_80"): pred_dir / "hybrid_seq_neural_80" / "tomorrow.csv",
            ("Hybrid-Par", "xgb_selected"): pred_dir / "hybrid_par_xgb_selected" / "tomorrow.csv",
            ("Hybrid-Par", "neural_40"): pred_dir / "hybrid_par_neural_40" / "tomorrow.csv",
            ("Hybrid-Par", "neural_80"): pred_dir / "hybrid_par_neural_80" / "tomorrow.csv",
            ("Ensemble", "all_models"): outputs_dir / "tomorrow.csv",
        }
        
        for (model, feature_set), path in tomorrow_paths.items():
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if len(df) > 0:
                        row = df.iloc[0]
                        tomorrow_results.append({
                            "run_id": latest_run_id,
                            "model": model,
                            "feature_set": row.get("feature_set", feature_set),
                            "last_data_date": row.get("last_data_date", ""),
                            "pred_logret": float(row.get("pred_logret", 0)),
                            "pred_return_pct": float(row.get("pred_return_pct", row.get("pred_logret", 0) * 100)),
                        })
                        print(f"  ✓ {model} ({feature_set}): {row.get('pred_return_pct', 0):.4f}%")
                except Exception as e:
                    pass
    
    if tomorrow_results:
        tomorrow_df = pd.DataFrame(tomorrow_results)
        # Sort by predicted return (highest first)
        tomorrow_df = tomorrow_df.sort_values("pred_return_pct", ascending=False).reset_index(drop=True)
        tomorrow_df.insert(0, "rank", range(1, len(tomorrow_df) + 1))
        
        tomorrow_df.to_csv(summary_dir / "tomorrow_summary.csv", index=False)
        print(f"[OK] Saved: tomorrow_summary.csv ({len(tomorrow_df)} predictions)")
        
        # Add to RESULTS.md
        md_lines.extend([
            "",
            "---",
            "",
            "## Tomorrow Predictions (Next Trading Day)",
            "",
            "| Rank | Model | Feature Set | Pred Return % |",
            "|------|-------|-------------|---------------|",
        ])
        
        for _, row in tomorrow_df.iterrows():
            pred_pct = f"{row['pred_return_pct']:+.4f}%"
            md_lines.append(f"| {row['rank']} | {row['model']} | {row['feature_set']} | {pred_pct} |")
    else:
        print("[WARN] No tomorrow predictions found")
    
    (summary_dir / "RESULTS.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Saved: RESULTS.md")
    
    print(f"\n[INFO] Results saved to: {summary_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Google Stock ML Pipeline")
    parser.add_argument("--steps", default="all", 
                        help="Steps to run: all | load,features,split,fs,hpo,models,ensemble,summary")
    parser.add_argument("--models", default=None, help="xgb,lgb,lstm,gru,hybrid_seq,hybrid_par")
    parser.add_argument("--ensemble-method", default=None, 
                        help="Ensemble method: simple_average,weighted_average,stacking,rank_average")
    parser.add_argument("--compare-ensembles", action="store_true", help="Compare all ensemble methods")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    if args.quiet:
        warnings.filterwarnings("ignore")
    
    paths = setup_cli_paths()
    
    print_header("Google Stock ML Pipeline")
    print(f"[INFO] RUN_ID: {paths['run_id']}")
    print(f"[INFO] Config: src/config.py")
    
    save_json(CONFIG, paths["run_config"] / "run_params.json")
    
    # Parse steps
    all_steps = ["load", "features", "split", "fs", "hpo", "models", "ensemble", "summary"]
    if args.steps == "all":
        steps = all_steps
    else:
        steps = [s.strip() for s in args.steps.split(",")]
    
    # Parse models
    all_models = ["xgb", "lgb", "lstm", "gru", "hybrid_seq", "hybrid_par"]
    models = [m.strip() for m in args.models.split(",")] if args.models else all_models
    
    print(f"[INFO] Steps: {steps}")
    print(f"[INFO] Models: {models}")
    
    # Run pipeline
    full_df = None
    splits = None
    
    if "load" in steps:
        full_df = step_load_data(paths)
    
    if "features" in steps:
        if full_df is None:
            full_df = pd.read_pickle(paths["raw"] / "prices_raw.pkl")
        full_df = step_features(paths, full_df)
    
    if "split" in steps:
        if full_df is None:
            full_df = pd.read_pickle(paths["interim"] / "full_df.pkl")
        splits = step_split(paths, full_df)
    
    if "fs" in steps:
        if splits is None:
            proc = paths["processed"]
            splits = {
                "X_train": pd.read_pickle(proc / "X_train.pkl"),
                "X_valid": pd.read_pickle(proc / "X_valid.pkl"),
                "X_test": pd.read_pickle(proc / "X_test.pkl"),
                "y_train": load_pickle(proc / "y_train.pkl"),
                "y_valid": load_pickle(proc / "y_valid.pkl"),
                "y_test": load_pickle(proc / "y_test.pkl"),
                "w_train": load_pickle(proc / "weights_train.pkl"),
                "w_valid": load_pickle(proc / "weights_valid.pkl"),
                "w_test": load_pickle(proc / "weights_test.pkl"),
            }
        step_feature_selection(paths, splits)
    
    if "hpo" in steps:
        step_hpo(paths)
    
    if "models" in steps:
        step_models(paths, models)
    
    if "ensemble" in steps:
        # Build ensemble config from args
        ensemble_config = CONFIG.get("ensemble", {}).copy()
        if args.ensemble_method:
            ensemble_config["method"] = args.ensemble_method
        if args.compare_ensembles:
            ensemble_config["compare_methods"] = True
        if args.models:
            ensemble_config["models"] = models
        step_ensemble(paths, ensemble_config)
    
    if "summary" in steps:
        step_summary(paths)
    
    print_header("COMPLETE")


if __name__ == "__main__":
    main()
