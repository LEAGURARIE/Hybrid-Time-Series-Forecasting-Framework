"""
XGBoost final model training module for Google Stock ML project.
Matches notebook Cell 61 (BLOCK 25) exactly.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional, Any
from pathlib import Path

from ..utils import (
    w_rmse, w_mae, directional_accuracy as dir_acc,
    save_pickle, save_json, copy_file, ensure_dir, load_pickle,
    apply_shap_feature_selection, save_shap_top_features
)


# ==============================================================================
# CELL 61 (BLOCK 25): FINAL MODEL + BASELINES + SHAP
# ==============================================================================

def split_valid_es_score(Xv, yv, wv, valid_es_start, valid_es_end, valid_score_start, valid_score_end):
    """
    Split validation into ES (early stopping) and SCORE (model selection) sets.
    Matches notebook split_valid_es_score() exactly.
    """
    if not isinstance(Xv.index, pd.DatetimeIndex):
        return (Xv, yv, wv), (Xv, yv, wv), "FULL_VALID"

    wv_s = pd.Series(wv, index=Xv.index)
    
    es_start, es_end = pd.Timestamp(valid_es_start), pd.Timestamp(valid_es_end)
    sc_start, sc_end = pd.Timestamp(valid_score_start), pd.Timestamp(valid_score_end)
    
    m_es = (Xv.index >= es_start) & (Xv.index <= es_end)
    m_sc = (Xv.index >= sc_start) & (Xv.index <= sc_end)
    mode_str = f"VALID_ES={valid_es_start}:{valid_es_end} / VALID_SCORE={valid_score_start}:{valid_score_end}"

    if m_es.sum() > 0 and m_sc.sum() > 0:
        return (Xv.loc[m_es], yv.loc[m_es], wv_s.loc[m_es].to_numpy(float)), \
               (Xv.loc[m_sc], yv.loc[m_sc], wv_s.loc[m_sc].to_numpy(float)), \
               mode_str

    return (Xv, yv, wv), (Xv, yv, wv), "FULL_VALID"


def train_final_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    w_test: np.ndarray,
    best_params: Dict,
    config: Dict,
    models_out_local: Optional[Path] = None,
    models_out_drive: Optional[Path] = None,
    outputs_local: Optional[Path] = None,
    outputs_drive: Optional[Path] = None,
    pred_xgb_local: Optional[Path] = None,
    pred_xgb_drive: Optional[Path] = None,
    proc_data_local: Optional[Path] = None,
    proc_data_drive: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Train final XGBoost model and evaluate.
    Matches notebook Cell 61 (BLOCK 25) exactly.
    
    Includes:
    - BASELINE_ZERO and BASELINE_NAIVE
    - Final model training with early stopping on VALID_ES
    - Evaluation on VALID_SCORE and TEST
    - Tomorrow prediction
    - SHAP analysis (optional)
    
    Saves to models_out_local/models_out_drive:
        - final_model_xgb.json
        - final_model_xgb.pkl
        - final_metrics.csv
    
    Saves to outputs_local/outputs_drive:
        - baseline_zero_results.json
        - baseline_naive_results.json
    
    Saves to pred_xgb_local/pred_xgb_drive:
        - predictions_valid.csv
        - predictions_test.csv
        - tomorrow.csv
        - backtest.csv
        - plot.png
        - shap_values_test.csv
        - shap_feature_importance.csv
        - shap_summary_bar.png
        - shap_summary_beeswarm.png
    
    Returns:
        Dict with model, predictions, metrics, baselines
    """
    HPO_CFG = config
    
    # -------------------------
    # Apply SHAP feature selection if configured
    # -------------------------
    SHAP_CFG = config.get("shap", {})
    X_train, X_valid, X_test, feature_source = apply_shap_feature_selection(
        X_train, X_valid, X_test, 
        SHAP_CFG, 
        proc_data_local,
        proc_data_drive
    )
    print(f"[INFO] Feature source: {feature_source} ({X_train.shape[1]} features)")
    
    # -------------------------
    # Prepare data
    # -------------------------
    y_train_f = y_train.astype(float).copy()
    y_valid_f = y_valid.astype(float).copy()
    y_test_f = y_test.astype(float).copy()
    
    w_train_arr = np.asarray(w_train, dtype=float)
    w_valid_arr = np.asarray(w_valid, dtype=float)
    w_test_arr = np.asarray(w_test, dtype=float)
    
    # -------------------------
    # Align VALID + TEST data with Neural Networks (skip first lookback-1 rows)
    # -------------------------
    LOOKBACK = int(HPO_CFG["lookback"])
    SKIP_ROWS = LOOKBACK - 1
    
    # Align VALID
    X_valid_sel = X_valid.iloc[SKIP_ROWS:]
    y_valid_f = y_valid_f.iloc[SKIP_ROWS:]
    w_valid_arr = w_valid_arr[SKIP_ROWS:]
    
    # Align TEST
    X_test_sel = X_test.iloc[SKIP_ROWS:]
    y_test_f = y_test_f.iloc[SKIP_ROWS:]
    w_test_arr = w_test_arr[SKIP_ROWS:]
    
    print(f"[INFO] Aligned VALID+TEST with lookback={LOOKBACK}: skipped first {SKIP_ROWS} rows")
    print(f"[INFO] VALID shapes after alignment: X={X_valid_sel.shape}")
    print(f"[INFO] TEST shapes after alignment: X={X_test_sel.shape}")
    
    # -------------------------
    # Split VALID into ES and SCORE (date-based)
    # -------------------------
    VALID_ES_START = HPO_CFG["valid_es_start"]
    VALID_ES_END = HPO_CFG["valid_es_end"]
    VALID_SCORE_START = HPO_CFG["valid_score_start"]
    VALID_SCORE_END = HPO_CFG["valid_score_end"]
    
    (X_es, y_es, w_es), (X_sc, y_sc, w_sc), valid_mode = split_valid_es_score(
        X_valid_sel, y_valid_f, w_valid_arr,
        VALID_ES_START, VALID_ES_END, VALID_SCORE_START, VALID_SCORE_END
    )
    
    print(f"[INFO] VALID mode: {valid_mode}")
    print(f"[INFO] VALID_ES: {X_es.shape} | VALID_SCORE: {X_sc.shape}")
    
    # -------------------------
    # BASELINES (Zero + Naive)
    # -------------------------
    print("\n[INFO] Computing BASELINES...")
    baseline_results = []
    
    # ----- BASELINE ZERO (predict 0) -----
    print("  [1] BASELINE_ZERO (predict 0):")
    
    # Zero baseline on VALID_SCORE
    pred_zero_sc = np.zeros(len(y_sc), dtype=float)
    baseline_results.append({
        "model": "BASELINE_ZERO",
        "split": "VALID_SCORE",
        "n": int(len(y_sc)),
        "wRMSE": w_rmse(y_sc, pred_zero_sc, w_sc),
        "wMAE": w_mae(y_sc, pred_zero_sc, w_sc),
        "DirAcc": dir_acc(y_sc, pred_zero_sc),
    })
    
    # Zero baseline on TEST
    pred_zero_test = np.zeros(len(y_test_f), dtype=float)
    baseline_results.append({
        "model": "BASELINE_ZERO",
        "split": "TEST",
        "n": int(len(y_test_f)),
        "wRMSE": w_rmse(y_test_f, pred_zero_test, w_test_arr),
        "wMAE": w_mae(y_test_f, pred_zero_test, w_test_arr),
        "DirAcc": dir_acc(y_test_f, pred_zero_test),
    })
    
    for r in baseline_results[-2:]:
        print(f"      {r['split']}: wRMSE={r['wRMSE']:.6f} | DirAcc={r['DirAcc']:.4f}")
    
    # ----- BASELINE NAIVE (last-value forecast) -----
    print("  [2] BASELINE_NAIVE (last-value forecast):")
    
    # Simple shift within splits (matches notebook)
    pred_naive_sc = np.roll(np.asarray(y_sc), 1)
    pred_naive_sc[0] = 0
    
    pred_naive_test = np.roll(np.asarray(y_test_f), 1)
    pred_naive_test[0] = 0
    
    # Naive baseline on VALID_SCORE
    baseline_results.append({
        "model": "BASELINE_NAIVE",
        "split": "VALID_SCORE",
        "n": int(len(y_sc)),
        "wRMSE": w_rmse(y_sc, pred_naive_sc, w_sc),
        "wMAE": w_mae(y_sc, pred_naive_sc, w_sc),
        "DirAcc": dir_acc(y_sc, pred_naive_sc),
    })
    
    # Naive baseline on TEST
    baseline_results.append({
        "model": "BASELINE_NAIVE",
        "split": "TEST",
        "n": int(len(y_test_f)),
        "wRMSE": w_rmse(y_test_f, pred_naive_test, w_test_arr),
        "wMAE": w_mae(y_test_f, pred_naive_test, w_test_arr),
        "DirAcc": dir_acc(y_test_f, pred_naive_test),
    })
    
    for r in baseline_results[-2:]:
        print(f"      {r['split']}: wRMSE={r['wRMSE']:.6f} | DirAcc={r['DirAcc']:.4f}")
    
    print("\n[INFO] BASELINE Summary:")
    for r in baseline_results:
        print(f"  - {r['model']:15} | {r['split']:12} | wRMSE={r['wRMSE']:.6f} | DirAcc={r['DirAcc']:.4f}")
    
    # -------------------------
    # Train FINAL MODEL (early stop on VALID_ES)
    # -------------------------
    print("\n[INFO] Training FINAL MODEL...")
    
    N_ESTIMATORS = int(HPO_CFG["n_estimators"])
    EARLY_STOP = int(HPO_CFG["early_stopping_rounds"])
    RANDOM_SEED = int(HPO_CFG["random_state"])
    
    # Ensure correct types in best_params
    best_params = dict(best_params)
    
    # Remove params that are set explicitly to avoid duplicates
    for key in ["n_estimators", "objective", "eval_metric", "tree_method", "random_state", "n_jobs", "verbosity", "early_stopping_rounds"]:
        best_params.pop(key, None)
    
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])
    for k in ["learning_rate", "min_child_weight", "subsample", "colsample_bytree",
              "gamma", "reg_alpha", "reg_lambda", "max_delta_step"]:
        if k in best_params:
            best_params[k] = float(best_params[k])
    
    # XGBoost settings from config
    OBJECTIVE = HPO_CFG["objective"]
    EVAL_METRIC = HPO_CFG["eval_metric"]
    TREE_METHOD = HPO_CFG["tree_method"]
    
    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        objective=OBJECTIVE,
        eval_metric=EVAL_METRIC,
        tree_method=TREE_METHOD,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOP,
        **best_params
    )
    
    model.fit(
        X_train, y_train_f,
        sample_weight=w_train_arr,
        eval_set=[(X_es, y_es)],
        sample_weight_eval_set=[w_es],
        verbose=False
    )
    
    best_iter = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    print(f"[INFO] Training complete. best_iteration={best_iter} | best_score(ES)={best_score}")
    
    # -------------------------
    # Evaluate FINAL MODEL on VALID_SCORE + TEST
    # -------------------------
    print("\n[INFO] Evaluating FINAL MODEL...")
    
    model_results = []
    
    # Model on VALID_SCORE
    pred_model_sc = model.predict(X_sc)
    model_results.append({
        "model": "FINAL_XGB",
        "split": "VALID_SCORE",
        "n": int(len(y_sc)),
        "wRMSE": w_rmse(y_sc, pred_model_sc, w_sc),
        "wMAE": w_mae(y_sc, pred_model_sc, w_sc),
        "DirAcc": dir_acc(y_sc, pred_model_sc),
    })
    
    # Model on TEST
    pred_model_test = model.predict(X_test_sel)
    model_results.append({
        "model": "FINAL_XGB",
        "split": "TEST",
        "n": int(len(y_test_f)),
        "wRMSE": w_rmse(y_test_f, pred_model_test, w_test_arr),
        "wMAE": w_mae(y_test_f, pred_model_test, w_test_arr),
        "DirAcc": dir_acc(y_test_f, pred_model_test),
    })
    
    print("[INFO] FINAL MODEL results:")
    for r in model_results:
        print(f"  - {r['split']}: wRMSE={r['wRMSE']:.6f} | DirAcc={r['DirAcc']:.4f}")
    
    # -------------------------
    # Comparison: BASELINE vs MODEL
    # -------------------------
    all_results = baseline_results + model_results
    metrics_df = pd.DataFrame(all_results)
    
    # Add improvement column (compared to BASELINE_ZERO)
    baseline_zero_wrmse = {r["split"]: r["wRMSE"] for r in baseline_results if r["model"] == "BASELINE_ZERO"}
    metrics_df["wRMSE_vs_zero"] = metrics_df.apply(
        lambda row: baseline_zero_wrmse.get(row["split"], 0) - row["wRMSE"], axis=1
    )
    
    # Add improvement vs BASELINE_NAIVE
    baseline_naive_wrmse = {r["split"]: r["wRMSE"] for r in baseline_results if r["model"] == "BASELINE_NAIVE"}
    metrics_df["wRMSE_vs_naive"] = metrics_df.apply(
        lambda row: baseline_naive_wrmse.get(row["split"], 0) - row["wRMSE"], axis=1
    )
    
    print("\n[INFO] BASELINE vs FINAL MODEL comparison:")
    print(metrics_df.to_string())
    
    # -------------------------
    # Build predictions DataFrames
    # -------------------------
    preds_valid_score_df = pd.DataFrame({
        "date": X_sc.index,
        "actual": y_sc.values,
        "baseline_zero": pred_zero_sc,
        "baseline_naive": pred_naive_sc,
        "predicted": pred_model_sc,
        "sample_weight": w_sc,
    }).reset_index(drop=True)
    
    preds_test_df = pd.DataFrame({
        "date": X_test_sel.index,
        "actual": y_test_f.values,
        "baseline_zero": pred_zero_test,
        "baseline_naive": pred_naive_test,
        "predicted": pred_model_test,
        "sample_weight": w_test_arr,
    }).reset_index(drop=True)
    
    # -------------------------
    # Save artifacts (LOCAL + DRIVE)
    # -------------------------
    # Model
    if models_out_local:
        models_out_local = ensure_dir(Path(models_out_local))
        
        model_path_local = models_out_local / "final_model_xgb.json"
        model.get_booster().save_model(str(model_path_local))
        
        model_pkl_local = models_out_local / "final_model_xgb.pkl"
        save_pickle(model, model_pkl_local)
        
        metrics_path_local = models_out_local / "final_metrics.csv"
        metrics_df.to_csv(metrics_path_local, index=False)
        
        if models_out_drive:
            models_out_drive = ensure_dir(Path(models_out_drive))
            copy_file(model_path_local, models_out_drive / model_path_local.name)
            copy_file(model_pkl_local, models_out_drive / model_pkl_local.name)
            copy_file(metrics_path_local, models_out_drive / metrics_path_local.name)
        
        print("\n[OK] Saved FINAL MODEL artifacts:")
        print("  -", model_path_local.name)
        print("  -", model_pkl_local.name)
        print("  -", metrics_path_local.name)
    
    # Baseline results JSON (for CLI summary)
    if outputs_local:
        outputs_local = ensure_dir(Path(outputs_local))
        
        for baseline_row in baseline_results:
            if baseline_row["split"] == "TEST":
                baseline_json = {
                    "model": baseline_row["model"],
                    "test_wrmse": baseline_row["wRMSE"],
                    "test_wmae": baseline_row.get("wMAE"),
                    "test_diracc": baseline_row["DirAcc"],
                }
                baseline_name = baseline_row["model"].lower()
                save_json(baseline_json, outputs_local / f"{baseline_name}_results.json")
                
                if outputs_drive:
                    outputs_drive = ensure_dir(Path(outputs_drive))
                    save_json(baseline_json, outputs_drive / f"{baseline_name}_results.json")
    
    # Predictions
    if pred_xgb_local:
        pred_xgb_local = ensure_dir(Path(pred_xgb_local))
        
        preds_valid_path_local = pred_xgb_local / "predictions_valid.csv"
        preds_valid_score_df.to_csv(preds_valid_path_local, index=False)
        
        preds_test_path_local = pred_xgb_local / "predictions_test.csv"
        preds_test_df.to_csv(preds_test_path_local, index=False)
        
        if pred_xgb_drive:
            pred_xgb_drive = ensure_dir(Path(pred_xgb_drive))
            copy_file(preds_valid_path_local, pred_xgb_drive / preds_valid_path_local.name)
            copy_file(preds_test_path_local, pred_xgb_drive / preds_test_path_local.name)
        
        print("  - predictions/xgb/", preds_valid_path_local.name)
        print("  - predictions/xgb/", preds_test_path_local.name)
    
    # -------------------------
    # Tomorrow Prediction + Plot
    # -------------------------
    PLOT_CFG = config.get("plot", {})
    N_PLOT = int(PLOT_CFG.get("n_plot", 60))
    FIGSIZE = tuple(PLOT_CFG.get("figsize", [12, 6]))
    DPI = int(PLOT_CFG.get("dpi", 100))
    
    # Predict tomorrow
    last_date = X_test_sel.index[-1]
    X_last = X_test_sel.iloc[[-1]]
    pred_tomorrow = float(model.predict(X_last)[0])
    pred_tomorrow_date = last_date + pd.Timedelta(days=1)
    
    # Save tomorrow prediction
    pred_tomorrow_df = pd.DataFrame([{
        "feature_set": "xgb_selected",
        "last_data_date": last_date,
        "predicted_for": "next_trading_day",
        "pred_logret": pred_tomorrow,
        "pred_return_pct": float(np.expm1(pred_tomorrow) * 100),
    }])
    
    # Create backtest dataframe
    hist_df = pd.DataFrame({
        "date": X_test_sel.index,
        "actual": y_test_f.values,
        "y_pred": pred_model_test,
    }).set_index("date")
    
    hist_tail = hist_df.tail(N_PLOT).copy()
    
    if pred_xgb_local:
        pred_tomorrow_df.to_csv(pred_xgb_local / "tomorrow.csv", index=False)
        hist_tail.to_csv(pred_xgb_local / "backtest.csv")
        
        if pred_xgb_drive:
            copy_file(pred_xgb_local / "tomorrow.csv", pred_xgb_drive / "tomorrow.csv")
            copy_file(pred_xgb_local / "backtest.csv", pred_xgb_drive / "backtest.csv")
        
        # Plot
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(hist_tail.index, hist_tail["actual"].values, linewidth=1, label="Actual")
            ax.plot(hist_tail.index, hist_tail["y_pred"].values, linewidth=1, label="Predicted (y_pred)")
            ax.scatter([pred_tomorrow_date], [pred_tomorrow], s=90, marker="X", color="red", label=f"Tomorrow: {pred_tomorrow:.4f}")
            ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"XGBoost Predictions — xgb_selected — last {len(hist_tail)} days + tomorrow")
            ax.set_xlabel("Date")
            ax.set_ylabel("Log Return")
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(pred_xgb_local / "plot.png", dpi=DPI)
            
            if pred_xgb_drive:
                copy_file(pred_xgb_local / "plot.png", pred_xgb_drive / "plot.png")
            
            plt.close(fig)
        except ImportError:
            print("[WARN] matplotlib not installed, skipping plot")
    
    print(f"[INFO] Tomorrow prediction: {pred_tomorrow:.6f} ({np.expm1(pred_tomorrow)*100:.4f}%)")
    print(f"[OK] Saved: predictions/xgb/ (tomorrow.csv, backtest.csv, plot.png)")
    
    # -------------------------
    # SHAP Analysis (config-based)
    # -------------------------
    SHAP_CFG = config.get("shap", {})
    SHAP_ENABLED = bool(SHAP_CFG.get("enabled", True))
    
    shap_values = None
    shap_importance = None
    
    if SHAP_ENABLED and pred_xgb_local:
        print("\n[INFO] Computing SHAP values...")
        
        try:
            import shap
            import matplotlib.pyplot as plt
            
            # Config
            SHAP_MAX_DISPLAY = int(SHAP_CFG.get("max_display", 20))
            SHAP_FIGSIZE = tuple(SHAP_CFG.get("figsize", [10, 8]))
            SHAP_BAR = bool(SHAP_CFG.get("plot_type_bar", True))
            SHAP_BEESWARM = bool(SHAP_CFG.get("plot_type_beeswarm", True))
            SHAP_SAVE_VALUES = bool(SHAP_CFG.get("save_values", True))
            
            # Create explainer (TreeExplainer is fast for XGBoost)
            explainer = shap.TreeExplainer(model)
            
            # Compute SHAP values on TEST set
            shap_values = explainer.shap_values(X_test_sel)
            
            # Save SHAP values as DataFrame
            if SHAP_SAVE_VALUES:
                shap_df = pd.DataFrame(shap_values, columns=X_test_sel.columns, index=X_test_sel.index)
                shap_df.to_csv(pred_xgb_local / "shap_values_test.csv")
                if pred_xgb_drive:
                    copy_file(pred_xgb_local / "shap_values_test.csv", pred_xgb_drive / "shap_values_test.csv")
            
            # Feature importance (mean |SHAP|)
            shap_importance = pd.DataFrame({
                "feature": X_test_sel.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0)
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_importance.to_csv(pred_xgb_local / "shap_feature_importance.csv", index=False)
            if pred_xgb_drive:
                copy_file(pred_xgb_local / "shap_feature_importance.csv", pred_xgb_drive / "shap_feature_importance.csv")
            
            print(f"[INFO] Top 10 features by SHAP importance:")
            print(shap_importance.head(10).to_string(index=False))
            
            # Save top N features for other models to use
            SHAP_TOP_N = SHAP_CFG.get("top_n_features", None)
            if SHAP_TOP_N:
                save_shap_top_features(
                    shap_importance, SHAP_TOP_N, 
                    proc_data_local, proc_data_drive
                )
            
            # SHAP Summary Plot (bar)
            if SHAP_BAR:
                fig_bar, ax_bar = plt.subplots(figsize=SHAP_FIGSIZE)
                shap.summary_plot(shap_values, X_test_sel, plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY)
                plt.tight_layout()
                plt.savefig(pred_xgb_local / "shap_summary_bar.png", dpi=DPI, bbox_inches="tight")
                if pred_xgb_drive:
                    copy_file(pred_xgb_local / "shap_summary_bar.png", pred_xgb_drive / "shap_summary_bar.png")
                plt.close()
            
            # SHAP Summary Plot (beeswarm)
            if SHAP_BEESWARM:
                fig_bee, ax_bee = plt.subplots(figsize=SHAP_FIGSIZE)
                shap.summary_plot(shap_values, X_test_sel, show=False, max_display=SHAP_MAX_DISPLAY)
                plt.tight_layout()
                plt.savefig(pred_xgb_local / "shap_summary_beeswarm.png", dpi=DPI, bbox_inches="tight")
                if pred_xgb_drive:
                    copy_file(pred_xgb_local / "shap_summary_beeswarm.png", pred_xgb_drive / "shap_summary_beeswarm.png")
                plt.close()
            
            print("[OK] Saved SHAP analysis:")
            if SHAP_SAVE_VALUES:
                print("  - shap_values_test.csv")
            print("  - shap_feature_importance.csv")
            if SHAP_BAR:
                print("  - shap_summary_bar.png")
            if SHAP_BEESWARM:
                print("  - shap_summary_beeswarm.png")
            
        except ImportError:
            print("[WARN] shap not installed. Run: pip install shap")
        except Exception as e:
            print(f"[WARN] SHAP analysis failed: {e}")
    else:
        print("[INFO] SHAP analysis disabled in config.")
    
    print("[OK] BLOCK 25 complete.")
    
    return {
        "model": model,
        "best_params": best_params,
        "metrics_df": metrics_df,
        "baseline_results": baseline_results,
        "model_results": model_results,
        "pred_valid_score": pred_model_sc,
        "pred_test": pred_model_test,
        "pred_tomorrow": pred_tomorrow,
        "preds_valid_df": preds_valid_score_df,
        "preds_test_df": preds_test_df,
        "backtest_df": hist_tail,
        "shap_values": shap_values,
        "shap_importance": shap_importance,
    }
