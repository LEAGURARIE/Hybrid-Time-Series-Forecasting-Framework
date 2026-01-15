"""
Hyperparameter optimization module for Google Stock ML project.
Matches notebook Cell 58 (BLOCK 24) exactly.
"""

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional, Tuple
from pathlib import Path

from ..utils import w_rmse, w_mae, directional_accuracy as dir_acc, save_pickle, copy_file, ensure_dir, _to_np


# ==============================================================================
# CELL 58 (BLOCK 24): HPO - Manual Sampling
# ==============================================================================

def _clip(v, lo, hi) -> float:
    """Clip value to range."""
    return float(min(max(float(v), lo), hi))


def _log_uniform(rng, lo_exp, hi_exp) -> float:
    """Log-uniform sampling."""
    return float(10 ** rng.uniform(lo_exp, hi_exp))


def _log_jitter(rng, v, sigma=0.6, lo_exp=-12, hi_exp=2) -> float:
    """Log-space jitter."""
    v = float(max(v, 1e-12))
    logv = np.log10(v) + rng.normal(0.0, sigma)
    logv = float(np.clip(logv, lo_exp, hi_exp))
    return float(10 ** logv)


def sample_broad(rng, broad_cfg):
    """
    Broad sampling biased toward lower learning_rate for stability.
    Matches notebook sample_broad() exactly.
    """
    lr_high_prob = broad_cfg["lr_high_prob"]
    if rng.random() < (1 - lr_high_prob):
        lr_lo, lr_hi = broad_cfg["lr_low"]
        lr = float(np.exp(rng.uniform(np.log(lr_lo), np.log(lr_hi))))
    else:
        lr_lo, lr_hi = broad_cfg["lr_high"]
        lr = float(rng.uniform(lr_lo, lr_hi))

    md_lo, md_hi = broad_cfg["max_depth"]
    mcw_lo, mcw_hi = broad_cfg["min_child_weight_log"]
    ss_lo, ss_hi = broad_cfg["subsample"]
    cs_lo, cs_hi = broad_cfg["colsample_bytree"]
    gm_lo, gm_hi = broad_cfg["gamma"]
    ra_lo, ra_hi = broad_cfg["reg_alpha_exp"]
    rl_lo, rl_hi = broad_cfg["reg_lambda_exp"]
    mds_lo, mds_hi = broad_cfg["max_delta_step"]

    return {
        "max_depth": int(rng.integers(md_lo, md_hi)),  # exclusive upper bound
        "learning_rate": lr,
        "min_child_weight": float(np.exp(rng.uniform(np.log(mcw_lo), np.log(mcw_hi)))),
        "subsample": float(rng.uniform(ss_lo, ss_hi)),
        "colsample_bytree": float(rng.uniform(cs_lo, cs_hi)),
        "gamma": float(rng.uniform(gm_lo, gm_hi)),
        "reg_alpha": _log_uniform(rng, ra_lo, ra_hi),
        "reg_lambda": _log_uniform(rng, rl_lo, rl_hi),
        "max_delta_step": float(rng.uniform(mds_lo, mds_hi)),
    }


def sample_refine(rng, best, refine_cfg):
    """
    Refine around best parameters.
    Matches notebook sample_refine() exactly.
    """
    md_delta = refine_cfg["max_depth_delta"]
    md_clip = refine_cfg["max_depth_clip"]
    lr_sigma = refine_cfg["lr_sigma"]
    lr_clip = refine_cfg["lr_clip"]
    mcw_sigma = refine_cfg["min_child_weight_sigma"]
    mcw_clip = refine_cfg["min_child_weight_clip"]
    ss_sigma = refine_cfg["subsample_sigma"]
    ss_clip = refine_cfg["subsample_clip"]
    cs_sigma = refine_cfg["colsample_sigma"]
    cs_clip = refine_cfg["colsample_clip"]
    gm_sigma = refine_cfg["gamma_sigma"]
    gm_clip = refine_cfg["gamma_clip"]
    ra_sigma = refine_cfg["reg_alpha_sigma"]
    ra_clip = refine_cfg["reg_alpha_exp_clip"]
    rl_sigma = refine_cfg["reg_lambda_sigma"]
    rl_clip = refine_cfg["reg_lambda_exp_clip"]
    mds_sigma = refine_cfg["max_delta_step_sigma"]
    mds_clip = refine_cfg["max_delta_step_clip"]

    return {
        "max_depth": int(np.clip(int(best["max_depth"] + rng.integers(md_delta[0], md_delta[1])), md_clip[0], md_clip[1])),
        "learning_rate": _clip(best["learning_rate"] * float(np.exp(rng.normal(0.0, lr_sigma))), lr_clip[0], lr_clip[1]),
        "min_child_weight": _clip(best["min_child_weight"] * float(np.exp(rng.normal(0.0, mcw_sigma))), mcw_clip[0], mcw_clip[1]),
        "subsample": _clip(best["subsample"] + rng.normal(0.0, ss_sigma), ss_clip[0], ss_clip[1]),
        "colsample_bytree": _clip(best["colsample_bytree"] + rng.normal(0.0, cs_sigma), cs_clip[0], cs_clip[1]),
        "gamma": _clip(best["gamma"] + rng.normal(0.0, gm_sigma), gm_clip[0], gm_clip[1]),
        "reg_alpha": _log_jitter(rng, best["reg_alpha"], sigma=ra_sigma, lo_exp=ra_clip[0], hi_exp=ra_clip[1]),
        "reg_lambda": _log_jitter(rng, best["reg_lambda"], sigma=rl_sigma, lo_exp=rl_clip[0], hi_exp=rl_clip[1]),
        "max_delta_step": _clip(best["max_delta_step"] + rng.normal(0.0, mds_sigma), mds_clip[0], mds_clip[1]),
    }


def sample_refine_low_lr(rng, best, refine_cfg, lowlr_cfg):
    """
    Refine with lower learning rate for stability.
    Matches notebook sample_refine_low_lr() exactly.
    """
    lr_shift = lowlr_cfg["lr_shift"]
    lr_clip = lowlr_cfg["lr_clip"]
    lr_sigma = refine_cfg["lr_sigma"]
    low_lr = _clip(best["learning_rate"] * float(np.exp(rng.normal(lr_shift, lr_sigma))), lr_clip[0], lr_clip[1])
    params = sample_refine(rng, best, refine_cfg)
    params["learning_rate"] = low_lr
    return params


def split_valid_for_es_and_score(
    Xv: pd.DataFrame, 
    yv: pd.Series, 
    wv: np.ndarray,
    valid_es_start: str,
    valid_es_end: str,
    valid_score_start: str,
    valid_score_end: str
) -> Tuple[Tuple, Tuple, str]:
    """
    Split validation into ES (early stopping) and SCORE (model selection) sets.
    Matches notebook split_valid_for_es_and_score() exactly.
    """
    if not isinstance(Xv.index, pd.DatetimeIndex):
        return (Xv, yv, wv), (Xv, yv, wv), "FULL_VALID"

    es_start = pd.Timestamp(valid_es_start)
    es_end = pd.Timestamp(valid_es_end)
    sc_start = pd.Timestamp(valid_score_start)
    sc_end = pd.Timestamp(valid_score_end)
    
    mask_es = (Xv.index >= es_start) & (Xv.index <= es_end)
    mask_sc = (Xv.index >= sc_start) & (Xv.index <= sc_end)

    X_es = Xv.loc[mask_es]
    y_es = yv.loc[mask_es]
    X_sc = Xv.loc[mask_sc]
    y_sc = yv.loc[mask_sc]

    wv_s = pd.Series(wv, index=Xv.index)
    w_es = wv_s.loc[mask_es].to_numpy(dtype=float)
    w_sc = wv_s.loc[mask_sc].to_numpy(dtype=float)

    mode_str = f"VALID_ES={valid_es_start}:{valid_es_end} / VALID_SCORE={valid_score_start}:{valid_score_end}"
    
    if len(X_es) > 0 and len(X_sc) > 0:
        return (X_es, y_es, w_es), (X_sc, y_sc, w_sc), mode_str

    return (Xv, yv, wv), (Xv, yv, wv), "FULL_VALID"


def is_better(row, best_row, tie_tol):
    """
    Check if row is better than best_row (lower wRMSE, tie-break by higher iteration).
    Matches notebook is_better() exactly.
    """
    if best_row is None:
        return True

    a = float(row["valid_sc_wrmse"])
    b = float(best_row["valid_sc_wrmse"])

    if a < (b - tie_tol):
        return True
    if abs(a - b) <= tie_tol:
        ai = -1 if row["best_iteration"] is None else int(row["best_iteration"])
        bi = -1 if best_row["best_iteration"] is None else int(best_row["best_iteration"])
        return ai > bi
    return False


def run_hpo(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    selected_features: list,
    config: Dict,
    ms_out_local: Optional[Path] = None,
    ms_out_drive: Optional[Path] = None,
    proc_data_local: Optional[Path] = None,
    proc_data_drive: Optional[Path] = None
) -> Tuple[Dict, xgb.XGBRegressor, pd.DataFrame]:
    """
    Run 3-stage hyperparameter optimization.
    Matches notebook Cell 58 (BLOCK 24) exactly.
    
    Uses manual sampling: sample_broad -> sample_refine -> sample_refine_low_lr
    
    Saves to ms_out_local/ms_out_drive:
        - best_model_xgb_reg_t1.json
        - best_params_xgb_reg_t1.txt
        - best_params_xgb_reg_t1.pkl
    
    Saves to proc_data_local/proc_data_drive:
        - best_params_xgb_reg_t1.pkl (persistent for future runs without HPO)
    
    Returns:
        Tuple of (best_params, best_model, results_df)
    """
    HPO_CFG = config
    
    # -------------------------
    # Prepare data
    # -------------------------
    y_train_f = y_train.astype(float).copy()
    y_valid_f = y_valid.astype(float).copy()
    
    w_train_arr = np.asarray(w_train, dtype=float)
    w_valid_arr = np.asarray(w_valid, dtype=float)
    
    # -------------------------
    # Align VALID data with Neural Networks (skip first lookback-1 rows)
    # -------------------------
    LOOKBACK = int(HPO_CFG["lookback"])
    SKIP_ROWS = LOOKBACK - 1
    
    X_valid_sel = X_valid.iloc[SKIP_ROWS:]
    y_valid_f = y_valid_f.iloc[SKIP_ROWS:]
    w_valid_arr = w_valid_arr[SKIP_ROWS:]
    
    print(f"[INFO] Aligned VALID with lookback={LOOKBACK}: skipped first {SKIP_ROWS} rows")
    print(f"[INFO] VALID shape after alignment: {X_valid_sel.shape}")
    
    # -------------------------
    # VALID split for ES vs SCORE (date-based)
    # -------------------------
    VALID_ES_START = HPO_CFG["valid_es_start"]
    VALID_ES_END = HPO_CFG["valid_es_end"]
    VALID_SCORE_START = HPO_CFG["valid_score_start"]
    VALID_SCORE_END = HPO_CFG["valid_score_end"]
    
    (X_valid_es, y_valid_es, w_valid_es), (X_valid_sc, y_valid_sc, w_valid_sc), valid_mode = split_valid_for_es_and_score(
        X_valid_sel, y_valid_f, w_valid_arr,
        VALID_ES_START, VALID_ES_END, VALID_SCORE_START, VALID_SCORE_END
    )
    
    print("[INFO] VALID mode:", valid_mode)
    print("[INFO] VALID_ES shape:", X_valid_es.shape, "| VALID_SCORE shape:", X_valid_sc.shape)
    
    # -------------------------
    # Search configuration from config
    # -------------------------
    RANDOM_SEED = int(HPO_CFG["random_state"])
    rng = np.random.default_rng(RANDOM_SEED)
    
    N_ESTIMATORS = int(HPO_CFG["n_estimators"])
    EARLY_STOP = int(HPO_CFG["early_stopping_rounds"])
    
    N_TRIALS_STAGE1 = int(HPO_CFG["n_trials_stage1"])
    N_TRIALS_STAGE2 = int(HPO_CFG["n_trials_stage2"])
    N_TRIALS_STAGE2_LOWLR = int(HPO_CFG["n_trials_stage2_lowlr"])
    
    PRINT_EVERY_STAGE1 = int(HPO_CFG["print_every_stage1"])
    PRINT_EVERY_STAGE2 = int(HPO_CFG["print_every_stage2"])
    
    TIE_TOL = float(HPO_CFG["tie_tol"])
    
    BASE_MODEL_CFG = dict(
        n_estimators=N_ESTIMATORS,
        objective=HPO_CFG["objective"],
        eval_metric=HPO_CFG["eval_metric"],
        tree_method=HPO_CFG["tree_method"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOP,
    )
    
    # -------------------------
    # Parameter sampling config
    # -------------------------
    SAMP_CFG = HPO_CFG["sampling"]
    BROAD_CFG = SAMP_CFG["broad"]
    REFINE_CFG = SAMP_CFG["refine"]
    LOWLR_CFG = SAMP_CFG["refine_low_lr"]
    
    # -------------------------
    # Single trial runner
    # -------------------------
    def run_trial(trial_id: int, stage: str, params: dict):
        """Run a single HPO trial."""
        model = xgb.XGBRegressor(**BASE_MODEL_CFG, **params)

        t0 = time.time()
        model.fit(
            X_train, y_train_f,
            sample_weight=w_train_arr,
            eval_set=[(X_valid_es, y_valid_es)],
            sample_weight_eval_set=[w_valid_es],
            verbose=False
        )
        elapsed = time.time() - t0

        best_iter = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)

        pred_es = model.predict(X_valid_es)
        es_wrmse = w_rmse(y_valid_es, pred_es, w_valid_es)

        pred_sc = model.predict(X_valid_sc)
        sc_wrmse = w_rmse(y_valid_sc, pred_sc, w_valid_sc)

        row = {
            "stage": stage,
            "trial": int(trial_id),
            "valid_sc_wrmse": sc_wrmse,
            "valid_sc_wmae": w_mae(y_valid_sc, pred_sc, w_valid_sc),
            "valid_sc_diracc": dir_acc(y_valid_sc, pred_sc),
            "best_iteration": None if best_iter is None else int(best_iter),
            "best_score_rmse_eval_es": None if best_score is None else float(best_score),
            "valid_es_wrmse_explicit": float(es_wrmse),
            "elapsed_sec": float(elapsed),
            **params,
        }
        return model, row
    
    # -------------------------
    # HPO loop (Stage 1 + Stage 2 + Stage 2 LOW-LR)
    # -------------------------
    results = []
    best_model = None
    best_row = None
    best_valid = np.inf
    
    best_params_keys = [
        "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree",
        "gamma", "reg_alpha", "reg_lambda", "max_delta_step"
    ]
    
    print(f"\n[INFO] HPO start | VALID mode: {valid_mode}")
    print(f"[INFO] Stage1 trials={N_TRIALS_STAGE1} | Stage2 trials={N_TRIALS_STAGE2} | Stage2 LOW-LR={N_TRIALS_STAGE2_LOWLR}")
    print(f"[INFO] n_estimators={N_ESTIMATORS} | early_stop={EARLY_STOP}")
    
    # Stage 1 (broad)
    for i in range(N_TRIALS_STAGE1):
        params = sample_broad(rng, BROAD_CFG)
        model_i, row_i = run_trial(trial_id=i, stage="STAGE1_BROAD", params=params)
        results.append(row_i)

        if is_better(row_i, best_row, TIE_TOL):
            best_model = model_i
            best_row = row_i
            best_valid = float(best_row["valid_sc_wrmse"])

        if (i + 1) % PRINT_EVERY_STAGE1 == 0 or i == 0:
            print(f"[INFO] S1 {i:04d} | sc_wrmse={row_i['valid_sc_wrmse']:.6f} | best={best_valid:.6f} | best_iter={row_i['best_iteration']}")
    
    best_params_stage1 = {k: best_row[k] for k in best_params_keys}
    
    # Stage 2 (refine around best)
    for j in range(N_TRIALS_STAGE2):
        params = sample_refine(rng, best_params_stage1, REFINE_CFG)
        model_j, row_j = run_trial(trial_id=j, stage="STAGE2_REFINE", params=params)
        results.append(row_j)

        if is_better(row_j, best_row, TIE_TOL):
            best_model = model_j
            best_row = row_j
            best_valid = float(best_row["valid_sc_wrmse"])

        if (j + 1) % PRINT_EVERY_STAGE2 == 0 or j == 0:
            print(f"[INFO] S2 {j:04d} | sc_wrmse={row_j['valid_sc_wrmse']:.6f} | best={best_valid:.6f} | best_iter={row_j['best_iteration']}")
    
    # Stage 2B (LOW-LR refine branch)
    for k in range(N_TRIALS_STAGE2_LOWLR):
        params = sample_refine_low_lr(rng, best_params_stage1, REFINE_CFG, LOWLR_CFG)
        model_k, row_k = run_trial(trial_id=k, stage="STAGE2_LOWLR", params=params)
        results.append(row_k)

        if is_better(row_k, best_row, TIE_TOL):
            best_model = model_k
            best_row = row_k
            best_valid = float(best_row["valid_sc_wrmse"])

        if (k + 1) % PRINT_EVERY_STAGE2 == 0 or k == 0:
            print(f"[INFO] S2L {k:04d} | sc_wrmse={row_k['valid_sc_wrmse']:.6f} | best={best_valid:.6f} | best_iter={row_k['best_iteration']}")
    
    res_df = pd.DataFrame(results).sort_values("valid_sc_wrmse", ascending=True).reset_index(drop=True)
    
    print("\n[INFO] Top 10 trials by VALID_SCORE wRMSE:")
    print(res_df.head(10).to_string())
    
    print("\n[INFO] BEST summary:")
    best_summary = {
        "valid_mode": valid_mode,
        "n_features": int(len(selected_features)),
        "n_trials_total": int(len(res_df)),
        "best_valid_sc_wrmse": float(best_row["valid_sc_wrmse"]),
        "best_valid_sc_wmae": float(best_row["valid_sc_wmae"]),
        "best_valid_sc_diracc": float(best_row["valid_sc_diracc"]),
        "best_iteration": best_row["best_iteration"],
    }
    print(pd.Series(best_summary))
    
    best_params = {k: best_row[k] for k in best_params_keys}
    best_params["max_depth"] = int(best_params["max_depth"])
    for k in best_params_keys:
        if k != "max_depth":
            best_params[k] = float(best_params[k])
    
    print("\n[INFO] BEST params:")
    print(best_params)
    
    # -------------------------
    # Save outputs (LOCAL + DRIVE)
    # -------------------------
    if ms_out_local:
        ms_out_local = ensure_dir(Path(ms_out_local))
        
        BEST_MODEL_PATH = ms_out_local / "best_model_xgb_reg_t1.json"
        BEST_PARAMS_TXT = ms_out_local / "best_params_xgb_reg_t1.txt"
        BEST_PARAMS_PKL = ms_out_local / "best_params_xgb_reg_t1.pkl"
        
        # Save model
        best_model.get_booster().save_model(str(BEST_MODEL_PATH))
        
        # Save params as text
        lines = [
            f"valid_mode={valid_mode}",
            f"best_valid_sc_wrmse={best_row['valid_sc_wrmse']}",
            f"best_valid_sc_wmae={best_row['valid_sc_wmae']}",
            f"best_valid_sc_diracc={best_row['valid_sc_diracc']}",
            f"best_iteration={best_row['best_iteration']}",
            f"n_estimators={N_ESTIMATORS}",
            f"early_stop={EARLY_STOP}",
            f"trials_stage1={N_TRIALS_STAGE1}",
            f"trials_stage2={N_TRIALS_STAGE2}",
            f"trials_stage2_lowlr={N_TRIALS_STAGE2_LOWLR}",
            f"random_seed={RANDOM_SEED}",
        ]
        for k in best_params_keys:
            lines.append(f"{k}={best_params[k]}")
        
        BEST_PARAMS_TXT.write_text("\n".join(lines), encoding="utf-8")
        
        # Save params as pickle
        save_pickle(best_params, BEST_PARAMS_PKL)
        
        if ms_out_drive:
            ms_out_drive = ensure_dir(Path(ms_out_drive))
            copy_file(BEST_MODEL_PATH, ms_out_drive / BEST_MODEL_PATH.name)
            copy_file(BEST_PARAMS_TXT, ms_out_drive / BEST_PARAMS_TXT.name)
            copy_file(BEST_PARAMS_PKL, ms_out_drive / BEST_PARAMS_PKL.name)
        
        print("\n[OK] Saved HPO outputs:")
        print("  -", BEST_MODEL_PATH.name)
        print("  -", BEST_PARAMS_TXT.name)
        print("  -", BEST_PARAMS_PKL.name)
    
    # Save ALSO to persistent location (for future runs without HPO)
    if proc_data_local:
        proc_data_local = ensure_dir(Path(proc_data_local))
        save_pickle(best_params, proc_data_local / "best_params_xgb_reg_t1.pkl")
        
        if proc_data_drive:
            proc_data_drive = ensure_dir(Path(proc_data_drive))
            copy_file(proc_data_local / "best_params_xgb_reg_t1.pkl", 
                      proc_data_drive / "best_params_xgb_reg_t1.pkl")
        
        print("  - best_params_xgb_reg_t1.pkl (persistent)")
    
    print("[OK] BLOCK 24 complete.")
    
    return best_params, best_model, res_df
