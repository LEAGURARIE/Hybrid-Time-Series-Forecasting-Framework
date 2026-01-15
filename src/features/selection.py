"""
Feature selection module for Google Stock ML project.
Matches notebook Cell 55 (BLOCK 23) exactly.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils import w_rmse, save_pickle, copy_file, ensure_dir, _to_np, EPS


# ==============================================================================
# CELL 55 (BLOCK 23): FEATURE SELECTION
# ==============================================================================

def xgb_feature_selection(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    config: Dict,
    fs_out_local: Optional[Path] = None,
    fs_out_drive: Optional[Path] = None,
    proc_data_local: Optional[Path] = None,
    proc_data_drive: Optional[Path] = None,
    target_col: str = "target_t1",
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    XGBoost-based feature selection.
    Matches notebook Cell 55 (BLOCK 23) exactly.
    
    Steps:
    1. Identify missingness flags (for dependency closure)
    2. Build matrices for FS (exclude flags)
    3. Drop constant TRAIN columns
    4. Spearman pairwise filter (keep feature with higher corr to y)
    5. XGBoost FS model (GAIN importance from booster)
    6. Permutation importance on VALID (custom neg_weighted_rmse scorer)
    7. Selection policy + fallback ladder
    8. Dependency closure: add X_is_missing if exists
    9. Save outputs (LOCAL + DRIVE)
    
    Saves to fs_out_local/fs_out_drive:
        - selected_features_xgb.txt
        - selected_features_xgb.pkl
        - selected_features_xgb.csv
        - feature_importance_gain.csv
        - feature_importance_permutation_valid.csv
    
    Saves to proc_data_local/proc_data_drive:
        - X_train_xgb_selected.pkl
        - X_valid_xgb_selected.pkl
        - X_test_xgb_selected.pkl
    
    Returns:
        Tuple of (selected_features, gain_importance_df, perm_importance_df)
    """
    MISSING_SUFFIX = "_is_missing"
    
    # Config
    XGB_FS_CFG = config
    
    # -------------------------
    # 1) Identify missingness flags (for dependency closure)
    # -------------------------
    all_train_cols = list(X_train.columns)
    missing_flags = [c for c in all_train_cols if c.endswith(MISSING_SUFFIX)]
    missing_flag_set = set(missing_flags)
    
    base_to_missing = {}
    for flag in missing_flags:
        base = flag[:-len(MISSING_SUFFIX)]
        if base in missing_flag_set:
            continue
        if base in all_train_cols:
            base_to_missing[base] = flag
    
    print(f"[INFO] Found missingness flags: {len(missing_flags)}")
    
    # -------------------------
    # 2) Build matrices for FS (exclude flags)
    # -------------------------
    nonflag_cols = [c for c in all_train_cols if c not in missing_flag_set and c != target_col and c != "sample_weight"]
    
    Xtr = X_train.loc[:, nonflag_cols].copy()
    Xva = X_valid.loc[:, nonflag_cols].copy()
    
    ytr = y_train.astype(float).copy()
    yva = y_valid.astype(float).copy()
    
    wtr = _to_np(w_train)
    wva = _to_np(w_valid)
    
    # -------------------------
    # 3) Drop constant TRAIN columns
    # -------------------------
    nunique = Xtr.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[INFO] Dropping constant TRAIN columns: {len(const_cols)}")
        Xtr = Xtr.drop(columns=const_cols)
        Xva = Xva.drop(columns=const_cols, errors="ignore")
    
    # -------------------------
    # 4) Spearman filter (pairwise between features)
    # -------------------------
    SPEARMAN_THRESH = float(XGB_FS_CFG["spearman_thresh"])
    
    corr_ff = Xtr.corr(method="spearman").abs().fillna(0.0)
    cols = list(corr_ff.columns)
    
    feat_to_y = Xtr.apply(lambda s: s.corr(ytr, method="spearman"))
    feat_to_y_abs = feat_to_y.abs().fillna(0.0)
    
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            cval = float(corr_ff.iat[i, j])
            if cval > SPEARMAN_THRESH:
                pairs.append((cols[i], cols[j], cval))
    pairs.sort(key=lambda t: t[2], reverse=True)
    
    active = set(cols)
    
    for a, b, cval in pairs:
        if (a in active) and (b in active):
            ay = float(feat_to_y_abs.get(a, 0.0))
            by = float(feat_to_y_abs.get(b, 0.0))
            drop = a if ay < by else b
            active.remove(drop)
    
    kept_cols = [c for c in cols if c in active]
    print(f"[INFO] Spearman filter: start={len(cols)} | kept={len(kept_cols)} | thresh={SPEARMAN_THRESH}")
    
    X_train_fs = Xtr.loc[:, kept_cols].copy()
    X_valid_fs = Xva.loc[:, kept_cols].copy()
    
    # -------------------------
    # 5) XGBoost FS model (GAIN)
    # -------------------------
    xgb_params = dict(
        n_estimators=int(XGB_FS_CFG["n_estimators"]),
        learning_rate=float(XGB_FS_CFG["learning_rate"]),
        max_depth=int(XGB_FS_CFG["max_depth"]),
        min_child_weight=float(XGB_FS_CFG["min_child_weight"]),
        gamma=float(XGB_FS_CFG["gamma"]),
        subsample=float(XGB_FS_CFG["subsample"]),
        colsample_bytree=float(XGB_FS_CFG["colsample_bytree"]),
        reg_alpha=float(XGB_FS_CFG["reg_alpha"]),
        reg_lambda=float(XGB_FS_CFG["reg_lambda"]),
        max_delta_step=float(XGB_FS_CFG["max_delta_step"]),
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=int(XGB_FS_CFG["random_state"]),
        n_jobs=-1,
        early_stopping_rounds=int(XGB_FS_CFG["early_stopping_rounds"]),
    )
    
    print("[INFO] Training XGBoost for feature selection...")
    model_fs = xgb.XGBRegressor(**xgb_params)
    model_fs.fit(
        X_train_fs, ytr.loc[X_train_fs.index],
        sample_weight=wtr,
        eval_set=[(X_valid_fs, yva.loc[X_valid_fs.index])],
        sample_weight_eval_set=[wva],
        verbose=False
    )
    
    best_iter = getattr(model_fs, "best_iteration", None)
    best_score = getattr(model_fs, "best_score", None)
    print(f"[INFO] XGB train done. best_iteration={best_iter} | best_score(valid_rmse)={best_score}")
    
    # Get GAIN importance from booster (matches notebook exactly)
    booster = model_fs.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    
    imp_gain = pd.DataFrame({"feature": kept_cols})
    imp_gain["gain"] = imp_gain["feature"].map(score_gain).fillna(0.0).astype(float)
    imp_gain = imp_gain.sort_values("gain", ascending=False).reset_index(drop=True)
    
    gain_sum = float(imp_gain["gain"].sum())
    imp_gain["gain_frac"] = imp_gain["gain"] / (gain_sum + EPS)
    imp_gain["cum_gain"] = imp_gain["gain_frac"].cumsum()
    
    # -------------------------
    # 6) Permutation importance on VALID (custom scorer)
    # -------------------------
    def neg_weighted_rmse_scorer(estimator, X, y):
        p = estimator.predict(X)
        return -w_rmse(y, p, wva)
    
    PERM_REPEATS = int(XGB_FS_CFG["perm_repeats"])
    
    print(f"[INFO] Computing permutation importance (repeats={PERM_REPEATS})...")
    perm = permutation_importance(
        model_fs,
        X_valid_fs,
        yva.loc[X_valid_fs.index],
        scoring=neg_weighted_rmse_scorer,
        n_repeats=PERM_REPEATS,
        random_state=int(XGB_FS_CFG["random_state"]),
        n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        "feature": X_valid_fs.columns,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std
    }).reset_index(drop=True)
    
    perm_df = perm_df.merge(
        imp_gain.loc[:, ["feature", "gain", "gain_frac", "cum_gain"]],
        on="feature",
        how="left"
    )
    
    # -------------------------
    # 7) Selection policy + fallback ladder
    # -------------------------
    GAIN_CUM_THRESH = float(XGB_FS_CFG["gain_cum_thresh"])
    MIN_FEATURES = int(XGB_FS_CFG["min_features"])
    NEG_SIGMA = float(XGB_FS_CFG["neg_sigma"])
    POS_SIGMA = float(XGB_FS_CFG["pos_sigma"])
    MIN_GAIN = float(XGB_FS_CFG["min_gain"])
    
    perm_df["perm_strongly_negative"] = (
        (perm_df["perm_importance_mean"] < 0) &
        (np.abs(perm_df["perm_importance_mean"]) > (NEG_SIGMA * (perm_df["perm_importance_std"] + EPS)))
    )
    
    perm_df["perm_confident_positive"] = (
        (perm_df["perm_importance_mean"] > 0) &
        (perm_df["perm_importance_mean"] > (POS_SIGMA * (perm_df["perm_importance_std"] + EPS)))
    )
    
    strong_neg_set = set(perm_df.loc[perm_df["perm_strongly_negative"], "feature"].tolist())
    print(f"[DEBUG] Features in strong_neg_set: {len(strong_neg_set)}")
    
    gain_candidates = imp_gain.loc[
        (imp_gain["cum_gain"] <= GAIN_CUM_THRESH) | (imp_gain.index < MIN_FEATURES),
        "feature"
    ].tolist()
    print(f"[INFO] Candidates by GAIN: cum<={GAIN_CUM_THRESH} with min {MIN_FEATURES} => {len(gain_candidates)}")
    
    perm_map = perm_df.set_index("feature")[["perm_confident_positive", "perm_importance_mean", "gain"]]
    
    # Debug: check overlap
    in_perm_map = sum(1 for f in gain_candidates if f in perm_map.index)
    print(f"[DEBUG] gain_candidates in perm_map: {in_perm_map}/{len(gain_candidates)}")
    
    # Debug: check perm stats
    n_confident_pos = perm_df["perm_confident_positive"].sum()
    n_perm_positive = (perm_df["perm_importance_mean"] > 0).sum()
    print(f"[DEBUG] perm stats: confident_positive={n_confident_pos}, perm_mean>0={n_perm_positive}, total={len(perm_df)}")
    
    # Debug: check gain distribution
    n_gain_positive = (imp_gain["gain"] > 0).sum()
    n_gain_zero = (imp_gain["gain"] == 0).sum()
    print(f"[DEBUG] gain stats: gain>0={n_gain_positive}, gain=0={n_gain_zero}, total={len(imp_gain)}")
    
    # Debug: count how many features pass each filter
    debug_in_strong_neg = 0
    debug_gain_zero = 0
    debug_not_confident_pos = 0
    debug_passed = 0
    
    selected = []
    for f in gain_candidates:
        if f in strong_neg_set:
            debug_in_strong_neg += 1
            continue
        g = float(perm_map.loc[f, "gain"]) if f in perm_map.index else 0.0
        if g < MIN_GAIN:
            debug_gain_zero += 1
            continue
        is_pos = bool(perm_map.loc[f, "perm_confident_positive"]) if f in perm_map.index else False
        if not is_pos:
            debug_not_confident_pos += 1
            continue
        debug_passed += 1
        selected.append(f)
    
    print(f"[DEBUG] Strict filter breakdown: strong_neg={debug_in_strong_neg}, gain<min={debug_gain_zero}, not_confident_pos={debug_not_confident_pos}, passed={debug_passed}")
    
    print(f"[INFO] Selected after strict filters: {len(selected)}")
    
    # Fallback 1: relax to perm_mean > 0
    if len(selected) < MIN_FEATURES:
        print("[WARN] Too few features; relaxing to perm_mean > 0.")
        selected = []
        debug_in_strong_neg = 0
        debug_gain_zero = 0
        debug_perm_neg = 0
        debug_passed = 0
        for f in gain_candidates:
            if f in strong_neg_set:
                debug_in_strong_neg += 1
                continue
            g = float(perm_map.loc[f, "gain"]) if f in perm_map.index else 0.0
            if g < MIN_GAIN:
                debug_gain_zero += 1
                continue
            pm = float(perm_map.loc[f, "perm_importance_mean"]) if f in perm_map.index else -1.0
            if pm <= 0:
                debug_perm_neg += 1
                continue
            debug_passed += 1
            selected.append(f)
        print(f"[DEBUG] Fallback1 breakdown: strong_neg={debug_in_strong_neg}, gain<min={debug_gain_zero}, perm<=0={debug_perm_neg}, passed={debug_passed}")
        print(f"[INFO] Selected after relaxed filter: {len(selected)}")
    
    # Fallback 2: gain candidates excluding strong-neg
    if len(selected) < MIN_FEATURES:
        print("[WARN] Still too few; final fallback to GAIN candidates excluding strongly-negative.")
        selected = []
        debug_in_strong_neg = 0
        debug_gain_zero = 0
        debug_passed = 0
        for f in gain_candidates:
            if f in strong_neg_set:
                debug_in_strong_neg += 1
                continue
            g = float(perm_map.loc[f, "gain"]) if f in perm_map.index else 0.0
            if g < MIN_GAIN:
                debug_gain_zero += 1
                continue
            debug_passed += 1
            selected.append(f)
        print(f"[DEBUG] Fallback2 breakdown: strong_neg={debug_in_strong_neg}, gain<min={debug_gain_zero}, passed={debug_passed}")
        print(f"[INFO] Selected after final fallback: {len(selected)}")
    
    # Fallback 3: GUARANTEE min_features - take top by XGB GAIN
    if len(selected) < MIN_FEATURES:
        print(f"[WARN] Final fallback: taking top {MIN_FEATURES} features by XGB GAIN.")
        top_by_gain = imp_gain.head(MIN_FEATURES)["feature"].tolist()
        selected = top_by_gain
        print(f"[INFO] Selected after GAIN-only fallback: {len(selected)}")
    
    # -------------------------
    # 8) Dependency closure: add X_is_missing if exists
    # -------------------------
    selected_set = set(selected)
    flags_added = []
    
    for base, flag in base_to_missing.items():
        if (base in selected_set) and (flag in X_train.columns) and (flag not in selected_set):
            selected.append(flag)
            selected_set.add(flag)
            flags_added.append(flag)
    
    print(f"[INFO] Missingness flags added: {len(flags_added)}")
    
    selected_final = list(dict.fromkeys(selected))
    print(f"[INFO] FINAL selected features: {len(selected_final)}")
    
    # -------------------------
    # 9) Save outputs (LOCAL + DRIVE)
    # -------------------------
    if fs_out_local:
        fs_out_local = ensure_dir(Path(fs_out_local))
        
        # Selected features
        sel_txt = fs_out_local / "selected_features_xgb.txt"
        sel_pkl = fs_out_local / "selected_features_xgb.pkl"
        sel_csv = fs_out_local / "selected_features_xgb.csv"
        
        sel_txt.write_text("\n".join(selected_final), encoding="utf-8")
        save_pickle(selected_final, sel_pkl)
        pd.DataFrame({"feature": selected_final}).to_csv(sel_csv, index=False)
        
        if fs_out_drive:
            fs_out_drive = ensure_dir(Path(fs_out_drive))
            copy_file(sel_txt, fs_out_drive / sel_txt.name)
            copy_file(sel_pkl, fs_out_drive / sel_pkl.name)
            copy_file(sel_csv, fs_out_drive / sel_csv.name)
        
        # Importance tables
        imp_gain_csv = fs_out_local / "feature_importance_gain.csv"
        perm_csv = fs_out_local / "feature_importance_permutation_valid.csv"
        
        imp_gain.to_csv(imp_gain_csv, index=False)
        perm_df.sort_values("perm_importance_mean", ascending=False).to_csv(perm_csv, index=False)
        
        if fs_out_drive:
            copy_file(imp_gain_csv, fs_out_drive / imp_gain_csv.name)
            copy_file(perm_csv, fs_out_drive / perm_csv.name)
        
        print("\n[OK] Saved XGB Feature Selection artifacts:")
        print("  -", sel_txt.name)
        print("  -", sel_pkl.name)
        print("  -", sel_csv.name)
        print("  -", imp_gain_csv.name)
        print("  -", perm_csv.name)
    
    # Create and save filtered matrices (xgb_selected) to data/processed/
    if proc_data_local:
        proc_data_local = ensure_dir(Path(proc_data_local))
        
        X_train_xgb_selected = X_train.loc[:, selected_final].copy()
        X_valid_xgb_selected = X_valid.loc[:, selected_final].copy()
        X_test_xgb_selected = X_test.loc[:, selected_final].copy()
        
        X_train_xgb_selected.to_pickle(proc_data_local / "X_train_xgb_selected.pkl")
        X_valid_xgb_selected.to_pickle(proc_data_local / "X_valid_xgb_selected.pkl")
        X_test_xgb_selected.to_pickle(proc_data_local / "X_test_xgb_selected.pkl")
        
        if proc_data_drive:
            proc_data_drive = ensure_dir(Path(proc_data_drive))
            copy_file(proc_data_local / "X_train_xgb_selected.pkl", proc_data_drive / "X_train_xgb_selected.pkl")
            copy_file(proc_data_local / "X_valid_xgb_selected.pkl", proc_data_drive / "X_valid_xgb_selected.pkl")
            copy_file(proc_data_local / "X_test_xgb_selected.pkl", proc_data_drive / "X_test_xgb_selected.pkl")
        
        print(f"[OK] Saved to {proc_data_local}:")
        print(f"  - X_train_xgb_selected.pkl ({X_train_xgb_selected.shape})")
        print(f"  - X_valid_xgb_selected.pkl ({X_valid_xgb_selected.shape})")
        print(f"  - X_test_xgb_selected.pkl ({X_test_xgb_selected.shape})")
    
    print("[OK] BLOCK 23 complete.")
    
    return selected_final, imp_gain, perm_df
