"""
Train/Valid/Test split functionality.
Matches notebook Cells 36, 48, 50, 52 exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..utils import save_pickle, load_pickle, compute_sample_weights, ensure_dir


# ==============================================================================
# CELL 36: LIMIT DATA (BLOCK 15)
# ==============================================================================

def limit_data(
    full_df: pd.DataFrame,
    limit_start_date: str,
    interim_dir: Path = None
) -> pd.DataFrame:
    """
    Limit data to start from a specific date.
    Matches notebook Cell 36 (BLOCK 15) exactly.
    
    Args:
        full_df: Full DataFrame with features
        limit_start_date: Start date to limit data from (e.g., "2015-12-31")
        interim_dir: Directory to save full_df.pkl
    
    Returns:
        DataFrame limited to dates >= limit_start_date
    """
    assert isinstance(full_df.index, pd.DatetimeIndex), "[ERROR] full_df.index must be a DatetimeIndex."
    full_df = full_df.sort_index()
    
    # Limit start date
    limit_start = str(limit_start_date)
    full_df = full_df.loc[limit_start:].copy()
    
    # Save to interim
    if interim_dir:
        interim_dir = ensure_dir(Path(interim_dir))
        out_path = interim_dir / "full_df.pkl"
        full_df.to_pickle(out_path)
        print(f"[OK] Saved full_df snapshot to: {out_path}")
    
    print(f"[INFO] full_df rows: {len(full_df)} | range: {full_df.index.min()} -> {full_df.index.max()}")
    
    return full_df


# ==============================================================================
# CELL 48: DATA SPLIT (BLOCK 20)
# ==============================================================================

def split_data(
    full_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Split data into train/valid/test sets.
    Matches notebook Cell 48 (BLOCK 20) exactly.
    
    Note: Data should already be limited by limit_data() before calling this.
    
    Saves:
        - X_train_xgb.pkl, X_valid_xgb.pkl, X_test_xgb.pkl
        - y_train.pkl, y_valid.pkl, y_test.pkl
        - weights_train.pkl, weights_valid.pkl, weights_test.pkl
    
    Args:
        full_df: Full DataFrame with features and target (already limited)
        config: Configuration dict with data and weights sections
        output_dir: Directory to save splits (processed dir)
    
    Returns:
        Dict with X_train, X_valid, X_test, y_train, y_valid, y_test,
        w_train, w_valid, w_test
    """
    TARGET = config["data"]["target_col"]
    full_df = full_df.dropna(subset=[TARGET])
    
    # Create date masks
    dates = full_df.index.normalize()
    train_mask = dates <= config["data"]["train_end"]
    valid_mask = (dates >= config["data"]["valid_start"]) & (dates <= config["data"]["valid_end"])
    
    # Test mask - include test_end if specified
    test_start = config["data"]["test_start"]
    test_end = config["data"].get("test_end")
    if test_end:
        test_mask = (dates >= test_start) & (dates <= test_end)
    else:
        test_mask = dates >= test_start
    
    # Feature columns (all except target)
    feature_cols = [c for c in full_df.columns if c != TARGET]
    
    # Build splits dict
    splits = {}
    for name, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        df = full_df[mask]
        splits[f"X_{name}"] = df[feature_cols].copy()
        splits[f"y_{name}"] = df[TARGET].copy()
        splits[f"w_{name}"] = compute_sample_weights(df[TARGET], **config["weights"])
        print(f"[INFO] {name}: {len(df)}")
    
    # Save to output_dir (matching notebook exactly)
    if output_dir:
        output_dir = ensure_dir(Path(output_dir))
        
        # X files - saved as X_*_xgb.pkl in notebook
        splits["X_train"].to_pickle(output_dir / "X_train_xgb.pkl")
        splits["X_valid"].to_pickle(output_dir / "X_valid_xgb.pkl")
        splits["X_test"].to_pickle(output_dir / "X_test_xgb.pkl")
        
        # y files
        save_pickle(splits["y_train"], output_dir / "y_train.pkl")
        save_pickle(splits["y_valid"], output_dir / "y_valid.pkl")
        save_pickle(splits["y_test"], output_dir / "y_test.pkl")
        
        # weights files
        save_pickle(splits["w_train"], output_dir / "weights_train.pkl")
        save_pickle(splits["w_valid"], output_dir / "weights_valid.pkl")
        save_pickle(splits["w_test"], output_dir / "weights_test.pkl")
        
        print(f"[OK] Saved splits to {output_dir}")
    
    return splits


# ==============================================================================
# CELL 50: NEURAL FEATURE SELECTION (MI-based)
# ==============================================================================

def _dedup_preserve_order(seq: List) -> List:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in seq:
        if isinstance(x, str) and x and (x not in seen):
            out.append(x)
            seen.add(x)
    return out


def _feature_group(col: str) -> str:
    """Extract feature group from column name."""
    if not isinstance(col, str) or not col:
        return "OTHER"
    if col.startswith("^"):
        return col.split("_", 1)[0]
    return col.split("_", 1)[0]


def _safe_spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation on numeric DataFrame."""
    return df.corr(method="spearman")


def _select_with_decorrelation(
    ranked_cols: List[str],
    X_train_num: pd.DataFrame,
    k: int,
    corr_threshold: float
) -> List[str]:
    """Select top-k features with de-correlation."""
    ranked_cols = [c for c in ranked_cols if c in X_train_num.columns]
    if k <= 0:
        return []
    
    selected = []
    if len(ranked_cols) == 0:
        return selected
    
    # Precompute correlation only once for speed
    sub = X_train_num.loc[:, ranked_cols]
    corr = _safe_spearman_corr(sub).abs()
    
    for c in ranked_cols:
        if len(selected) == 0:
            selected.append(c)
            if len(selected) >= k:
                break
            continue
        
        too_close = False
        for s in selected:
            val = corr.at[c, s]
            if pd.notna(val) and float(val) >= corr_threshold:
                too_close = True
                break
        if not too_close:
            selected.append(c)
            if len(selected) >= k:
                break
    
    return selected


def select_neural_features_mi(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    output_dir: Path = None
) -> Dict[str, List[str]]:
    """
    Select neural network features using Mutual Information.
    Matches notebook Cell 50 (BLOCK 21) exactly.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        config: Configuration dict with nn_feature_select section
        output_dir: Directory to save feature lists (fs_dir)
    
    Returns:
        Dict with 'neural_40' and 'neural_80' feature lists
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # Get params from config
    nn_cfg = config.get("nn_feature_select", {})
    N40 = int(nn_cfg.get("n40", 40))
    N80 = int(nn_cfg.get("n80", 80))
    per_group_40 = int(nn_cfg.get("per_group_40", 2))
    per_group_80 = int(nn_cfg.get("per_group_80", 4))
    corr_thr = float(nn_cfg.get("corr_thr", 0.85))
    mi_neighbors = int(nn_cfg.get("mi_n_neighbors", 5))
    mi_random_state = int(nn_cfg.get("mi_random_state", 42))
    
    # Numeric matrix (TRAIN only) + cleaning
    X = X_train.copy()
    y = y_train.astype(float).to_numpy()
    
    # Keep numeric dtypes only
    X = X.select_dtypes(include=[np.number]).copy()
    
    # Drop constant columns
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
        print(f"[INFO] Dropped {len(const_cols)} constant columns")
    
    # Replace inf -> nan, then drop cols that have any non-finite
    X = X.replace([np.inf, -np.inf], np.nan)
    bad_cols = [c for c in X.columns if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
    if bad_cols:
        X = X.drop(columns=bad_cols)
        print(f"[INFO] Dropped {len(bad_cols)} columns with non-finite values")
    
    if X.shape[1] == 0:
        raise ValueError("[ERROR] No usable numeric features left after cleaning on TRAIN.")
    
    print(f"[INFO] Features after cleaning: {X.shape[1]}")
    
    # Mutual Information ranking (TRAIN only)
    print("[INFO] Computing Mutual Information (this may take a minute)...")
    mi = mutual_info_regression(
        X.to_numpy(dtype=float),
        y,
        n_neighbors=mi_neighbors,
        random_state=mi_random_state,
    )
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    # Group-aware shortlist
    groups = {}
    for col in mi_series.index:
        g = _feature_group(col)
        groups.setdefault(g, []).append(col)
    
    def _build_group_seed(per_group):
        """Build seed list with per_group features from each group."""
        seed = []
        for g, cols in groups.items():
            take = cols[:per_group]
            seed.extend(take)
        return _dedup_preserve_order(seed)
    
    seed40 = _build_group_seed(per_group_40)
    seed80 = _build_group_seed(per_group_80)
    
    # Global ranked list
    ranked_all = mi_series.index.tolist()
    
    # Final pick with de-correlation
    rank40 = _dedup_preserve_order(seed40 + ranked_all)
    rank80 = _dedup_preserve_order(seed80 + ranked_all)
    
    neural_40 = _select_with_decorrelation(rank40, X, N40, corr_thr)
    neural_80 = _select_with_decorrelation(rank80, X, N80, corr_thr)
    
    if len(neural_40) == 0 or len(neural_80) == 0:
        raise ValueError("[ERROR] NN feature selection produced empty sets.")
    
    # Ensure 40 ⊆ 80 if possible
    if not set(neural_40).issubset(set(neural_80)):
        base = _dedup_preserve_order(neural_40 + neural_80 + ranked_all)
        neural_80 = _select_with_decorrelation(base, X, N80, corr_thr)
    
    # Save
    if output_dir:
        output_dir = ensure_dir(Path(output_dir))
        p40 = output_dir / "neural_feature_cols_40_bygroup.pkl"
        p80 = output_dir / "neural_feature_cols_80_bygroup.pkl"
        save_pickle(neural_40, p40)
        save_pickle(neural_80, p80)
        
        print("[OK] NN feature lists saved:")
        print(f"  - {p40} | n={len(neural_40)}")
        print(f"  - {p80} | n={len(neural_80)}")
    
    # Diagnostics
    print(f"[INFO] Top-10 MI features (TRAIN): {mi_series.head(10).index.tolist()}")
    print(f"[INFO] Groups covered in NEURAL40: {sorted({_feature_group(c) for c in neural_40})}")
    print(f"[INFO] Groups covered in NEURAL80: {sorted({_feature_group(c) for c in neural_80})}")
    
    return {
        "neural_40": neural_40,
        "neural_80": neural_80,
        "mi_series": mi_series,
    }


# ==============================================================================
# CELL 52: NEURAL FEATURE RESOLUTION
# ==============================================================================

def resolve_neural_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    neural_cols: List[str],
    n_features: int,
    output_dir: Path = None,
    fs_output_dir: Path = None
) -> Dict[str, Any]:
    """
    Resolve and validate neural feature columns across splits.
    Matches notebook Cell 52 (BLOCK 22) exactly.
    
    Args:
        X_train, X_valid, X_test: Feature DataFrames
        neural_cols: List of neural feature column names
        n_features: Number of features (40 or 80) for naming
        output_dir: Directory for X_neural files (processed dir)
        fs_output_dir: Directory for resolved feature list
    
    Returns:
        Dict with X_train_neural, X_valid_neural, X_test_neural, selected_cols
    """
    # Clean and dedupe
    neural_cols = list(dict.fromkeys([c for c in neural_cols if isinstance(c, str) and c.strip()]))
    
    # Filter to columns that exist in X_train
    train_cols = set(X_train.columns)
    neural_cols = [c for c in neural_cols if c in train_cols]
    
    if len(neural_cols) == 0:
        raise ValueError(f"[ERROR] neural_cols_{n_features} became empty after filtering to X_train.columns.")
    
    # Verify columns exist in valid/test splits
    valid_cols = set(X_valid.columns)
    test_cols = set(X_test.columns)
    
    missing_valid = sorted(set(neural_cols) - valid_cols)
    missing_test = sorted(set(neural_cols) - test_cols)
    
    def _report_missing(tag, miss):
        if miss:
            print(f"[ERROR] Missing in {tag}: {len(miss)} columns (showing up to 20): {miss[:20]}")
            return True
        return False
    
    err = False
    err |= _report_missing(f"X_valid (neural_{n_features})", missing_valid)
    err |= _report_missing(f"X_test (neural_{n_features})", missing_test)
    
    if err:
        raise KeyError("[ERROR] Inconsistent columns across splits for neural feature sets.")
    
    # Build neural feature matrices
    X_train_neural = X_train.loc[:, neural_cols].copy()
    X_valid_neural = X_valid.loc[:, neural_cols].copy()
    X_test_neural = X_test.loc[:, neural_cols].copy()
    
    print(f"\n[OK] Neural feature matrices created ({n_features}):")
    print(f"  - TRAIN={X_train_neural.shape} | VALID={X_valid_neural.shape} | TEST={X_test_neural.shape}")
    
    # Save resolved neural feature list
    if fs_output_dir:
        fs_output_dir = ensure_dir(Path(fs_output_dir))
        p_res = fs_output_dir / f"neural_feature_cols_{n_features}_bygroup_resolved.pkl"
        save_pickle(neural_cols, p_res)
        print(f"[INFO] Saved {p_res}")
    
    # Save Neural splits to data/processed
    if output_dir:
        output_dir = ensure_dir(Path(output_dir))
        X_train_neural.to_pickle(output_dir / f"X_train_neural_{n_features}.pkl")
        X_valid_neural.to_pickle(output_dir / f"X_valid_neural_{n_features}.pkl")
        X_test_neural.to_pickle(output_dir / f"X_test_neural_{n_features}.pkl")
        print(f"[INFO] Saved X_train/valid/test_neural_{n_features}.pkl to {output_dir}")
    
    return {
        "X_train_neural": X_train_neural,
        "X_valid_neural": X_valid_neural,
        "X_test_neural": X_test_neural,
        "selected_cols": neural_cols,
    }


def prepare_all_neural_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    config: Dict = None,
    output_dir: Path = None,
    fs_input_dir: Path = None,
    fs_output_dir: Path = None,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Full pipeline for neural feature preparation (BOTH 40 and 80).
    Matches notebook Cell 52 (BLOCK 22) exactly.
    
    If pkl files exist in fs_input_dir and force_recompute=False, loads them.
    Otherwise computes MI-based selection.
    
    Saves to fs_output_dir:
        - neural_feature_cols_40_bygroup_resolved.pkl
        - neural_feature_cols_80_bygroup_resolved.pkl
    
    Saves to output_dir:
        - X_train_neural_40.pkl, X_valid_neural_40.pkl, X_test_neural_40.pkl
        - X_train_neural_80.pkl, X_valid_neural_80.pkl, X_test_neural_80.pkl
    
    Args:
        X_train, X_valid, X_test: Feature DataFrames
        y_train: Training target Series
        config: Configuration dict
        output_dir: Directory for X_neural files (processed dir)
        fs_input_dir: Directory to load feature selection files from
        fs_output_dir: Directory to save resolved feature selection files
        force_recompute: If True, recompute MI even if pkl exists
    
    Returns:
        Dict with neural_40, neural_80 results
    """
    neural_cols_40 = None
    neural_cols_80 = None
    
    # Try to load pre-computed feature lists
    if fs_input_dir and not force_recompute:
        fs_input_dir = Path(fs_input_dir)
        pkl_40 = fs_input_dir / "neural_feature_cols_40_bygroup.pkl"
        pkl_80 = fs_input_dir / "neural_feature_cols_80_bygroup.pkl"
        
        if pkl_40.exists() and pkl_80.exists():
            try:
                neural_cols_40 = load_pickle(pkl_40)
                neural_cols_80 = load_pickle(pkl_80)
                print(f"[INFO] Loaded neural_40: {len(neural_cols_40)} features from {pkl_40}")
                print(f"[INFO] Loaded neural_80: {len(neural_cols_80)} features from {pkl_80}")
            except Exception as e:
                print(f"[WARN] Failed to load pkl files: {e}")
                neural_cols_40 = None
                neural_cols_80 = None
    
    # Compute MI-based selection if not loaded
    if neural_cols_40 is None or neural_cols_80 is None:
        print("[INFO] Computing MI-based feature selection...")
        result = select_neural_features_mi(
            X_train, y_train, config,
            output_dir=fs_input_dir or fs_output_dir
        )
        neural_cols_40 = result["neural_40"]
        neural_cols_80 = result["neural_80"]
    
    # Clean and dedupe
    neural_cols_40 = list(dict.fromkeys([c for c in neural_cols_40 if isinstance(c, str) and c.strip()]))
    neural_cols_80 = list(dict.fromkeys([c for c in neural_cols_80 if isinstance(c, str) and c.strip()]))
    
    # Filter to columns that exist in X_train
    train_cols = set(X_train.columns)
    neural_cols_40 = [c for c in neural_cols_40 if c in train_cols]
    neural_cols_80 = [c for c in neural_cols_80 if c in train_cols]
    
    if len(neural_cols_40) == 0:
        raise ValueError("[ERROR] neural_cols_40 became empty after filtering to X_train.columns.")
    if len(neural_cols_80) == 0:
        raise ValueError("[ERROR] neural_cols_80 became empty after filtering to X_train.columns.")
    
    # Verify columns exist in valid/test splits
    valid_cols = set(X_valid.columns)
    test_cols = set(X_test.columns)
    
    missing_valid_40 = sorted(set(neural_cols_40) - valid_cols)
    missing_test_40 = sorted(set(neural_cols_40) - test_cols)
    missing_valid_80 = sorted(set(neural_cols_80) - valid_cols)
    missing_test_80 = sorted(set(neural_cols_80) - test_cols)
    
    def _report_missing(tag, miss):
        if miss:
            print(f"[ERROR] Missing in {tag}: {len(miss)} columns (showing up to 20): {miss[:20]}")
            return True
        return False
    
    err = False
    err |= _report_missing("X_valid (neural_40)", missing_valid_40)
    err |= _report_missing("X_test (neural_40)", missing_test_40)
    err |= _report_missing("X_valid (neural_80)", missing_valid_80)
    err |= _report_missing("X_test (neural_80)", missing_test_80)
    
    if err:
        raise KeyError("[ERROR] Inconsistent columns across splits for neural feature sets.")
    
    # Build neural feature matrices
    X_train_neural_40 = X_train.loc[:, neural_cols_40].copy()
    X_valid_neural_40 = X_valid.loc[:, neural_cols_40].copy()
    X_test_neural_40 = X_test.loc[:, neural_cols_40].copy()
    
    X_train_neural_80 = X_train.loc[:, neural_cols_80].copy()
    X_valid_neural_80 = X_valid.loc[:, neural_cols_80].copy()
    X_test_neural_80 = X_test.loc[:, neural_cols_80].copy()
    
    print("\n[OK] Neural feature matrices created:")
    print(f"  - 40: TRAIN={X_train_neural_40.shape} | VALID={X_valid_neural_40.shape} | TEST={X_test_neural_40.shape}")
    print(f"  - 80: TRAIN={X_train_neural_80.shape} | VALID={X_valid_neural_80.shape} | TEST={X_test_neural_80.shape}")
    
    # Save resolved neural feature lists to fs_output_dir
    if fs_output_dir:
        fs_output_dir = ensure_dir(Path(fs_output_dir))
        p40_res = fs_output_dir / "neural_feature_cols_40_bygroup_resolved.pkl"
        p80_res = fs_output_dir / "neural_feature_cols_80_bygroup_resolved.pkl"
        save_pickle(neural_cols_40, p40_res)
        save_pickle(neural_cols_80, p80_res)
        print(f"[INFO] Saved {p40_res}")
        print(f"[INFO] Saved {p80_res}")
    
    # Save Neural splits to output_dir (data/processed)
    if output_dir:
        output_dir = ensure_dir(Path(output_dir))
        
        # Neural 40
        X_train_neural_40.to_pickle(output_dir / "X_train_neural_40.pkl")
        X_valid_neural_40.to_pickle(output_dir / "X_valid_neural_40.pkl")
        X_test_neural_40.to_pickle(output_dir / "X_test_neural_40.pkl")
        
        # Neural 80
        X_train_neural_80.to_pickle(output_dir / "X_train_neural_80.pkl")
        X_valid_neural_80.to_pickle(output_dir / "X_valid_neural_80.pkl")
        X_test_neural_80.to_pickle(output_dir / "X_test_neural_80.pkl")
        
        print(f"\n[OK] Saved Neural splits to: {output_dir}")
        print("  - X_train/valid/test_neural_40.pkl")
        print("  - X_train/valid/test_neural_80.pkl")
    
    return {
        "neural_40": {
            "X_train": X_train_neural_40,
            "X_valid": X_valid_neural_40,
            "X_test": X_test_neural_40,
            "cols": neural_cols_40,
        },
        "neural_80": {
            "X_train": X_train_neural_80,
            "X_valid": X_valid_neural_80,
            "X_test": X_test_neural_80,
            "cols": neural_cols_80,
        },
    }
