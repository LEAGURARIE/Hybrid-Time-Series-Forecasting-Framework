"""
Utility functions for Google Stock ML project.
File I/O, metrics, and general helpers.
"""

import json
import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


EPS = 1e-12


# ==============================================================================
# FILE I/O
# ==============================================================================

def ensure_dir(p: Path) -> Path:
    """Create directory if not exists, return path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_text(text: str, path: Path) -> Path:
    """Save text to file."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def save_json(obj: Any, path: Path, indent: int = 2) -> Path:
    """Save object as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)
    return path


def load_json(path: Path) -> Any:
    """Load object from JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> Path:
    """Save object as pickle."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: Path) -> Any:
    """Load object from pickle."""
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def copy_file(src: Path, dst: Path) -> Path:
    """Copy file with metadata."""
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def load_with_fallback(
    filename: str,
    primary_dir: Path,
    fallback_dir: Path,
    use_pandas: bool = False
) -> Any:
    """
    Load file with fallback directory.
    """
    paths_to_try = [
        (Path(primary_dir) / filename, "primary"),
        (Path(fallback_dir) / filename, "fallback"),
    ]
    
    for path, source_name in paths_to_try:
        if path.exists():
            print(f"  [LOAD] {filename} <- {source_name}")
            if use_pandas:
                return pd.read_pickle(path)
            return load_pickle(path)
    
    tried = "\n  ".join([f"{src}: {p}" for p, src in paths_to_try])
    raise FileNotFoundError(f"File not found: {filename}\nTried:\n  {tried}")


# ==============================================================================
# FEATURE LIST & MISSING SUMMARY
# ==============================================================================

def save_feature_list(
    feature_cols: List[str],
    output_dir: Path,
    prefix: str = "feature_list_all_columns"
) -> Dict[str, Path]:
    """
    Save feature list in multiple formats (.txt, .pkl, .csv).
    """
    output_dir = ensure_dir(Path(output_dir))
    paths = {}
    
    # TXT
    txt_path = output_dir / f"{prefix}.txt"
    save_text("\n".join(feature_cols), txt_path)
    paths["txt"] = txt_path
    
    # PKL
    pkl_path = output_dir / f"{prefix}.pkl"
    save_pickle(feature_cols, pkl_path)
    paths["pkl"] = pkl_path
    
    # CSV
    csv_path = output_dir / f"{prefix}.csv"
    pd.DataFrame({"feature": feature_cols}).to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    
    print(f"[INFO] Saved feature list ({len(feature_cols)} features) to {output_dir}")
    return paths


def save_missing_summary(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "missing_summary_all_columns.csv"
) -> Path:
    """
    Save missing values summary.
    """
    output_dir = ensure_dir(Path(output_dir))
    
    missing = df.isnull().sum()
    total = len(df)
    
    summary = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": (missing.values / total * 100).round(2),
        "dtype": [str(df[c].dtype) for c in missing.index]
    })
    summary = summary.sort_values("missing_count", ascending=False)
    
    path = output_dir / filename
    summary.to_csv(path, index=False)
    print(f"[INFO] Saved missing summary to {path}")
    return path


# ==============================================================================
# METRICS
# ==============================================================================

def _to_np(x: Union[np.ndarray, pd.Series, pd.DataFrame, list]) -> np.ndarray:
    """Convert to numpy array (flattened)."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values.flatten()
    if isinstance(x, list):
        return np.array(x).flatten()
    return np.asarray(x).flatten()


def w_rmse(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """Weighted Root Mean Squared Error."""
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    
    if weights is None:
        weights = np.ones_like(y_true)
    else:
        weights = _to_np(weights)
    
    sq_err = (y_true - y_pred) ** 2
    return float(np.sqrt(np.sum(weights * sq_err) / (np.sum(weights) + EPS)))


def w_mae(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """Weighted Mean Absolute Error."""
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    
    if weights is None:
        weights = np.ones_like(y_true)
    else:
        weights = _to_np(weights)
    
    abs_err = np.abs(y_true - y_pred)
    return float(np.sum(weights * abs_err) / (np.sum(weights) + EPS))


def directional_accuracy(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """Directional accuracy (sign match rate)."""
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """Compute all standard metrics."""
    return {
        f"{prefix}wrmse": w_rmse(y_true, y_pred, weights),
        f"{prefix}wmae": w_mae(y_true, y_pred, weights),
        f"{prefix}diracc": directional_accuracy(y_true, y_pred),
    }


# ==============================================================================
# BASELINE PREDICTIONS
# ==============================================================================

def baseline_zero(y: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Zero baseline: predict 0 for all values.
    
    Args:
        y: Target values (used only for length)
    
    Returns:
        Array of zeros with same length as y
    """
    return np.zeros(len(y), dtype=float)


def baseline_naive(
    y: Union[np.ndarray, pd.Series], 
    fill_first: float = 0.0
) -> np.ndarray:
    """
    Naive (last-value) baseline: predict y[t] = y[t-1].
    
    This is the "persistence forecast" - predicting that tomorrow's
    value will equal today's value.
    
    Args:
        y: Target values
        fill_first: Value to use for first prediction (no previous value available)
    
    Returns:
        Array of naive predictions (shifted y values)
    """
    y = _to_np(y)
    pred = np.roll(y, 1)
    pred[0] = fill_first
    return pred


def compute_baseline_metrics(
    y_true: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    baselines: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for multiple baseline methods.
    
    Args:
        y_true: Actual target values
        weights: Sample weights (optional)
        baselines: List of baseline names to compute (default: ["zero", "naive"])
    
    Returns:
        Dict with baseline metrics: {baseline_name: {wrmse, wmae, diracc}}
    """
    y_true = _to_np(y_true)
    
    if weights is None:
        weights = np.ones(len(y_true))
    weights = _to_np(weights)
    
    if baselines is None:
        baselines = ["zero", "naive"]
    
    results = {}
    
    for baseline_name in baselines:
        if baseline_name == "zero":
            pred = baseline_zero(y_true)
        elif baseline_name == "naive":
            pred = baseline_naive(y_true)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        
        results[baseline_name] = {
            "wrmse": w_rmse(y_true, pred, weights),
            "wmae": w_mae(y_true, pred, weights),
            "diracc": directional_accuracy(y_true, pred),
        }
    
    return results


# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    metric_fn: callable = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a single metric.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        weights: Sample weights (optional)
        metric_fn: Function(y_true, y_pred, weights) -> float
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility
    
    Returns:
        Dict with: point_estimate, ci_lower, ci_upper, std
    """
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    n = len(y_true)
    
    if weights is None:
        weights = np.ones(n)
    weights = _to_np(weights)
    
    rng = np.random.RandomState(random_state)
    
    # Point estimate
    point_estimate = metric_fn(y_true, y_pred, weights)
    
    # Bootstrap samples
    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        val = metric_fn(y_true[idx], y_pred[idx], weights[idx])
        bootstrap_values.append(val)
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(np.std(bootstrap_values)),
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for all standard metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        weights: Sample weights (optional)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95)
        random_state: Random seed
    
    Returns:
        Dict with CI for each metric: {metric_name: {point_estimate, ci_lower, ci_upper, std}}
    """
    # Wrapper functions that accept weights
    def wrmse_fn(yt, yp, w):
        return w_rmse(yt, yp, w)
    
    def wmae_fn(yt, yp, w):
        return w_mae(yt, yp, w)
    
    def diracc_fn(yt, yp, _w):
        return directional_accuracy(yt, yp)  # DirAcc doesn't use weights
    
    metrics = {
        "wrmse": wrmse_fn,
        "wmae": wmae_fn,
        "diracc": diracc_fn,
    }
    
    results = {}
    for name, fn in metrics.items():
        results[name] = bootstrap_metric(
            y_true, y_pred, weights, fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state
        )
    
    return results


def format_ci(ci_result: Dict[str, float], decimals: int = 6) -> str:
    """Format confidence interval as string: point [lower, upper]."""
    return (
        f"{ci_result['point_estimate']:.{decimals}f} "
        f"[{ci_result['ci_lower']:.{decimals}f}, {ci_result['ci_upper']:.{decimals}f}]"
    )


def make_weighted_rmse_scorer(weights: np.ndarray):
    """
    Create a sklearn-compatible scorer for weighted RMSE.
    Used in permutation importance and HPO.
    
    Args:
        weights: Sample weights for validation set
    
    Returns:
        Scorer function for sklearn
    """
    def neg_weighted_rmse_scorer(estimator, X, y):
        pred = estimator.predict(X)
        return -w_rmse(y, pred, weights)
    return neg_weighted_rmse_scorer


# ==============================================================================
# SAMPLE WEIGHTS
# ==============================================================================

def compute_sample_weights(
    y: Union[np.ndarray, pd.Series],
    c: float = 1.0,
    max_w: float = 4.0,
    time_weight: bool = True
) -> np.ndarray:
    """
    Compute sample weights: time * |y| normalized to mean=1.
    """
    y = _to_np(y)
    n = len(y)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Time weight
    w_time = np.linspace(1.0, 2.0, n, dtype=float) if time_weight else np.ones(n)
    
    # Y-based weight
    abs_y = np.abs(y)
    med_abs = float(np.median(abs_y))
    ratio = abs_y / (med_abs + EPS)
    w_y = 1.0 + c * np.sqrt(ratio)
    w_y = np.clip(w_y, 1.0, max_w)
    
    # Combined & normalized
    weights = w_time * w_y
    weights = weights / (weights.mean() + EPS)
    
    return weights


def compute_weight_diagnostics(
    weights: np.ndarray,
    y: Union[np.ndarray, pd.Series],
    split_name: str,
    c: float,
    max_w: float
) -> Dict[str, Any]:
    """Compute diagnostics for sample weights."""
    y = _to_np(y)
    n = len(weights)
    
    if n == 0:
        return {"split": split_name, "n": 0}
    
    abs_y = np.abs(y)
    cap_rate = float(np.mean(weights >= max_w * 0.99))
    
    return {
        "split": split_name,
        "n": n,
        "c": c,
        "max_w": max_w,
        "y_med_abs": float(np.median(abs_y)),
        "w_min": float(weights.min()),
        "w_max": float(weights.max()),
        "w_mean": float(weights.mean()),
        "cap_rate": cap_rate,
    }


# ==============================================================================
# SPLITS SAVING
# ==============================================================================

def save_splits(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    w_test: np.ndarray,
    output_dir: Path,
    prefix: str = "xgb"
) -> Dict[str, Path]:
    """
    Save train/valid/test splits to disk.
    """
    output_dir = ensure_dir(Path(output_dir))
    paths = {}
    
    # Features (X)
    X_train.to_pickle(output_dir / f"X_train_{prefix}.pkl")
    X_valid.to_pickle(output_dir / f"X_valid_{prefix}.pkl")
    X_test.to_pickle(output_dir / f"X_test_{prefix}.pkl")
    
    # Target (y)
    save_pickle(y_train, output_dir / "y_train.pkl")
    save_pickle(y_valid, output_dir / "y_valid.pkl")
    save_pickle(y_test, output_dir / "y_test.pkl")
    
    # Weights
    save_pickle(w_train, output_dir / "weights_train.pkl")
    save_pickle(w_valid, output_dir / "weights_valid.pkl")
    save_pickle(w_test, output_dir / "weights_test.pkl")
    
    print(f"[INFO] Saved splits to {output_dir}")
    print(f"  - X_train/valid/test_{prefix}.pkl")
    print(f"  - y_train/valid/test.pkl")
    print(f"  - weights_train/valid/test.pkl")
    
    return paths


# ==============================================================================
# TARGET METADATA
# ==============================================================================

def save_target_meta(
    target_col: str,
    source_col: str,
    y: pd.Series,
    output_dir: Path
) -> Path:
    """Save target variable metadata."""
    output_dir = ensure_dir(Path(output_dir))
    
    meta = {
        "target_col": target_col,
        "source_col": source_col,
        "n_samples": len(y),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_median": float(y.median()),
        "y_missing": int(y.isna().sum()),
    }
    
    path = output_dir / "target_meta.json"
    save_json(meta, path)
    print(f"[INFO] Saved target metadata to {path}")
    return path


# ==============================================================================
# RUN OUTPUTS
# ==============================================================================

def save_run_outputs(
    metrics: Dict[str, Any],
    outputs_dir: Path,
    models_dir: Path,
    predictions_valid: Optional[pd.DataFrame] = None,
    predictions_test: Optional[pd.DataFrame] = None,
    extra_artifacts: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    model_filename: str = "model.json",
) -> Dict[str, Path]:
    """Save standard run outputs (metrics, predictions, model)."""
    outputs_dir = ensure_dir(Path(outputs_dir))
    models_dir = ensure_dir(Path(models_dir))
    paths = {}
    
    # Metrics
    paths["metrics_json"] = save_json(metrics, outputs_dir / "metrics.json")
    paths["metrics_txt"] = save_text(
        "\n".join([f"{k}: {v}" for k, v in metrics.items()]),
        outputs_dir / "metrics.txt",
    )
    
    # Predictions
    if predictions_valid is not None:
        path = outputs_dir / "predictions_valid.csv"
        predictions_valid.to_csv(path, index=True)
        paths["predictions_valid"] = path
    
    if predictions_test is not None:
        path = outputs_dir / "predictions_test.csv"
        predictions_test.to_csv(path, index=True)
        paths["predictions_test"] = path
    
    # Extra artifacts
    if extra_artifacts:
        for name, obj in extra_artifacts.items():
            path = save_pickle(obj, outputs_dir / f"{name}.pkl")
            paths[name] = path
    
    # Model
    if model is not None:
        path = models_dir / model_filename
        if hasattr(model, "save_model"):
            model.save_model(str(path))
        else:
            path = models_dir / (Path(model_filename).stem + ".pkl")
            save_pickle(model, path)
        paths["model"] = path
    
    return paths


def compute_model_comparison_ci(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_model1: Union[np.ndarray, pd.Series],
    y_pred_model2: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap CI for the DIFFERENCE between two models.
    
    If CI excludes 0, the difference is statistically significant.
    
    Args:
        y_true: Actual values
        y_pred_model1: Predictions from model 1
        y_pred_model2: Predictions from model 2
        weights: Sample weights (optional)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_state: Random seed
    
    Returns:
        Dict with difference metrics: {metric_name: {diff, ci_lower, ci_upper, significant}}
    """
    y_true = _to_np(y_true)
    y_pred1 = _to_np(y_pred_model1)
    y_pred2 = _to_np(y_pred_model2)
    n = len(y_true)
    
    if weights is not None:
        weights = _to_np(weights)
    
    rng = np.random.RandomState(random_state)
    
    # Point estimates
    if weights is not None:
        wrmse1 = w_rmse(y_true, y_pred1, weights)
        wrmse2 = w_rmse(y_true, y_pred2, weights)
    else:
        wrmse1 = w_rmse(y_true, y_pred1)
        wrmse2 = w_rmse(y_true, y_pred2)
    
    diracc1 = directional_accuracy(y_true, y_pred1)
    diracc2 = directional_accuracy(y_true, y_pred2)
    
    # Bootstrap differences
    wrmse_diffs = []
    diracc_diffs = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp1 = y_pred1[idx]
        yp2 = y_pred2[idx]
        
        if weights is not None:
            w = weights[idx]
            diff_wrmse = w_rmse(yt, yp1, w) - w_rmse(yt, yp2, w)
        else:
            diff_wrmse = w_rmse(yt, yp1) - w_rmse(yt, yp2)
        
        diff_diracc = directional_accuracy(yt, yp1) - directional_accuracy(yt, yp2)
        
        wrmse_diffs.append(diff_wrmse)
        diracc_diffs.append(diff_diracc)
    
    wrmse_diffs = np.array(wrmse_diffs)
    diracc_diffs = np.array(diracc_diffs)
    
    alpha = 1 - confidence_level
    
    def make_result(diffs, point_diff):
        ci_lower = np.percentile(diffs, 100 * alpha / 2)
        ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
        # Significant if CI excludes 0
        significant = (ci_lower > 0) or (ci_upper < 0)
        return {
            "diff": float(point_diff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(diffs)),
            "significant": significant,
        }
    
    return {
        "wrmse_diff": make_result(wrmse_diffs, wrmse1 - wrmse2),
        "diracc_diff": make_result(diracc_diffs, diracc1 - diracc2),
    }
