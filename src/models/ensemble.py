"""
Ensemble methods for Google Stock ML project.
Combines predictions from multiple models (XGBoost, LightGBM, LSTM, GRU, Hybrid).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from ..utils import (
    w_rmse, w_mae, directional_accuracy, compute_all_metrics,
    save_pickle, save_json, load_pickle, _to_np, EPS
)


# ==============================================================================
# PREDICTION LOADING
# ==============================================================================

def load_tomorrow_predictions(
    models_dir: Path,
    models: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Load tomorrow predictions from all available models.
    
    Args:
        models_dir: Directory containing model outputs
        models: List of models to include, or None for all available
    
    Returns:
        Dict mapping model_name to tomorrow prediction (log return)
    """
    models_dir = Path(models_dir)
    pred_base = models_dir.parent / "predictions"
    
    # Model tomorrow.csv paths
    # Build paths dynamically - check for all possible feature sets
    all_feature_sets = ["xgb_selected", "neural_40", "neural_80"]
    
    tomorrow_paths = {
        "xgb": [pred_base / "xgb" / "tomorrow.csv"],
        "lgb": [pred_base / "lgb" / "tomorrow.csv"],
    }
    
    # Add neural model paths for all feature sets
    for model_key in ["lstm", "gru", "hybrid_seq", "hybrid_par"]:
        tomorrow_paths[model_key] = [
            pred_base / f"{model_key}_{fs}" / "tomorrow.csv"
            for fs in all_feature_sets
        ]
    
    if models is None:
        models = list(tomorrow_paths.keys())
    
    tomorrow_preds = {}
    
    for model_name in models:
        if model_name not in tomorrow_paths:
            continue
        
        # Find first existing file
        for path in tomorrow_paths[model_name]:
            if path.exists():
                df = pd.read_csv(path)
                if "pred_logret" in df.columns and len(df) > 0:
                    tomorrow_preds[model_name] = float(df["pred_logret"].iloc[0])
                    break
    
    return tomorrow_preds


def load_model_predictions(
    models_dir: Path,
    split: str = "test",
    models: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load predictions from all available models.
    
    Args:
        models_dir: Directory containing model outputs
        split: Which split to load ("valid" or "test")
        models: List of models to include, or None for all available
    
    Returns:
        Tuple of (predictions_df, actual_values)
        predictions_df has columns for each model
    """
    models_dir = Path(models_dir)
    
    # Predictions are in predictions/{model}/ or predictions/{model}_{feature_set}/
    # models_dir is typically runs/{RUN_ID}/models/
    # predictions are in runs/{RUN_ID}/predictions/
    pred_base = models_dir.parent / "predictions"
    
    # Build paths dynamically - check for all possible feature sets
    all_feature_sets = ["xgb_selected", "neural_40", "neural_80"]
    
    # Model prediction file patterns (model_key, possible_folders)
    model_paths = {
        "xgb": [pred_base / "xgb" / f"predictions_{split}.csv"],
        "lgb": [pred_base / "lgb" / f"predictions_{split}.csv"],
    }
    
    # Add neural model paths for all feature sets
    for model_key in ["lstm", "gru", "hybrid_seq", "hybrid_par"]:
        model_paths[model_key] = [
            pred_base / f"{model_key}_{fs}" / f"predictions_{split}.csv"
            for fs in all_feature_sets
        ]
    
    if models is None:
        models = list(model_paths.keys())
    
    predictions = {}
    actual = None
    weights = None
    
    for model_name in models:
        if model_name not in model_paths:
            print(f"[WARN] Unknown model: {model_name}")
            continue
        
        # Find first existing file
        file_path = None
        for path in model_paths[model_name]:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            print(f"[WARN] Predictions not found for {model_name}")
            continue
        
        df = pd.read_csv(file_path)
        
        # Handle different column names
        pred_col = None
        actual_col = None
        
        for col in ["predicted", "prediction", "y_pred_model", "y_pred"]:
            if col in df.columns:
                pred_col = col
                break
        
        for col in ["actual", "y_true"]:
            if col in df.columns:
                actual_col = col
                break
        
        if pred_col is None:
            print(f"[WARN] No prediction column in {file_path}")
            continue
        
        predictions[model_name] = df[pred_col].values
        if actual is None and actual_col is not None:
            actual = df[actual_col].values
        
        # Load weights if available (take from first model that has them)
        if weights is None and "sample_weight" in df.columns:
            weights = df["sample_weight"].values
        
        print(f"[OK] Loaded {model_name}: {len(predictions[model_name])} predictions")
    
    if len(predictions) == 0:
        raise ValueError("No model predictions found!")
    
    # Align lengths (take minimum)
    min_len = min(len(p) for p in predictions.values())
    
    pred_df = pd.DataFrame({
        name: pred[:min_len] for name, pred in predictions.items()
    })
    
    if actual is not None:
        actual = pd.Series(actual[:min_len], name="actual")
    
    if weights is not None:
        weights = np.array(weights[:min_len])
    
    return pred_df, actual, weights


def load_model_metrics(
    models_dir: Path,
    models: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Load metrics from all available models (both valid and test).
    
    Args:
        models_dir: Directory containing model outputs
        models: List of models to include, or None for all available
    
    Returns:
        Dict mapping model name to metrics dict with:
            - valid_wrmse, valid_diracc (for weight calculation)
            - test_wrmse, test_diracc (for final evaluation)
    """
    models_dir = Path(models_dir)
    
    metric_files = {
        "xgb": "final_metrics.csv",
        "lgb": "final_metrics_lgb.csv",
        "lstm": "lstm_summary.csv",
        "gru": "gru_summary.csv",
        "hybrid_seq": "hybrid_seq_summary.csv",
        "hybrid_par": "hybrid_par_summary.csv",
    }
    
    if models is None:
        models = list(metric_files.keys())
    
    all_metrics = {}
    
    for model_name in models:
        if model_name not in metric_files:
            continue
            
        file_path = models_dir / metric_files[model_name]
        
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        
        test_wrmse = None
        test_diracc = None
        valid_wrmse = None
        valid_diracc = None
        
        # --- Extract TEST metrics ---
        if "test_wrmse" in df.columns:
            test_wrmse = df["test_wrmse"].iloc[0]
        elif "wRMSE" in df.columns:
            # XGBoost/LightGBM format - find TEST row
            test_row = df[df.get("split", "") == "TEST"]
            if len(test_row) > 0:
                test_wrmse = test_row["wRMSE"].iloc[0]
            else:
                test_wrmse = df["wRMSE"].iloc[-1]  # Last row
        elif "model_test_wrmse" in df.columns:
            test_wrmse = df["model_test_wrmse"].iloc[0]
        
        if "test_diracc" in df.columns:
            test_diracc = df["test_diracc"].iloc[0]
        elif "DirAcc" in df.columns:
            test_row = df[df.get("split", "") == "TEST"]
            if len(test_row) > 0:
                test_diracc = test_row["DirAcc"].iloc[0]
            else:
                test_diracc = df["DirAcc"].iloc[-1]
        elif "model_test_diracc" in df.columns:
            test_diracc = df["model_test_diracc"].iloc[0]
        
        # --- Extract VALID metrics ---
        if "valid_wrmse" in df.columns:
            valid_wrmse = df["valid_wrmse"].iloc[0]
        elif "wRMSE" in df.columns:
            # XGBoost/LightGBM format - find VALID row
            valid_row = df[df.get("split", "") == "VALID"]
            if len(valid_row) > 0:
                valid_wrmse = valid_row["wRMSE"].iloc[0]
        elif "model_valid_wrmse" in df.columns:
            valid_wrmse = df["model_valid_wrmse"].iloc[0]
        
        if "valid_diracc" in df.columns:
            valid_diracc = df["valid_diracc"].iloc[0]
        elif "DirAcc" in df.columns:
            valid_row = df[df.get("split", "") == "VALID"]
            if len(valid_row) > 0:
                valid_diracc = valid_row["DirAcc"].iloc[0]
        elif "model_valid_diracc" in df.columns:
            valid_diracc = df["model_valid_diracc"].iloc[0]
        
        # Fallback: if no valid metrics, use test (with warning)
        if valid_wrmse is None and test_wrmse is not None:
            valid_wrmse = test_wrmse
            print(f"[WARN] {model_name}: No valid_wrmse found, using test_wrmse for weights")
        
        if test_wrmse is None:
            print(f"[WARN] No wRMSE found in {file_path}")
            continue
        
        all_metrics[model_name] = {
            "valid_wrmse": float(valid_wrmse) if valid_wrmse is not None else float(test_wrmse),
            "valid_diracc": float(valid_diracc) if valid_diracc is not None else None,
            "test_wrmse": float(test_wrmse),
            "test_diracc": float(test_diracc) if test_diracc is not None else None
        }
        
        valid_str = f"valid_wRMSE={valid_wrmse:.6f}" if valid_wrmse else "valid_wRMSE=N/A"
        test_str = f"test_wRMSE={test_wrmse:.6f}"
        diracc_str = f"test_DirAcc={test_diracc:.4f}" if test_diracc is not None else "test_DirAcc=N/A"
        print(f"[OK] Loaded {model_name}: {valid_str} | {test_str} | {diracc_str}")
    
    return all_metrics


def filter_models_by_metrics(
    metrics: Dict[str, Dict[str, float]],
    filter_config: Optional[Dict] = None
) -> Dict[str, Dict[str, float]]:
    """
    Filter models based on performance thresholds.
    
    Args:
        metrics: Dict from load_model_metrics()
        filter_config: Dict with optional keys:
            - min_diracc: Minimum valid_diracc threshold (e.g., 0.52)
            - max_wrmse: Maximum valid_wrmse threshold (e.g., 0.025)
            - top_n: Keep only top N models by valid_wrmse
            - use_test: If True, filter by test metrics instead of valid (default: False)
    
    Returns:
        Filtered metrics dict
    
    Note:
        By default, filtering uses valid metrics to avoid data leakage.
    """
    if filter_config is None or len(filter_config) == 0:
        return metrics
    
    filtered = dict(metrics)
    
    min_diracc = filter_config.get("min_diracc")
    max_wrmse = filter_config.get("max_wrmse")
    top_n = filter_config.get("top_n")
    use_test = filter_config.get("use_test", False)  # Default: use valid metrics
    
    # Choose which metrics to use
    wrmse_key = "test_wrmse" if use_test else "valid_wrmse"
    diracc_key = "test_diracc" if use_test else "valid_diracc"
    metric_source = "TEST" if use_test else "VALID"
    
    # Filter by min_diracc
    if min_diracc is not None:
        before_count = len(filtered)
        filtered = {
            name: m for name, m in filtered.items()
            if m.get(diracc_key) is not None and m[diracc_key] >= min_diracc
        }
        removed = before_count - len(filtered)
        if removed > 0:
            print(f"[FILTER] Removed {removed} models with {metric_source} DirAcc < {min_diracc}")
    
    # Filter by max_wrmse
    if max_wrmse is not None:
        before_count = len(filtered)
        filtered = {
            name: m for name, m in filtered.items()
            if m[wrmse_key] <= max_wrmse
        }
        removed = before_count - len(filtered)
        if removed > 0:
            print(f"[FILTER] Removed {removed} models with {metric_source} wRMSE > {max_wrmse}")
    
    # Keep only top_n by wRMSE (lower is better)
    if top_n is not None and len(filtered) > top_n:
        sorted_models = sorted(filtered.items(), key=lambda x: x[1][wrmse_key])
        kept = [name for name, _ in sorted_models[:top_n]]
        removed_models = [name for name, _ in sorted_models[top_n:]]
        filtered = {name: filtered[name] for name in kept}
        print(f"[FILTER] Kept top {top_n} models by {metric_source} wRMSE, removed: {removed_models}")
    
    if len(filtered) == 0:
        print("[WARN] All models filtered out! Using original metrics.")
        return metrics
    
    print(f"[FILTER] Final models: {list(filtered.keys())}")
    return filtered


# ==============================================================================
# ENSEMBLE METHODS
# ==============================================================================

def simple_average(predictions: pd.DataFrame) -> np.ndarray:
    """
    Simple average of all model predictions.
    
    Args:
        predictions: DataFrame with model predictions as columns
    
    Returns:
        Averaged predictions
    """
    return predictions.mean(axis=1).values


def weighted_average(
    predictions: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    weight_method: str = "inverse_wrmse"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Weighted average of model predictions.
    
    Args:
        predictions: DataFrame with model predictions as columns
        weights: Manual weights dict, or None to compute automatically
        metrics: Model metrics for automatic weight computation
        weight_method: How to compute weights (uses valid_wrmse to avoid data leakage)
            - "inverse_wrmse": w = 1/valid_wrmse
            - "inverse_wrmse_squared": w = 1/valid_wrmse^2
    
    Returns:
        Tuple of (weighted predictions, weights used)
    
    Note:
        Weights are computed from valid_wrmse (not test) to avoid data leakage.
    """
    models = predictions.columns.tolist()
    
    if weights is not None:
        # Use provided weights
        final_weights = {m: weights.get(m, 0.0) for m in models}
    elif metrics is not None:
        # Compute weights from VALID metrics (avoid data leakage!)
        if weight_method == "inverse_wrmse":
            raw_weights = {m: 1.0 / (metrics[m]["valid_wrmse"] + EPS) 
                          for m in models if m in metrics}
        elif weight_method == "inverse_wrmse_squared":
            raw_weights = {m: 1.0 / ((metrics[m]["valid_wrmse"] ** 2) + EPS) 
                          for m in models if m in metrics}
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}")
        
        # Normalize
        total = sum(raw_weights.values())
        final_weights = {m: w / total for m, w in raw_weights.items()}
    else:
        # Equal weights
        final_weights = {m: 1.0 / len(models) for m in models}
    
    # Apply weights
    weighted_pred = np.zeros(len(predictions))
    for model, weight in final_weights.items():
        if model in predictions.columns:
            weighted_pred += weight * predictions[model].values
    
    return weighted_pred, final_weights


def stacking(
    train_predictions: pd.DataFrame,
    train_actual: np.ndarray,
    test_predictions: pd.DataFrame,
    train_weights: Optional[np.ndarray] = None,
    meta_model: str = "ridge",
    meta_params: Optional[Dict] = None
) -> Tuple[np.ndarray, Any, Dict[str, float]]:
    """
    Stacking ensemble with a meta-model.
    
    Args:
        train_predictions: Training set predictions from base models
        train_actual: Actual training values
        test_predictions: Test set predictions from base models
        train_weights: Sample weights for training
        meta_model: Type of meta-model ("ridge", "lasso", "elasticnet", "xgb", "lgb")
        meta_params: Parameters for meta-model
    
    Returns:
        Tuple of (test predictions, fitted meta-model, feature importances)
    """
    if meta_params is None:
        meta_params = {}
    
    X_train = train_predictions.values
    y_train = np.asarray(train_actual)
    X_test = test_predictions.values
    
    if meta_model == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=meta_params.get("alpha", 1.0))
        model.fit(X_train, y_train, sample_weight=train_weights)
        importances = dict(zip(train_predictions.columns, model.coef_))
        
    elif meta_model == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=meta_params.get("alpha", 0.01))
        model.fit(X_train, y_train, sample_weight=train_weights)
        importances = dict(zip(train_predictions.columns, model.coef_))
        
    elif meta_model == "elasticnet":
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(
            alpha=meta_params.get("alpha", 0.01),
            l1_ratio=meta_params.get("l1_ratio", 0.5)
        )
        model.fit(X_train, y_train, sample_weight=train_weights)
        importances = dict(zip(train_predictions.columns, model.coef_))
        
    elif meta_model == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=meta_params.get("n_estimators", 100),
            max_depth=meta_params.get("max_depth", 3),
            learning_rate=meta_params.get("learning_rate", 0.1),
            random_state=meta_params.get("random_state", 42),
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=train_weights)
        importances = dict(zip(train_predictions.columns, model.feature_importances_))
        
    elif meta_model == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=meta_params.get("n_estimators", 100),
            max_depth=meta_params.get("max_depth", 3),
            learning_rate=meta_params.get("learning_rate", 0.1),
            random_state=meta_params.get("random_state", 42),
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=train_weights)
        importances = dict(zip(train_predictions.columns, model.feature_importances_))
        
    else:
        raise ValueError(f"Unknown meta_model: {meta_model}")
    
    predictions = model.predict(X_test)
    
    return predictions, model, importances


def rank_average(predictions: pd.DataFrame) -> np.ndarray:
    """
    Rank-based averaging. Converts predictions to ranks, averages, then converts back.
    More robust to outliers and different prediction scales.
    
    Args:
        predictions: DataFrame with model predictions as columns
    
    Returns:
        Rank-averaged predictions
    """
    # Convert to ranks (0-1 scale)
    ranks = predictions.rank(pct=True)
    
    # Average ranks
    avg_rank = ranks.mean(axis=1)
    
    # Convert back to prediction scale using average of original distributions
    mean_pred = predictions.mean(axis=1).mean()
    std_pred = predictions.std(axis=1).mean()
    
    # Map ranks to predictions (assuming normal distribution)
    from scipy import stats
    z_scores = stats.norm.ppf(avg_rank.clip(0.001, 0.999))
    
    return mean_pred + std_pred * z_scores


# ==============================================================================
# MAIN ENSEMBLE FUNCTION
# ==============================================================================

def run_ensemble(
    models_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    valid_weights: Optional[np.ndarray] = None,
    test_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run ensemble combining multiple model predictions.
    
    Args:
        models_dir: Directory containing model outputs
        output_dir: Directory to save ensemble outputs
        config: Ensemble configuration
        valid_weights: Sample weights for validation set
        test_weights: Sample weights for test set
    
    Returns:
        Dict with ensemble results and metrics
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method = config.get("method", "weighted_average")
    models = config.get("models", None)  # None = all available
    weight_method = config.get("weight_method", "inverse_wrmse")
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE: {method}")
    print(f"{'='*60}")
    
    # Load predictions
    print("\n[INFO] Loading validation predictions...")
    valid_pred, valid_actual, loaded_valid_weights = load_model_predictions(models_dir, "valid", models)
    
    print("\n[INFO] Loading test predictions...")
    test_pred, test_actual, loaded_test_weights = load_model_predictions(models_dir, "test", models)
    
    # Use loaded weights if not provided
    if valid_weights is None:
        valid_weights = loaded_valid_weights
    if test_weights is None:
        test_weights = loaded_test_weights
    
    # Load metrics for weighting
    metrics = load_model_metrics(models_dir, models)
    
    # Apply filtering based on config
    filter_config = config.get("filter", {})
    if filter_config:
        print(f"\n[INFO] Applying model filter: {filter_config}")
        metrics = filter_models_by_metrics(metrics, filter_config)
        
        # Filter predictions to only include filtered models
        filtered_models = list(metrics.keys())
        valid_pred = valid_pred[[m for m in valid_pred.columns if m in filtered_models]]
        test_pred = test_pred[[m for m in test_pred.columns if m in filtered_models]]
    
    print(f"\n[INFO] Models included: {list(valid_pred.columns)}")
    
    # Run ensemble method
    if method == "simple_average":
        ensemble_valid = simple_average(valid_pred)
        ensemble_test = simple_average(test_pred)
        weights_used = {m: 1.0 / len(valid_pred.columns) for m in valid_pred.columns}
        meta_model = None
        
    elif method == "weighted_average":
        manual_weights = config.get("weights", None)
        if isinstance(manual_weights, str) and manual_weights == "auto":
            manual_weights = None
            
        ensemble_valid, weights_used = weighted_average(
            valid_pred, 
            weights=manual_weights,
            metrics=metrics,
            weight_method=weight_method
        )
        ensemble_test, _ = weighted_average(
            test_pred,
            weights=weights_used
        )
        meta_model = None
        
    elif method == "stacking":
        meta_model_type = config.get("meta_model", "ridge")
        meta_params = config.get("meta_params", {})
        
        ensemble_test, meta_model, weights_used = stacking(
            valid_pred, valid_actual,
            test_pred,
            train_weights=valid_weights,
            meta_model=meta_model_type,
            meta_params=meta_params
        )
        # Also predict on validation for metrics
        ensemble_valid = meta_model.predict(valid_pred.values)
        
    elif method == "rank_average":
        ensemble_valid = rank_average(valid_pred)
        ensemble_test = rank_average(test_pred)
        weights_used = {m: 1.0 / len(valid_pred.columns) for m in valid_pred.columns}
        meta_model = None
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Compute metrics
    if valid_weights is None:
        valid_weights = np.ones(len(valid_actual))
    if test_weights is None:
        test_weights = np.ones(len(test_actual))
    
    # Align weights length
    valid_weights = valid_weights[:len(valid_actual)]
    test_weights = test_weights[:len(test_actual)]
    
    results = {
        "model": f"Ensemble-{method}",
        "method": method,
        "weight_method": weight_method,
        "models": list(valid_pred.columns),
        "weights": weights_used,
        "valid_wrmse": w_rmse(valid_actual, ensemble_valid, valid_weights),
        "valid_wmae": w_mae(valid_actual, ensemble_valid, valid_weights),
        "valid_diracc": directional_accuracy(valid_actual, ensemble_valid),
        "test_wrmse": w_rmse(test_actual, ensemble_test, test_weights),
        "test_wmae": w_mae(test_actual, ensemble_test, test_weights),
        "test_diracc": directional_accuracy(test_actual, ensemble_test),
    }
    
    print(f"\n[RESULTS] Ensemble ({method}):")
    print(f"  Weights: {weights_used}")
    print(f"  Valid wRMSE: {results['valid_wrmse']:.6f} | wMAE: {results['valid_wmae']:.6f} | DirAcc: {results['valid_diracc']:.4f}")
    print(f"  Test  wRMSE: {results['test_wrmse']:.6f} | wMAE: {results['test_wmae']:.6f} | DirAcc: {results['test_diracc']:.4f}")
    
    # Compare with individual models
    print("\n[COMPARE] Individual models vs Ensemble:")
    print(f"  {'Model':<15} {'Test wRMSE':<12} {'Improvement':<12}")
    print(f"  {'-'*39}")
    for model_name, model_metrics in metrics.items():
        model_wrmse = model_metrics["test_wrmse"]
        improvement = (model_wrmse - results["test_wrmse"]) / model_wrmse * 100
        print(f"  {model_name:<15} {model_wrmse:<12.6f} {improvement:>+.2f}%")
    print(f"  {'ENSEMBLE':<15} {results['test_wrmse']:<12.6f} {'---':<12}")
    
    # Save outputs
    print(f"\n[INFO] Saving ensemble outputs to {output_dir}")
    
    # Predictions with weights
    valid_df = pd.DataFrame({
        "actual": valid_actual,
        "predicted": ensemble_valid
    })
    if valid_weights is not None:
        valid_df["sample_weight"] = valid_weights[:len(valid_actual)]
    valid_df.to_csv(output_dir / "ensemble_predictions_valid.csv", index=False)
    
    test_df = pd.DataFrame({
        "actual": test_actual,
        "predicted": ensemble_test
    })
    if test_weights is not None:
        test_df["sample_weight"] = test_weights[:len(test_actual)]
    test_df.to_csv(output_dir / "ensemble_predictions_test.csv", index=False)
    
    # Metrics
    pd.DataFrame([results]).to_csv(output_dir / "ensemble_metrics.csv", index=False)
    
    # Full results JSON
    save_json(results, output_dir / "ensemble_results.json")
    
    # Meta-model (if stacking)
    if meta_model is not None:
        save_pickle(meta_model, output_dir / "ensemble_meta_model.pkl")
    
    # Weights
    save_json(weights_used, output_dir / "ensemble_weights.json")
    
    # Tomorrow prediction (ensemble of individual model predictions)
    tomorrow_preds = load_tomorrow_predictions(models_dir, list(valid_pred.columns))
    
    if tomorrow_preds:
        print(f"\n[INFO] Tomorrow predictions from {len(tomorrow_preds)} models")
        
        # Calculate weighted average of tomorrow predictions
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_name, pred in tomorrow_preds.items():
            weight = weights_used.get(model_name, 0.0)
            if weight > 0:
                weighted_sum += weight * pred
                total_weight += weight
                print(f"  {model_name}: {pred:.6f} (weight: {weight:.4f})")
        
        if total_weight > 0:
            ensemble_tomorrow = weighted_sum / total_weight
            ensemble_tomorrow_pct = float(np.expm1(ensemble_tomorrow) * 100)
            
            print(f"\n[INFO] ENSEMBLE Tomorrow prediction: {ensemble_tomorrow:.6f} ({ensemble_tomorrow_pct:.4f}%)")
            
            # Save tomorrow prediction
            tomorrow_df = pd.DataFrame([{
                "method": method,
                "n_models": len(tomorrow_preds),
                "predicted_for": "next_trading_day",
                "pred_logret": ensemble_tomorrow,
                "pred_return_pct": ensemble_tomorrow_pct,
            }])
            tomorrow_df.to_csv(output_dir / "tomorrow.csv", index=False)
            
            # Add to results
            results["pred_tomorrow_logret"] = ensemble_tomorrow
            results["pred_tomorrow_pct"] = ensemble_tomorrow_pct
            
            # Update the JSON with tomorrow prediction
            save_json(results, output_dir / "ensemble_results.json")
            
            print(f"[OK] Saved: ensemble/tomorrow.csv")
    
    print(f"\n[OK] Ensemble complete!")
    
    return results


# ==============================================================================
# ENSEMBLE COMPARISON
# ==============================================================================

def compare_ensemble_methods(
    models_dir: Path,
    output_dir: Path,
    methods: Optional[List[str]] = None,
    models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare different ensemble methods.
    
    Args:
        models_dir: Directory containing model outputs
        output_dir: Directory to save comparison outputs
        methods: List of methods to compare, or None for all
        models: List of models to include, or None for all available
    
    Returns:
        DataFrame with comparison results
    """
    if methods is None:
        methods = ["simple_average", "weighted_average", "rank_average", "stacking"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for method in methods:
        config = {
            "method": method,
            "models": models,
            "weights": "auto",
            "weight_method": "inverse_wrmse",
            "meta_model": "ridge",
        }
        
        try:
            result = run_ensemble(
                models_dir=models_dir,
                output_dir=output_dir / method,
                config=config
            )
            result["method"] = method
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] {method}: {e}")
    
    if len(all_results) > 0:
        comparison_df = pd.DataFrame(all_results)
        comparison_df = comparison_df.sort_values("test_wrmse")
        comparison_df.to_csv(output_dir / "ensemble_comparison.csv", index=False)
        
        print("\n" + "="*60)
        print("ENSEMBLE COMPARISON (sorted by Test wRMSE)")
        print("="*60)
        print(comparison_df[["method", "test_wrmse", "test_diracc"]].to_string(index=False))
        
        return comparison_df
    
    return pd.DataFrame()
