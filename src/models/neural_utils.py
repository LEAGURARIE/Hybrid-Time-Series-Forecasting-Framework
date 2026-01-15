"""
Shared utilities for neural network models.
Used by lstm.py, gru.py, hybrid_seq.py, hybrid_par.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

from ..utils import (
    w_rmse, w_mae, directional_accuracy as dir_acc,
    save_pickle, save_json, copy_file, ensure_dir, load_pickle, EPS
)


# ==============================================================================
# SEQUENCE CREATION
# ==============================================================================

def make_sequences_eod_nn(X_2d, y_1d, w_1d, idx, lookback, stride=1):
    """
    Create sequences for training: use X up to day t (inclusive) -> predict y[t].
    
    Args:
        X_2d: Feature array (N, F)
        y_1d: Target array (N,)
        w_1d: Weight array (N,)
        idx: DatetimeIndex
        lookback: Number of timesteps
        stride: Step size between sequences
    
    Returns:
        Tuple of (X_seq, y_seq, w_seq, idx_seq)
    """
    X_2d, y_1d, w_1d = np.asarray(X_2d), np.asarray(y_1d), np.asarray(w_1d)
    N, F = X_2d.shape
    if N < lookback:
        raise ValueError(f"[ERROR] Not enough rows N={N} for lookback={lookback}.")
    X_seq, y_seq, w_seq, idx_seq = [], [], [], []
    for t in range(lookback - 1, N, stride):
        X_seq.append(X_2d[t - lookback + 1:t + 1, :])
        y_seq.append(y_1d[t])
        w_seq.append(w_1d[t])
        idx_seq.append(idx[t])
    return (np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_seq, dtype=np.float32),
            np.asarray(w_seq, dtype=np.float32),
            pd.DatetimeIndex(idx_seq))


def make_sequences_pred_nn(X_2d, y_1d, idx, lookback):
    """
    Create sequences for prediction (no weights, stride=1).
    
    Args:
        X_2d: Feature array (N, F)
        y_1d: Target array (N,)
        idx: DatetimeIndex
        lookback: Number of timesteps
    
    Returns:
        Tuple of (X_seq, y_seq, idx_seq)
    """
    X_2d, y_1d = np.asarray(X_2d), np.asarray(y_1d)
    N, F = X_2d.shape
    X_seq, y_seq, idx_seq = [], [], []
    for t in range(lookback - 1, N):
        X_seq.append(X_2d[t - lookback + 1:t + 1, :])
        y_seq.append(y_1d[t])
        idx_seq.append(idx[t])
    return (np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_seq, dtype=np.float32),
            pd.DatetimeIndex(idx_seq))


def split_valid_es_score(Xv_df, yv, wv, valid_es_start, valid_es_end, valid_score_start, valid_score_end):
    """
    Split validation into ES (early stopping) and SCORE sets.
    
    Returns:
        Tuple of ((X_es, y_es, w_es), (X_sc, y_sc, w_sc), mode_str)
    """
    if not isinstance(Xv_df.index, pd.DatetimeIndex):
        return (Xv_df, yv, wv), (Xv_df, yv, wv), "FULL_VALID"
    
    yv_s = pd.Series(yv, index=Xv_df.index)
    wv_s = pd.Series(wv, index=Xv_df.index)
    
    es_start, es_end = pd.Timestamp(valid_es_start), pd.Timestamp(valid_es_end)
    sc_start, sc_end = pd.Timestamp(valid_score_start), pd.Timestamp(valid_score_end)
    m_es = (Xv_df.index >= es_start) & (Xv_df.index <= es_end)
    m_sc = (Xv_df.index >= sc_start) & (Xv_df.index <= sc_end)
    mode_str = f"VALID_ES={valid_es_start}:{valid_es_end} / VALID_SCORE={valid_score_start}:{valid_score_end}"
    
    X_es, X_sc = Xv_df.loc[m_es], Xv_df.loc[m_sc]
    y_es, y_sc = yv_s.loc[m_es].to_numpy(float), yv_s.loc[m_sc].to_numpy(float)
    w_es, w_sc = wv_s.loc[m_es].to_numpy(float), wv_s.loc[m_sc].to_numpy(float)
    
    if len(X_es) > 0 and len(X_sc) > 0:
        return (X_es, y_es, w_es), (X_sc, y_sc, w_sc), mode_str
    return (Xv_df, yv, wv), (Xv_df, yv, wv), "FULL_VALID"


# ==============================================================================
# TRAINING HELPER
# ==============================================================================

def train_single_nn_model(
    model_type: str,
    X_train_dict: Dict[str, pd.DataFrame],
    X_valid_dict: Dict[str, pd.DataFrame],
    X_test_dict: Dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    w_test: np.ndarray,
    nn_config: Dict,
    hpo_config: Dict,
    build_model_fn,
    models_out_local: Optional[Path] = None,
    models_out_drive: Optional[Path] = None,
    pred_out_local: Optional[Path] = None,
    pred_out_drive: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Generic training function for a single neural network model type.
    
    Args:
        model_type: Model name (e.g., "lstm", "gru", "hybrid_seq", "hybrid_par")
        X_train_dict, X_valid_dict, X_test_dict: Feature dicts {feature_set: df}
        y_train, y_valid, y_test: Target arrays
        w_train, w_valid, w_test: Weight arrays
        nn_config: Model-specific config (from RUN_PARAMS[model_type])
        hpo_config: HPO config with valid_es/score dates
        build_model_fn: Function(inp, n_features, config) -> output layer
        models_out_local/drive: Model output directories
        pred_out_local/drive: Predictions output directories
    
    Returns:
        Dict with results or None if TensorFlow unavailable
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print(f"[SKIP] {model_type.upper()} — TensorFlow not available")
        return None
    
    # Config
    LOOKBACK = int(nn_config["lookback"])
    STRIDE = int(nn_config["stride"])
    DROPOUT = float(nn_config["dropout"])
    LR = float(nn_config["learning_rate"])
    CLIPNORM = float(nn_config["clipnorm"])
    EPOCHS = int(nn_config["epochs"])
    BATCH_SIZE = int(nn_config["batch_size"])
    PATIENCE = int(nn_config["patience"])
    RANDOM_SEED = int(nn_config["random_state"])
    FEATURE_SETS = nn_config["feature_sets"]
    LOSS = nn_config.get("loss", "mse")
    
    # Valid split dates
    VALID_ES_START = hpo_config["valid_es_start"]
    VALID_ES_END = hpo_config["valid_es_end"]
    VALID_SCORE_START = hpo_config["valid_score_start"]
    VALID_SCORE_END = hpo_config["valid_score_end"]
    
    # Convert to numpy
    y_train_np = np.asarray(y_train, dtype=float)
    y_valid_np = np.asarray(y_valid, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)
    w_train_np = np.asarray(w_train, dtype=float)
    w_valid_np = np.asarray(w_valid, dtype=float)
    w_test_np = np.asarray(w_test, dtype=float)
    
    if models_out_local:
        models_out_local = ensure_dir(Path(models_out_local))
    if models_out_drive:
        models_out_drive = ensure_dir(Path(models_out_drive))
    
    all_results = []
    
    print(f"\n{'#'*70}")
    print(f"# TRAINING {model_type.upper()} MODEL")
    print(f"{'#'*70}")
    
    for feature_set in FEATURE_SETS:
        print(f"\n{'='*60}")
        print(f"[INFO] Training {model_type.upper()} with feature set: {feature_set}")
        print(f"{'='*60}")
        
        X_train_nn = X_train_dict.get(feature_set)
        X_valid_nn = X_valid_dict.get(feature_set)
        X_test_nn = X_test_dict.get(feature_set)
        
        if X_train_nn is None:
            print(f"[WARN] Feature set {feature_set} not found, skipping...")
            continue
        
        n_features = X_train_nn.shape[1]
        print(f"[INFO] Shapes: TRAIN={X_train_nn.shape} | VALID={X_valid_nn.shape} | TEST={X_test_nn.shape}")
        
        # Split VALID -> ES + SCORE
        (X_valid_es_df, y_valid_es, w_valid_es), (X_valid_sc_df, y_valid_sc, w_valid_sc), valid_mode = split_valid_es_score(
            X_valid_nn, y_valid_np, w_valid_np,
            VALID_ES_START, VALID_ES_END, VALID_SCORE_START, VALID_SCORE_END
        )
        print(f"[INFO] VALID mode: {valid_mode}")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_nn.values)
        X_valid_es_scaled = scaler.transform(X_valid_es_df.values)
        X_valid_sc_scaled = scaler.transform(X_valid_sc_df.values)
        X_test_scaled = scaler.transform(X_test_nn.values)
        
        # Create sequences
        Xtr_seq, ytr_seq, wtr_seq, idx_tr = make_sequences_eod_nn(X_train_scaled, y_train_np, w_train_np, X_train_nn.index, LOOKBACK, STRIDE)
        Xes_seq, yes_seq, wes_seq, idx_es = make_sequences_eod_nn(X_valid_es_scaled, y_valid_es, w_valid_es, X_valid_es_df.index, LOOKBACK, STRIDE)
        Xsc_seq, ysc_seq, wsc_seq, idx_sc = make_sequences_eod_nn(X_valid_sc_scaled, y_valid_sc, w_valid_sc, X_valid_sc_df.index, LOOKBACK, STRIDE)
        Xte_seq, yte_seq, wte_seq, idx_te = make_sequences_eod_nn(X_test_scaled, y_test_np, w_test_np, X_test_nn.index, LOOKBACK, STRIDE)
        
        print(f"[INFO] Sequence shapes: TRAIN={Xtr_seq.shape} | VALID_ES={Xes_seq.shape} | VALID_SCORE={Xsc_seq.shape} | TEST={Xte_seq.shape}")
        
        # Build model
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        
        inp = keras.Input(shape=(LOOKBACK, n_features))
        out = build_model_fn(inp, n_features, nn_config)
        
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM), loss=LOSS)
        print(f"[INFO] Model built: {model.count_params()} parameters")
        
        # Score callback
        class ScoreSetCallback(keras.callbacks.Callback):
            def __init__(self, X_score, y_score, w_score):
                super().__init__()
                self.Xs, self.ys, self.ws = X_score, y_score, w_score
                self.best = np.inf
                self.best_weights = None
            def on_epoch_end(self, epoch, logs=None):
                pred = self.model.predict(self.Xs, verbose=0).reshape(-1)
                score = w_rmse(self.ys, pred, self.ws)
                if score < self.best:
                    self.best = score
                    self.best_weights = self.model.get_weights()
        
        score_cb = ScoreSetCallback(Xsc_seq, ysc_seq, wsc_seq)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            score_cb,
        ]
        
        # Train
        print(f"[INFO] Training {model_type.upper()} (epochs={EPOCHS}, batch_size={BATCH_SIZE}, patience={PATIENCE})...")
        history = model.fit(
            Xtr_seq, ytr_seq,
            sample_weight=wtr_seq,
            validation_data=(Xes_seq, yes_seq, wes_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=callbacks
        )
        
        # Restore best weights
        if score_cb.best_weights is not None:
            model.set_weights(score_cb.best_weights)
            print(f"[INFO] Restored best weights by VALID_SCORE wRMSE = {score_cb.best:.6f}")
        
        # Evaluate
        pred_sc = model.predict(Xsc_seq, verbose=0).reshape(-1)
        pred_te = model.predict(Xte_seq, verbose=0).reshape(-1)
        baseline_sc, baseline_te = np.zeros_like(ysc_seq), np.zeros_like(yte_seq)
        
        # Metrics
        results = {
            "model_type": model_type,
            "feature_set": feature_set,
            "n_features": n_features,
            "valid_mode": valid_mode,
            "epochs_trained": len(history.history["loss"]),
            "baseline_valid_wrmse": w_rmse(ysc_seq, baseline_sc, wsc_seq),
            "baseline_valid_diracc": dir_acc(ysc_seq, baseline_sc),
            "baseline_test_wrmse": w_rmse(yte_seq, baseline_te, wte_seq),
            "baseline_test_diracc": dir_acc(yte_seq, baseline_te),
            "model_valid_wrmse": w_rmse(ysc_seq, pred_sc, wsc_seq),
            "model_valid_wmae": w_mae(ysc_seq, pred_sc, wsc_seq),
            "model_valid_diracc": dir_acc(ysc_seq, pred_sc),
            "model_test_wrmse": w_rmse(yte_seq, pred_te, wte_seq),
            "model_test_wmae": w_mae(yte_seq, pred_te, wte_seq),
            "model_test_diracc": dir_acc(yte_seq, pred_te),
        }
        results["valid_wrmse_improvement"] = results["baseline_valid_wrmse"] - results["model_valid_wrmse"]
        results["test_wrmse_improvement"] = results["baseline_test_wrmse"] - results["model_test_wrmse"]
        all_results.append(results)
        
        print(f"\n[RESULT] {model_type.upper()} | {feature_set} | n_features={n_features}")
        print(f"  BASELINE VALID_SCORE: wRMSE={results['baseline_valid_wrmse']:.6f} | DirAcc={results['baseline_valid_diracc']:.4f}")
        print(f"  MODEL    VALID_SCORE: wRMSE={results['model_valid_wrmse']:.6f} | DirAcc={results['model_valid_diracc']:.4f}")
        print(f"  MODEL    TEST:        wRMSE={results['model_test_wrmse']:.6f} | DirAcc={results['model_test_diracc']:.4f}")
        
        # Save
        if models_out_local:
            model_path = models_out_local / f"{model_type}_{feature_set}.keras"
            scaler_path = models_out_local / f"{model_type}_{feature_set}_scaler.pkl"
            config_path = models_out_local / f"{model_type}_{feature_set}_config.json"
            model.save(model_path)
            save_pickle(scaler, scaler_path)
            save_json(nn_config, config_path)
            
            if models_out_drive:
                copy_file(model_path, models_out_drive / model_path.name)
                copy_file(scaler_path, models_out_drive / scaler_path.name)
                copy_file(config_path, models_out_drive / config_path.name)
            
            print(f"[OK] Saved: {model_path.name}")
        
        # Predictions
        if pred_out_local:
            PRED_LOCAL = ensure_dir(Path(pred_out_local) / f"{model_type}_{feature_set}")
            PRED_DRIVE = ensure_dir(Path(pred_out_drive) / f"{model_type}_{feature_set}") if pred_out_drive else None
            
            pd.DataFrame({
                "date": idx_sc, "actual": ysc_seq, "baseline_zero": baseline_sc,
                "predicted": pred_sc, "sample_weight": wsc_seq,
            }).reset_index(drop=True).to_csv(PRED_LOCAL / "predictions_valid.csv", index=False)
            
            pd.DataFrame({
                "date": idx_te, "actual": yte_seq, "baseline_zero": baseline_te,
                "predicted": pred_te, "sample_weight": wte_seq,
            }).reset_index(drop=True).to_csv(PRED_LOCAL / "predictions_test.csv", index=False)
            
            if PRED_DRIVE:
                copy_file(PRED_LOCAL / "predictions_valid.csv", PRED_DRIVE / "predictions_valid.csv")
                copy_file(PRED_LOCAL / "predictions_test.csv", PRED_DRIVE / "predictions_test.csv")
            
            print(f"[OK] Predictions: predictions/{model_type}_{feature_set}/")
    
    # Summary
    results_df = pd.DataFrame(all_results)
    print(f"\n{'='*60}")
    print(f"[INFO] {model_type.upper()} TRAINING SUMMARY")
    print(f"{'='*60}")
    print(results_df[["feature_set", "n_features", "model_valid_wrmse", "model_test_wrmse", "valid_wrmse_improvement"]].to_string())
    
    if models_out_local:
        summary_path = models_out_local / f"{model_type}_summary.csv"
        results_df.to_csv(summary_path, index=False)
        if models_out_drive:
            copy_file(summary_path, models_out_drive / summary_path.name)
    
    return {"results_df": results_df, "all_results": all_results}


# ==============================================================================
# PREDICTION HELPER
# ==============================================================================

def predict_single_nn_model(
    model_type: str,
    X_test_dict: Dict[str, pd.DataFrame],
    y_test: np.ndarray,
    w_test: np.ndarray,
    nn_config: Dict,
    plot_config: Dict,
    models_dir_local: Path,
    pred_out_local: Optional[Path] = None,
    pred_out_drive: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Generic prediction function for a single neural network model type.
    
    Returns:
        Dict with prediction results or None if TensorFlow unavailable
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[SKIP] {model_type.upper()} predict — TensorFlow not available")
        return None
    
    N_PLOT = int(plot_config.get("n_plot", 60))
    FIGSIZE = tuple(plot_config.get("figsize", [12, 6]))
    DPI = int(plot_config.get("dpi", 100))
    
    LOOKBACK = int(nn_config["lookback"])
    FEATURE_SETS = nn_config["feature_sets"]
    
    y_test_arr = np.asarray(y_test, dtype=float)
    w_test_arr = np.asarray(w_test, dtype=float)
    
    models_dir_local = Path(models_dir_local)
    
    if pred_out_local:
        pred_out_local = ensure_dir(Path(pred_out_local))
    if pred_out_drive:
        pred_out_drive = ensure_dir(Path(pred_out_drive))
    
    all_pred_results = []
    
    print(f"\n{'#'*70}")
    print(f"# PREDICTING WITH {model_type.upper()} MODEL")
    print(f"{'#'*70}")
    
    for feature_set in FEATURE_SETS:
        print(f"\n{'='*60}")
        print(f"[INFO] Predicting with {model_type.upper()}: {feature_set}")
        print(f"{'='*60}")
        
        model_path = models_dir_local / f"{model_type}_{feature_set}.keras"
        scaler_path = models_dir_local / f"{model_type}_{feature_set}_scaler.pkl"
        
        if not model_path.exists():
            print(f"[WARN] Model not found: {model_path}, skipping...")
            continue
        
        model = keras.models.load_model(model_path)
        scaler = load_pickle(scaler_path)
        
        X_test = X_test_dict.get(feature_set)
        if X_test is None:
            print(f"[WARN] Feature set {feature_set} not found, skipping...")
            continue
        
        n_features = X_test.shape[1]
        print(f"[INFO] Loaded: model={model_path.name} | X_test={X_test.shape}")
        
        # Scale + Sequences
        X_test_scaled = scaler.transform(X_test.values)
        X_seq, y_seq, idx_seq = make_sequences_pred_nn(X_test_scaled, y_test_arr, X_test.index, LOOKBACK)
        print(f"[INFO] Sequences: {X_seq.shape}")
        
        # Predict
        pred_seq = model.predict(X_seq, verbose=0).reshape(-1)
        hist_df = pd.DataFrame({"date": idx_seq, "actual": y_seq, "y_pred": pred_seq}).set_index("date")
        
        # Tomorrow
        last_date = X_test.index[-1]
        X_last_window = X_test_scaled[-LOOKBACK:, :].reshape(1, LOOKBACK, -1).astype(np.float32)
        pred_tomorrow = float(model.predict(X_last_window, verbose=0).reshape(-1)[0])
        pred_tomorrow_date = last_date + pd.Timedelta(days=1)
        
        # Metrics
        w_seq = w_test_arr[LOOKBACK - 1:]
        w_norm = w_seq / (w_seq.sum() + EPS)
        test_wrmse = float(np.sqrt(np.sum(w_norm * (y_seq - pred_seq) ** 2)))
        test_dir_acc = float(np.mean((y_seq > 0) == (pred_seq > 0)))
        
        print(f"[INFO] TEST wRMSE={test_wrmse:.6f} | DirAcc={test_dir_acc:.4f}")
        print(f"[INFO] Tomorrow prediction: {pred_tomorrow:.6f} ({np.expm1(pred_tomorrow)*100:.4f}%)")
        
        result = {
            "model_type": model_type,
            "feature_set": feature_set,
            "n_features": n_features,
            "test_wrmse": test_wrmse,
            "test_dir_acc": test_dir_acc,
            "last_data_date": str(last_date.date()),
            "pred_tomorrow_logret": pred_tomorrow,
            "pred_tomorrow_pct": float(np.expm1(pred_tomorrow) * 100),
        }
        all_pred_results.append(result)
        
        # Save
        if pred_out_local:
            PRED_LOCAL = ensure_dir(pred_out_local / f"{model_type}_{feature_set}")
            PRED_DRIVE = ensure_dir(pred_out_drive / f"{model_type}_{feature_set}") if pred_out_drive else None
            
            pd.DataFrame([{
                "feature_set": feature_set, "last_data_date": last_date,
                "predicted_for": "next_trading_day",
                "pred_logret": pred_tomorrow,
                "pred_return_pct": float(np.expm1(pred_tomorrow) * 100),
            }]).to_csv(PRED_LOCAL / "tomorrow.csv", index=False)
            
            hist_tail = hist_df.tail(N_PLOT).copy()
            hist_tail.to_csv(PRED_LOCAL / "backtest.csv")
            
            if PRED_DRIVE:
                copy_file(PRED_LOCAL / "tomorrow.csv", PRED_DRIVE / "tomorrow.csv")
                copy_file(PRED_LOCAL / "backtest.csv", PRED_DRIVE / "backtest.csv")
            
            # Plot
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(hist_tail.index, hist_tail["actual"].values, linewidth=1, label="Actual")
            ax.plot(hist_tail.index, hist_tail["y_pred"].values, linewidth=1, label="Predicted")
            ax.scatter([pred_tomorrow_date], [pred_tomorrow], s=90, marker="X", color="red", label=f"Tomorrow: {pred_tomorrow:.4f}")
            ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"{model_type.upper()} — {feature_set} — last {len(hist_tail)} days + tomorrow")
            ax.set_xlabel("Date")
            ax.set_ylabel("Log Return")
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(PRED_LOCAL / "plot.png", dpi=DPI)
            
            if PRED_DRIVE:
                copy_file(PRED_LOCAL / "plot.png", PRED_DRIVE / "plot.png")
            
            plt.close(fig)
            print(f"[OK] Saved: predictions/{model_type}_{feature_set}/")
    
    # Summary
    pred_summary_df = pd.DataFrame(all_pred_results)
    print(f"\n{'='*60}")
    print(f"[INFO] {model_type.upper()} PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(pred_summary_df.to_string())
    
    if pred_out_local:
        summary_path = pred_out_local / f"{model_type}_predictions_summary.csv"
        pred_summary_df.to_csv(summary_path, index=False)
        if pred_out_drive:
            copy_file(summary_path, pred_out_drive / summary_path.name)
    
    return {"pred_results": all_pred_results, "summary_df": pred_summary_df}
