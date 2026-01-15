# Google Stock ML - Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    ██████╗  ██████╗  ██████╗  ██████╗ ██╗                    ║
║                   ██╔════╝ ██╔═══██╗██╔═══██╗██╔════╝ ██║                    ║
║                   ██║  ███╗██║   ██║██║   ██║██║  ███╗██║                    ║
║                   ██║   ██║██║   ██║██║   ██║██║   ██║██║                    ║
║                   ╚██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝███████╗               ║
║                    ╚═════╝  ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝               ║
║                                                                              ║
║                        STOCK  ML  PIPELINE                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Author:** Lea G.

---

## Overview

This project implements a **modular architecture** for stock return prediction. Each component is a standalone module that can be:
- Used independently
- Tested in isolation
- Combined into a full pipeline

Two interfaces are available:
- **CLI** (`run_pipeline.py`) - orchestrates modules for batch execution
- **Notebook** (`google_stock_ml_unified.ipynb`) - self-contained interactive version

---

## Modular Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           src/ (Core Modules)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │   config    │    │   utils     │    │    data/    │                    │
│   │             │    │             │    │             │                    │
│   │ • CONFIG    │    │ • I/O       │    │ • loaders   │                    │
│   │ • paths     │    │ • metrics   │    │ • split     │                    │
│   └─────────────┘    └─────────────┘    └─────────────┘                    │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │  features/  │    │   models/   │    │   tuning/   │                    │
│   │             │    │             │    │             │                    │
│   │ • engineer  │    │ • xgboost   │    │ • hpo       │                    │
│   │ • selection │    │ • lightgbm  │    │             │                    │
│   │             │    │ • lstm_gru  │    │             │                    │
│   │             │    │ • hybrid    │    │             │                    │
│   │             │    │ • ensemble  │    │             │                    │
│   └─────────────┘    └─────────────┘    └─────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### `src/config.py`
Central configuration for CLI execution.

| Export | Description |
|--------|-------------|
| `CONFIG` | Dictionary with all hyperparameters |
| `setup_cli_paths()` | Creates directory structure for runs |

### `src/utils.py`
Shared utilities across all modules.

| Function | Description |
|----------|-------------|
| `save_pickle()` / `load_pickle()` | Binary serialization |
| `save_json()` / `load_json()` | JSON serialization |
| `compute_metrics()` | wRMSE, wMAE, DirAcc calculation |
| `bootstrap_all_metrics()` | Bootstrap confidence intervals |

### `src/data/loaders.py`
Data fetching from external APIs.

| Function | Source | Data |
|----------|--------|------|
| `load_prices()` | yfinance | OHLCV for tickers |
| `load_macro()` | FRED | CPI, Fed Funds Rate |
| `load_earnings()` | yfinance | EPS surprise |
| `load_all_data()` | All above | Combined DataFrame |

### `src/data/split.py`
Temporal train/validation/test splitting.

| Function | Description |
|----------|-------------|
| `create_splits()` | Creates X_train, X_valid, X_test, y_*, weights |

### `src/features/engineering.py`
Feature construction from raw data.

| Feature Type | Examples |
|--------------|----------|
| Technical | Rolling mean, std, momentum |
| Cross-asset | Beta, correlation vs SPY/QQQ |
| Macro | CPI acceleration, rate changes |
| Regime | VIX levels, volatility regime |
| EU signals | DAX overnight gaps |

### `src/features/selection.py`
Multi-stage feature selection.

| Stage | Method |
|-------|--------|
| 1 | Spearman correlation filter (ρ > 0.9) |
| 2 | XGBoost GAIN importance |
| 3 | Permutation importance validation |
| 4 | Mutual Information (for neural nets) |

### `src/models/`
Model implementations.

| Module | Models |
|--------|--------|
| `xgboost_model.py` | XGBoost + SHAP |
| `lightgbm_model.py` | LightGBM + SHAP |
| `lstm_gru.py` | LSTM, GRU |
| `hybrid.py` | Sequential, Parallel hybrids |
| `ensemble.py` | Weighted average, Stacking |
| `neural_utils.py` | Training utilities for NN |

### `src/tuning/hpo.py`
Hyperparameter optimization (3-stage Optuna).

| Stage | Trials | Description |
|-------|--------|-------------|
| 1 | 160 | Broad search |
| 2 | 80 | Refinement around best |
| 3 | 40 | Low learning rate variants |

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ yfinance │    │   FRED   │    │ Earnings │    │  Events  │             │
│   │   API    │    │   API    │    │ Calendar │    │   Data   │             │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│        │               │               │               │                    │
│        └───────────────┴───────────────┴───────────────┘                    │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   1. LOAD DATA        ║  data/loaders.py              │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   2. FEATURES         ║  features/engineering.py      │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   3. SPLIT DATA       ║  data/split.py                │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   4. FEATURE SELECT   ║  features/selection.py        │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   5. HPO (XGB/LGB)    ║  tuning/hpo.py                │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│   ┌────────────────────────────┼────────────────────────────┐               │
│   │                            │                            │               │
│   ▼                            ▼                            ▼               │
│ ┌─────────┐              ┌──────────┐              ┌─────────────┐         │
│ │ XGBoost │              │ LightGBM │              │   Neural    │         │
│ │         │              │          │              │   Networks  │         │
│ └────┬────┘              └────┬─────┘              └──────┬──────┘         │
│      │                        │         ┌────────────────┬┴───────────┐    │
│      │                        │         │                │            │    │
│      │                        │         ▼                ▼            ▼    │
│      │                        │    ┌────────┐      ┌────────┐    ┌───────┐│
│      │                        │    │  LSTM  │      │  GRU   │    │Hybrid ││
│      │                        │    └────┬───┘      └────┬───┘    └───┬───┘│
│      │                        │         │               │            │    │
│      └────────────────────────┴─────────┴───────────────┴────────────┘    │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   7. ENSEMBLE         ║  models/ensemble.py           │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   8. SUMMARY          ║  Bootstrap CI + Reports       │
│                    ╚═══════════════════════╝                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Two Interfaces

### CLI (`scripts/run_pipeline.py`)

Orchestrates modules for batch/automated execution.

```bash
# Full pipeline
python scripts/run_pipeline.py

# Specific steps
python scripts/run_pipeline.py --steps models,ensemble,summary

# Specific models
python scripts/run_pipeline.py --models xgb,lgb,lstm
```

**Configuration:** `src/config.py` → `CONFIG` dictionary

### Notebook (`notebooks/google_stock_ml_unified.ipynb`)

Self-contained interactive version for Google Colab.

- All code is embedded in cells
- Does NOT import from `src/` modules
- Independent configuration via `RUN_PARAMS` (Cell 3)
- Google Drive integration for persistence

**Configuration:** `RUN_PARAMS` dictionary in notebook Cell 3

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CLI (run_pipeline.py)              Notebook (Colab)                       │
│   ─────────────────────              ─────────────────                      │
│                                                                             │
│   src/config.py                      RUN_PARAMS (Cell 3)                    │
│        │                                   │                                │
│        ▼                                   ▼                                │
│   ┌─────────┐                        ┌─────────────┐                       │
│   │ CONFIG  │                        │ Embedded    │                       │
│   └────┬────┘                        │ Code Cells  │                       │
│        │                             └──────┬──────┘                       │
│        ▼                                    ▼                              │
│   src/ modules                       Self-contained                        │
│        │                             functions                             │
│        ▼                                    ▼                              │
│   output/runs/                       Google Drive                          │
│                                      /runs/                                │
│                                                                             │
│   ⚠️ Configurations are INDEPENDENT - changes in one don't affect other   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
GoogleStockProject/
│
├── src/                              # Core modules (CLI)
│   ├── config.py                     # CONFIG + setup_cli_paths()
│   ├── utils.py                      # I/O, metrics, statistics
│   ├── data/
│   │   ├── loaders.py                # Data fetching
│   │   └── split.py                  # Train/Valid/Test split
│   ├── features/
│   │   ├── engineering.py            # Feature construction
│   │   └── selection.py              # Feature selection
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── lstm_gru.py
│   │   ├── neural_utils.py
│   │   ├── hybrid.py
│   │   └── ensemble.py
│   └── tuning/
│       └── hpo.py
│
├── scripts/
│   └── run_pipeline.py               # CLI entry point
│
├── notebooks/
│   └── google_stock_ml_unified.ipynb # Self-contained Colab version
│
├── output/                           # Generated at runtime
│   ├── data/
│   │   ├── raw/                      # Downloaded data
│   │   ├── interim/                  # Engineered features
│   │   └── processed/                # Train/Valid/Test splits
│   ├── runs/
│   │   └── {RUN_ID}/
│   │       ├── config/               # Run configuration snapshot
│   │       ├── feature_selection/    # Feature importance
│   │       ├── model_selection/      # HPO results
│   │       ├── models/               # Trained models
│   │       ├── predictions/          # Model predictions
│   │       └── outputs/              # Ensemble results
│   └── results_summary/              # Accumulated results
│
└── requirements.txt
```

---

## Output Formats

### Predictions CSV

All models output predictions in a standardized format:

```
predictions/{model}/predictions_{split}.csv
predictions/{model}_{feature_set}/predictions_{split}.csv

Columns:
├── actual          ← True y values
├── predicted       ← Model predictions
└── sample_weight   ← Sample weights for weighted metrics
```

### Results Summary

```
results_summary/
├── all_results.csv           # All runs, all models
├── best_per_model.csv        # Best config per model type
├── bootstrap_ci.csv          # Confidence intervals
└── RESULTS.md                # Human-readable report
```

---

## Bootstrap Confidence Intervals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  For each model's TEST predictions:                                         │
│                                                                             │
│  1. Resample with replacement (N = 1000 times)                             │
│                                                                             │
│     Original:  [y1, y2, y3, y4, y5, ...]                                   │
│                     │                                                       │
│                     ▼                                                       │
│     Bootstrap 1: [y3, y1, y3, y5, y2, ...]  → compute metrics              │
│     Bootstrap 2: [y2, y2, y4, y1, y5, ...]  → compute metrics              │
│     ...                                                                     │
│     Bootstrap N: [y5, y3, y1, y4, y2, ...]  → compute metrics              │
│                                                                             │
│  2. Compute percentiles (95% CI)                                           │
│                                                                             │
│     CI_lower = percentile(metric_boots, 2.5%)                              │
│     CI_upper = percentile(metric_boots, 97.5%)                             │
│                                                                             │
│  3. Interpretation                                                          │
│                                                                             │
│     Model A: wRMSE = 0.0120 [0.0113, 0.0127]                              │
│     Model B: wRMSE = 0.0135 [0.0128, 0.0142]                              │
│                                                                             │
│     CIs don't overlap → Difference is statistically significant           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fallback Logic

Modules support loading from run-specific or project-level directories:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Load file request                                             │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────┐                          │
│   │  runs/{RUN_ID}/processed/       │  ← Try first             │
│   │  (run-specific)                 │                          │
│   └─────────────┬───────────────────┘                          │
│                 │                                               │
│           Found? ──── Yes ──→ Use it                           │
│                 │                                               │
│                No                                               │
│                 │                                               │
│                 ▼                                               │
│   ┌─────────────────────────────────┐                          │
│   │  data/processed/                │  ← Fallback              │
│   │  (project-level, persistent)    │                          │
│   └─────────────────────────────────┘                          │
│                                                                 │
│   Benefits:                                                     │
│   • Run individual steps without full pipeline                  │
│   • Share processed data between runs                           │
│   • Persist best_params across experiments                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **wRMSE** | Weighted Root Mean Squared Error | √(Σwᵢ(yᵢ-ŷᵢ)² / Σwᵢ) |
| **wMAE** | Weighted Mean Absolute Error | Σwᵢ|yᵢ-ŷᵢ| / Σwᵢ |
| **DirAcc** | Directional Accuracy | % correct sign predictions |

Sample weights use exponential decay to emphasize recent observations.

---

## Data Split Timeline

```
|-------- TRAIN --------|--- VALID_ES ---|--- VALID_SCORE ---|---- TEST ----|
2005                   2020            2021               2022            2023→
     limit_start        train_end    valid_es_end    valid_score_end    end_date

VALID_ES    = Early stopping during training
VALID_SCORE = Model selection during HPO
TEST        = Final evaluation (never seen during training/tuning)
```
