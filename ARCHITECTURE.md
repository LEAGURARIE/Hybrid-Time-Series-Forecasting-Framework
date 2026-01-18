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

**Author:** Lea G. with [Claude Code](https://claude.ai) (Anthropic)

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
│   │             │    │ • SHAP      │    │             │                    │
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
| `load_with_fallback()` | Load from run or project directory |
| `compute_metrics()` | wRMSE, wMAE, DirAcc calculation |
| `bootstrap_all_metrics()` | Bootstrap confidence intervals |
| `apply_shap_feature_selection()` | Filter to SHAP top-N features |
| `save_shap_top_features()` | Save SHAP feature list for reuse |

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
| 1 | Spearman correlation filter |
| 2 | XGBoost GAIN importance |
| 3 | Permutation importance validation |
| 4 | Mutual Information (for neural nets) |

### `src/models/`
Model implementations.

| Module | Models | Features |
|--------|--------|----------|
| `xgboost_model.py` | XGBoost | SHAP analysis, feature selection |
| `lightgbm_model.py` | LightGBM | SHAP analysis, feature selection |
| `lstm_gru.py` | LSTM, GRU | Multiple feature sets |
| `hybrid.py` | Sequential, Parallel | Combined architectures |
| `ensemble.py` | Weighted avg, Stacking | Flexible/Rigid filtering |
| `neural_utils.py` | - | Training utilities for NN |

### `src/tuning/hpo.py`
Hyperparameter optimization (3-stage manual sampling).

| Stage | Description |
|-------|-------------|
| 1 | Broad search across parameter space |
| 2 | Refinement around best configuration |
| 3 | Low learning rate variants |

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
│                                ▼                                            │
│        ┌───────────────────────┼───────────────────────┐                   │
│        │                       │                       │                    │
│        ▼                       ▼                       ▼                    │
│   ╔═════════╗            ╔═════════╗            ╔═════════╗                │
│   ║ XGBoost ║            ║ LightGBM║            ║ Neural  ║                │
│   ║ + SHAP  ║            ║ + SHAP  ║            ║ Models  ║                │
│   ╚════╤════╝            ╚════╤════╝            ╚════╤════╝                │
│        │                       │                       │                    │
│        │              ┌────────┴────────┐              │                    │
│        │              │                 │              │                    │
│        │              ▼                 ▼              │                    │
│        │         ┌────────┐       ┌────────┐          │                    │
│        │         │  LSTM  │       │  GRU   │          │                    │
│        │         └───┬────┘       └───┬────┘          │                    │
│        │             │                │               │                    │
│        │             └───────┬────────┘               │                    │
│        │                     │                        │                    │
│        │              ┌──────┴──────┐                 │                    │
│        │              ▼             ▼                 │                    │
│        │         ┌────────┐   ┌────────┐              │                    │
│        │         │Hybrid  │   │Hybrid  │              │                    │
│        │         │  Seq   │   │  Par   │              │                    │
│        │         └───┬────┘   └───┬────┘              │                    │
│        │             │            │                   │                    │
│        └─────────────┴────────────┴───────────────────┘                    │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   7. ENSEMBLE         ║  models/ensemble.py           │
│                    ║   (Flexible/Rigid)    ║                               │
│                    ╚═══════════╤═══════════╝                               │
│                                │                                            │
│                                ▼                                            │
│                    ╔═══════════════════════╗                               │
│                    ║   8. SUMMARY          ║  Metrics, Bootstrap CI        │
│                    ╚═══════════════════════╝                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## SHAP Feature Selection Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CONFIG["shap"]["top_n_features"] = None (default)                        │
│   └── All models use their default feature sets                            │
│                                                                             │
│   CONFIG["shap"]["top_n_features"] = N (e.g., 10)                          │
│   └── SHAP-based feature selection enabled                                 │
│                                                                             │
│       ┌─────────────────────────────────────────────────────────────┐      │
│       │                        RUN 1                                 │      │
│       │                                                             │      │
│       │   XGBoost trains on xgb_selected                            │      │
│       │         │                                                   │      │
│       │         ▼                                                   │      │
│       │   SHAP analysis computes feature importance                 │      │
│       │         │                                                   │      │
│       │         ▼                                                   │      │
│       │   Saves: data/processed/shap_top_N_features.pkl             │      │
│       │                                                             │      │
│       └─────────────────────────────────────────────────────────────┘      │
│                                    │                                        │
│                                    ▼                                        │
│       ┌─────────────────────────────────────────────────────────────┐      │
│       │                        RUN 2+                                │      │
│       │                                                             │      │
│       │   shap_top_N_features.pkl exists                            │      │
│       │         │                                                   │      │
│       │         ▼                                                   │      │
│       │   XGBoost/LightGBM: Load full X, filter to SHAP features    │      │
│       │   Neural/Hybrid: Add shap_top_N to feature_sets list        │      │
│       │                                                             │      │
│       │   Feature sets become:                                      │      │
│       │   • neural_40, neural_80, shap_top_N                        │      │
│       │                                                             │      │
│       └─────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Ensemble Modes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   FLEXIBLE MODE (all filters = None)                                       │
│   ├── All trained models included                                          │
│   └── Maximum diversity, exploratory use                                   │
│                                                                             │
│   RIGID MODE (one or more filters set)                                     │
│   ├── min_diracc: Remove models below accuracy threshold                   │
│   ├── max_wrmse: Remove models above error threshold                       │
│   ├── top_n: Keep only N best models                                       │
│   └── Production use, quality control                                      │
│                                                                             │
│   Filter Metrics:                                                          │
│   ├── use_test: False (default) → Filter by VALID metrics                  │
│   └── use_test: True → Filter by TEST metrics (analysis only)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CLI vs Notebook

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│        CLI (PyCharm/Terminal)              Notebook (Colab)                │
│        ──────────────────────              ─────────────────                │
│                │                                  │                         │
│                ▼                                  ▼                         │
│        ┌──────────────┐                   ┌──────────────┐                 │
│        │ src/config.py│                   │    Cell 3    │                 │
│        │    CONFIG    │                   │  RUN_PARAMS  │                 │
│        └──────────────┘                   └──────────────┘                 │
│                │                                  │                         │
│                ▼                                  ▼                         │
│        ┌──────────────┐                   ┌──────────────┐                 │
│        │run_pipeline  │                   │   Notebook   │                 │
│        │    .py       │                   │    Cells     │                 │
│        └──────────────┘                   └──────────────┘                 │
│                │                                  │                         │
│                ▼                                  ▼                         │
│        scripts/output/                    LOCAL + Google Drive             │
│        runs/{RUN_ID}/                     /runs/{RUN_ID}/                  │
│                                                                             │
│   Configurations should be kept synchronized for consistent results        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Snapshot (Reproducibility)

The notebook automatically saves a copy of itself at the start of each run for full reproducibility:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   EXPORT_CODE = True  (Cell 3)                                             │
│         │                                                                   │
│         ▼                                                                   │
│   snapshot_code() is called at run start                                   │
│         │                                                                   │
│         ├──────────────────────────────────────────┐                       │
│         ▼                                          ▼                        │
│   LOCAL: runs/{RUN_ID}/code_snapshot/     DRIVE: runs/{RUN_ID}/code_snapshot/
│         │                                          │                        │
│         ├── google_stock_ml_unified.ipynb          ├── google_stock_ml_unified.ipynb
│         └── snapshot_meta.json                     └── snapshot_meta.json   │
│                                                                             │
│   snapshot_meta.json contains:                                             │
│   • run_id                                                                 │
│   • timestamp                                                              │
│   • source_path                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Every run has its exact code preserved
- Can reproduce any experiment by loading the snapshot
- Both LOCAL and DRIVE have copies for redundancy
- Metadata tracks when and where the snapshot was taken

**Disable:** Set `EXPORT_CODE = False` in Cell 3 to skip snapshots.

---

## Directory Structure

```
GoogleStockProject/
│
├── src/                              # Core modules (CLI)
│   ├── config.py                     # CONFIG + setup_cli_paths()
│   ├── utils.py                      # I/O, metrics, SHAP utilities
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
│   ├── run_pipeline.py               # CLI entry point
│   └── output/                       # Generated at runtime
│       ├── data/
│       │   ├── raw/                  # Downloaded data
│       │   ├── interim/              # Engineered features
│       │   └── processed/            # Train/Valid/Test splits
│       │       └── shap_top_N_features.pkl  # SHAP feature list
│       ├── runs/
│       │   └── {RUN_ID}/
│       │       ├── config/           # Run configuration (run_params.json)
│       │       ├── code_snapshot/    # Notebook copy for reproducibility
│       │       ├── feature_selection/# Feature importance
│       │       ├── model_selection/  # HPO results
│       │       ├── models/           # Trained models + SHAP
│       │       ├── predictions/      # Model predictions
│       │       └── outputs/          # Ensemble results
│       └── results_summary/          # Accumulated results
│
├── notebooks/
│   └── google_stock_ml_unified.ipynb # Self-contained Colab version
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

### SHAP Outputs

```
models/
├── shap_feature_importance.csv    # Feature ranking
├── shap_summary_bar.png           # Bar plot
├── shap_summary_beeswarm.png      # Beeswarm plot
└── shap_values.pkl                # Raw SHAP values

data/processed/
└── shap_top_N_features.pkl        # Persistent feature list
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
│  1. Resample with replacement (N times)                                    │
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
│     CI_lower = percentile(metrics, 2.5%)                                   │
│     CI_upper = percentile(metrics, 97.5%)                                  │
│                                                                             │
│  3. Interpretation                                                          │
│                                                                             │
│     If CIs don't overlap → Difference is statistically significant         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fallback Logic

Modules support loading from run-specific or project-level directories:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   load_with_fallback(filename, run_dir, fallback_dir)                      │
│                                                                             │
│   Load file request                                                         │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────┐                                      │
│   │  runs/{RUN_ID}/processed/       │  ← Try first                         │
│   │  (run-specific)                 │                                      │
│   └─────────────┬───────────────────┘                                      │
│                 │                                                           │
│           Found? ──── Yes ──→ Use it                                       │
│                 │                                                           │
│                No                                                           │
│                 │                                                           │
│                 ▼                                                           │
│   ┌─────────────────────────────────┐                                      │
│   │  data/processed/                │  ← Fallback                          │
│   │  (project-level, persistent)    │                                      │
│   └─────────────────────────────────┘                                      │
│                                                                             │
│   Benefits:                                                                 │
│   • Run individual steps without full pipeline                              │
│   • Share processed data between runs                                       │
│   • Persist SHAP features and best_params across experiments                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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
      limit_start        train_end    valid_es_end    valid_score_end    end_date

VALID_ES    = Early stopping during training
VALID_SCORE = Model selection during HPO
TEST        = Final evaluation (never seen during training/tuning)
```
