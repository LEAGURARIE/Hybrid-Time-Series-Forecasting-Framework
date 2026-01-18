# Google Stock Returns Prediction Framework

**A Multi-Layered Research and Forecasting System for GOOGL Stock**

End-to-end ML pipeline for predicting daily stock log returns, combining tree-based models with Deep Learning, advanced ensemble methods, and statistical validation.

---

## Why This Framework?

### The Challenge
Predicting stock returns is notoriously difficult. Single models often fail to capture the complexity of financial markets, and overfitting is a constant risk.

### Our Approach
This framework addresses these challenges through:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREDICTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │   DATA      │    │  FEATURE    │    │   MODEL     │    │  ENSEMBLE  │  │
│   │  SOURCES    │───▶│ ENGINEERING │───▶│  TRAINING   │───▶│  & VOTING  │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│   • Stock prices     • 100+ features    • 6 diverse       • Weighted      │
│   • Macro data       • Auto-selection     models            combination   │
│   • Earnings         • SHAP ranking     • Parallel        • Quality       │
│   • VIX/Bonds                             training          filtering     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Benefits

### 1. Multi-Model Diversity
Instead of relying on a single model, the framework trains **6 different model types** that capture different patterns:

| Model Type | What It Captures |
|------------|------------------|
| **XGBoost** | Non-linear feature interactions |
| **LightGBM** | Fast gradient patterns |
| **LSTM** | Long-term sequential memory |
| **GRU** | Short-term temporal patterns |
| **Hybrid-Seq** | Combined sequential learning |
| **Hybrid-Par** | Parallel pattern extraction |

> **Why it matters**: When one model fails, others may succeed. The ensemble combines their strengths.

### 2. Intelligent Feature Selection

The framework automatically identifies the most predictive features through multiple methods:

```
Raw Features (100+)
       │
       ▼
┌──────────────────┐
│ Correlation      │ ──▶ Remove redundant features
│ Filter           │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ XGBoost          │ ──▶ Rank by predictive power
│ Importance       │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ SHAP Analysis    │ ──▶ Understand WHY features matter
│ (Optional)       │
└──────────────────┘
       │
       ▼
  Selected Features
  (Optimized subset)
```

> **Why it matters**: Fewer, better features = less overfitting, faster training, better generalization.

### 3. Flexible Ensemble Control

Choose how strictly to filter models before combining:

| Mode | Use When |
|------|----------|
| **Flexible** (all filters off) | Exploratory analysis, maximum diversity |
| **Rigid** (filters on) | Production, only proven performers |

### 4. Full Explainability

Every prediction can be explained:
- **SHAP values**: See which features drove each prediction
- **Feature importance**: Understand global patterns
- **Bootstrap CI**: Know the uncertainty in your metrics

---

## Modular Architecture

The framework is built as independent, reusable components:

```
┌────────────────────────────────────────────────────────────────┐
│                      YOU CAN USE...                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   FULL      │   │  SPECIFIC   │   │  INDIVIDUAL │          │
│  │  PIPELINE   │   │   STEPS     │   │   MODULES   │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│        │                 │                 │                   │
│        ▼                 ▼                 ▼                   │
│  Run everything    --steps models    Import & use              │
│  end-to-end        --steps ensemble  any component             │
│                                      in your code              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Benefits of modularity:**
- Run only what you need
- Easy to extend with new models
- Reuse components in other projects
- Test components independently

---

## Full Configurability

**Everything is configurable** - no code changes required for experimentation:

| What | Where to Change | Examples |
|------|-----------------|----------|
| **Time periods** | `CONFIG["data"]` | Train/valid/test dates |
| **Feature settings** | `CONFIG["features"]` | Rolling windows, assets |
| **Model architecture** | `CONFIG["lstm"]`, etc. | Layers, dropout, epochs |
| **Feature selection** | `CONFIG["shap"]` | Enable SHAP top-N |
| **Ensemble behavior** | `CONFIG["ensemble"]` | Filters, weighting |

### Configuration Locations

```
CLI (PyCharm/Terminal)           Notebook (Colab)
         │                              │
         ▼                              ▼
   src/config.py                    Cell 3
   CONFIG dict                   RUN_PARAMS dict
```

### Hyperparameter Flexibility

All model hyperparameters can be adjusted without touching the code:

| Model | Configurable Parameters |
|-------|------------------------|
| **XGBoost/LightGBM** | learning_rate, max_depth, regularization, early stopping |
| **LSTM/GRU** | units, layers, dropout, batch_size, epochs, patience |
| **Hybrid** | lstm_units, gru_units, architecture settings |
| **HPO** | n_trials, search ranges, sampling strategies |
| **Feature Selection** | thresholds, min_features, SHAP top_n |
| **Ensemble** | method, weights, filter criteria (min_diracc, max_wrmse, top_n) |

**Change once, apply everywhere** - modify the config dict and all relevant code uses the updated values automatically.

---

## Code Snapshot (Reproducibility)

The notebook automatically saves a copy of itself at the start of each run:

```
EXPORT_CODE = True (Cell 3)
         │
         ▼
   runs/{RUN_ID}/code_snapshot/
         ├── google_stock_ml_unified.ipynb
         └── snapshot_meta.json
```

**Saved to both LOCAL and Google Drive** for redundancy.

This ensures every experiment can be exactly reproduced by loading the snapshot from that run.

---

## Rich Feature Engineering

The framework automatically generates features from multiple data sources:

| Category | Features | Source |
|----------|----------|--------|
| **Price Action** | Returns, volatility, momentum | Stock prices |
| **Cross-Asset** | Correlations, beta vs market | SPY, QQQ, NASDAQ |
| **Macro** | Interest rates, inflation | FRED API |
| **Sentiment** | VIX levels, term structure | CBOE |
| **Fundamentals** | EPS surprises | Earnings reports |
| **Calendar** | Day of week, month effects | Date |
| **EU Signals** | Overnight gaps from Europe | DAX |

---

## Quick Start

### Option 1: Full Pipeline (Recommended)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

### Option 2: Specific Steps
```bash
python scripts/run_pipeline.py --steps models,ensemble
```

### Option 3: Google Colab
Upload `notebooks/google_stock_ml_unified.ipynb` and run all cells.

---

## What You Get

### Performance Metrics
- **wRMSE**: Weighted prediction error
- **Directional Accuracy**: % of correct up/down predictions
- **Bootstrap CI**: Confidence intervals for all metrics

### Visual Outputs
- SHAP importance plots
- Prediction vs actual charts
- Model comparison tables

### Saved Artifacts
- Trained models (reusable)
- Feature lists (for production)
- Full predictions (for analysis)

---

## Tuning Guide

### For Different Data Sizes

| Situation | Recommendation |
|-----------|----------------|
| **Small data** (< 500 days) | Reduce model complexity, increase regularization |
| **Large data** (> 2000 days) | Can use larger models, more features |
| **Recent focus** | Use shorter time windows |
| **Long-term patterns** | Use more historical data |

### For Different Goals

| Goal | Configuration Focus |
|------|---------------------|
| **Best accuracy** | Enable all models, use ensemble filtering |
| **Fast iteration** | Run fewer models, shorter HPO |
| **Explainability** | Enable SHAP, use simpler models |
| **Production** | Rigid ensemble, proven features only |

---

## SHAP Feature Selection

An optional but powerful feature that lets the best-performing model guide feature selection:

```
                    Run 1                              Run 2
                      │                                  │
                      ▼                                  ▼
              ┌──────────────┐                  ┌──────────────┐
              │   XGBoost    │                  │  All Models  │
              │   trains     │                  │    use       │
              │   + SHAP     │                  │  SHAP top N  │
              └──────────────┘                  └──────────────┘
                      │                                  │
                      ▼                                  ▼
              Saves top N                        Focused on
              features                           proven features
```

**Enable with**: `CONFIG["shap"]["top_n_features"] = 10`

---

## Project Structure

```
GoogleStockProject/
│
├── src/                     # Core modules (reusable)
│   ├── config.py           # ALL SETTINGS HERE (CLI)
│   ├── features/           # Feature engineering
│   ├── models/             # All model implementations
│   └── utils.py            # Shared utilities
│
├── scripts/
│   ├── run_pipeline.py     # Main entry point
│   └── output/             # Generated at runtime
│       ├── data/           # Downloaded & processed data
│       └── runs/{RUN_ID}/  # Experiment results
│
└── notebooks/
    └── google_stock_ml_unified.ipynb  # Colab version
                                       # Cell 3 = settings
```

---

## Disclaimer

This project is for **research and educational purposes only**. It does not constitute financial advice. Past performance does not guarantee future results.

---

## License

MIT License

---

## Author

Created by **Lea G.** with [Claude Code](https://claude.ai) (Anthropic)
