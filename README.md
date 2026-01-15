# 📈 Google Stock Returns Prediction Framework

**A Multi-Layered Research and Forecasting System for GOOGL Stock**

End-to-end ML pipeline for predicting daily stock log returns, combining tree-based models (XGBoost, LightGBM) with Deep Learning (LSTM, GRU, Hybrid architectures), advanced ensemble methods, and statistical validation via Bootstrap confidence intervals.

---

## 🌟 Key Features

### Feature Engineering
- **Macro indicators**: CPI, Federal Funds Rate, inflation acceleration
- **Market sentiment**: VIX levels, VIX term structure, regime detection
- **Fundamentals**: EPS surprise from earnings reports
- **Cross-asset signals**: Rolling beta and correlation vs SPY, QQQ, NASDAQ
- **EU market gaps**: Pre-market signals from DAX overnight moves (using ^GDAXI Close only)
- **Technical features**: Rolling statistics, volume patterns, momentum

### Feature Selection (Multi-Stage)
1. **Spearman filter**: Remove highly correlated features (ρ > 0.9)
2. **XGBoost importance**: GAIN-based ranking with permutation validation
3. **Mutual Information**: For neural network feature sets

### Models
| Type | Models | Notes |
|------|--------|-------|
| Tree-based | XGBoost, LightGBM | 3-stage HPO (280 trials) |
| Deep Learning | LSTM, GRU | Sequence-based with early stopping |
| Hybrid | Sequential (LSTM→GRU), Parallel (LSTM∥GRU) | Combined architectures |
| Ensemble | Weighted Average, Stacking | Inverse-wRMSE weighting |

### Validation & Explainability
- **Temporal split**: Train → Valid (ES + Score) → Test
- **Bootstrap CI**: 95% confidence intervals (n=1000)
- **SHAP analysis**: Global importance + Beeswarm plots
- **Baselines**: ZERO (predict 0) and NAIVE (last value) benchmarks

---

## 🏗️ Project Structure

The project follows a **modular architecture** where each component (data loading, feature engineering, models, etc.) is a separate module that can be used independently or as part of the full pipeline.

```
GoogleStockProject/
├── src/                          # Core modules
│   ├── config.py                 # CLI configuration (CONFIG dict + setup_cli_paths)
│   ├── utils.py                  # I/O, metrics, statistics
│   ├── data/
│   │   ├── loaders.py            # Data fetching (yfinance, FRED, etc.)
│   │   └── split.py              # Train/Valid/Test splitting
│   ├── features/
│   │   ├── engineering.py        # Feature construction
│   │   └── selection.py          # Feature selection pipeline
│   ├── models/
│   │   ├── xgboost_model.py      # XGBoost training + SHAP
│   │   ├── lightgbm_model.py     # LightGBM training + SHAP
│   │   ├── lstm_gru.py           # LSTM/GRU wrapper
│   │   ├── neural_utils.py       # NN training utilities
│   │   ├── hybrid.py             # Hybrid architectures
│   │   └── ensemble.py           # Ensemble methods
│   └── tuning/
│       └── hpo.py                # Hyperparameter optimization
│
├── scripts/
│   └── run_pipeline.py           # CLI entry point (orchestrates modules)
│
├── output/                       # All outputs (created at runtime)
│   ├── data/
│   │   ├── raw/                  # Downloaded price data
│   │   ├── interim/              # Feature-engineered data
│   │   └── processed/            # Train/Valid/Test splits
│   ├── runs/                     # Experiment snapshots
│   │   └── {RUN_ID}/
│   │       ├── config/           # Run configuration
│   │       ├── feature_selection/
│   │       ├── models/           # Trained models + metrics
│   │       ├── predictions/      # Model predictions
│   │       └── outputs/          # Ensemble results
│   └── results_summary/          # Leaderboards, reports
│
├── notebooks/
│   └── google_stock_ml_unified.ipynb  # Interactive Colab version (self-contained)
│
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### CLI Usage

**Full pipeline:**
```bash
python scripts/run_pipeline.py
```

**Specific steps only:**
```bash
python scripts/run_pipeline.py --steps models,ensemble,summary
```

**Selected models:**
```bash
python scripts/run_pipeline.py --models xgb,lgb,lstm
```

### Google Colab
Upload `notebooks/google_stock_ml_unified.ipynb` to Colab and run all cells.

---

## ⚙️ Configuration

### 📍 Where to Configure

| Environment | Configuration Location | Notes |
|-------------|----------------------|-------|
| **CLI** | `src/config.py` → `CONFIG` dict | Used by `run_pipeline.py` |
| **Notebook** | Cell 3 → `RUN_PARAMS` dict | Self-contained, independent from CLI |

> **Note**: CLI and Notebook configurations are completely separate. Changes in one do not affect the other.

---

### 📅 Date Configuration (Data Split)

The data split is controlled by the following parameters:

```python
"data": {
    "start_date": "2004-09-01",      # First date to download
    "end_date": "2026-01-15",        # Last date (None = today)
    "limit_start_date": "2005-12-31", # First date for modeling (after warmup)
    "train_end": "2020-12-31",       # Last date for training
    "valid_start": "2021-01-01",     # First date for validation
    "valid_end": "2023-12-31",       # Last date for validation
    "test_start": "2023-01-01",      # First date for test
    "test_end": None,                # Last date for test (None = end_date)
}
```

**HPO validation dates** (for hyperparameter tuning):

```python
"hpo": {
    "valid_es_start": "2021-01-01",    # Early stopping validation start
    "valid_es_end": "2021-12-31",      # Early stopping validation end
    "valid_score_start": "2022-01-01", # Scoring validation start
    "valid_score_end": "2023-12-31",   # Scoring validation end
}
```

#### Timeline Visualization

```
|-------- TRAIN --------|--- VALID_ES ---|--- VALID_SCORE ---|---- TEST ----|
2005                   2020            2021               2022            2023→
     limit_start        train_end    valid_es_end    valid_score_end    end_date
```

#### ⚠️ Important Notes

1. **Reproducibility**: Set `end_date` to a fixed date for consistent results across runs
2. **Warmup period**: `limit_start_date` should be after `start_date` to allow rolling feature calculation
3. **No overlap**: Ensure `valid_end` ≥ `valid_start` and `test_start` > `train_end`
4. **HPO validation**: `valid_score_end` is used for model selection during hyperparameter optimization

---

### 🔧 Other Key Settings

#### Feature Exclusions

Some tickers lack reliable Volume data and should be excluded from volume-based features:

```python
"features": {
    "exclude_raw_ohlc": ["^VIX", "^TNX", "^GDAXI"],  # No volume features
}
```

> **Note**: ^GDAXI (DAX index) is still used for EU break close flags (using Close price only), but excluded from volume feature engineering.

#### EU Break Close Configuration

```python
"eu_break_close": {
    "enabled": True,
    "eu_ticker": "^GDAXI",
    "apply_to": "next_us_trading_day",
}
```

---

## 📊 Example Output

```
============================================================
 RESULTS SUMMARY
============================================================
 rank  model                      test_wrmse  test_diracc
    1  Ensemble-weighted_average    0.020029     0.4693
    2  XGBoost                      0.022925     0.4244
    3  GRU                          0.023500     0.5102
    4  Hybrid-Seq                   0.023107     0.4407
    5  LightGBM                     0.023852     0.4431
    6  LSTM                         0.035235     0.5539

Bootstrap 95% CI (n=1000):
  Ensemble: 0.020029 [0.018237, 0.022062]
  XGBoost:  0.022925 [0.020658, 0.025923]
```

---

## 🔍 SHAP Explainability

The framework includes SHAP analysis for model interpretation:

- **Global Importance**: Ranking of most influential features
- **Beeswarm Plots**: Feature value impact on predictions
- **Per-prediction breakdown**: Understanding individual forecasts

Output files:
- `shap_feature_importance.csv`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`

---

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| **wRMSE** | Weighted Root Mean Squared Error (primary metric) |
| **wMAE** | Weighted Mean Absolute Error |
| **DirAcc** | Directional Accuracy (% correct sign predictions) |

Sample weights use exponential decay to emphasize recent observations.

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Missing Volume data for ^GDAXI | Already excluded via `exclude_raw_ohlc` |
| `xgb_selected` features not found | Ensure XGB feature selection ran before neural models |
| Date alignment errors | Check that all date parameters are consistent |

---

## 🤝 Contributing

Contributions welcome:

1. **Alternative data**: Sentiment analysis, news signals
2. **New architectures**: Transformers, Temporal Fusion Transformer
3. **Custom loss functions**: Asymmetric loss for risk management
4. **Backtesting framework**: Strategy simulation

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It does not constitute financial advice or a recommendation to trade securities. Past performance does not guarantee future results.

---

## 📄 License

MIT License

---

## 👩‍💻 Author

Created by **Lea G.**
