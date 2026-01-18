# Results Summary

**Last Updated:** 2026-01-18 11:23

**Total Runs:** 6 | **Total Configurations:** 82

---

## 🏆 Best Results (Top 10)

| # | Model | Feature Set | wRMSE | DirAcc | Run |
|---|-------|-------------|-------|--------|-----|
| 1 | Ensemble-weighted_average | all_models | 0.017904 | 54.67% | 111753 |
| 2 | Ensemble-weighted_average | all_models | 0.017904 | 54.67% | 092009 |
| 3 | Ensemble-weighted_average | all_models | 0.017942 | 56.85% | 213310 |
| 4 | Ensemble-weighted_average | all_models | 0.017945 | 56.85% | 205014 |
| 5 | Ensemble-weighted_average | all_models | 0.017986 | 57.05% | 211034 |
| 6 | Ensemble-weighted_average | all_models | 0.019100 | 52.93% | 222938 |
| 7 | XGBoost | xgb_selected | 0.019834 | 55.56% | 111753 |
| 8 | XGBoost | xgb_selected | 0.019834 | 55.56% | 092009 |
| 9 | LightGBM | xgb_selected | 0.019897 | 54.32% | 111753 |
| 10 | LightGBM | xgb_selected | 0.019897 | 54.32% | 092009 |

---

## 📊 Best per Model Type

| # | Model | Feature Set | wRMSE | DirAcc |
|---|-------|-------------|-------|--------|
| 1 | Ensemble-weighted_average | all_models | 0.017904 | 54.67% |
| 2 | XGBoost | xgb_selected | 0.019834 | 55.56% |
| 3 | LightGBM | xgb_selected | 0.019897 | 54.32% |
| 4 | BASELINE_ZERO | baseline | 0.019984 | 1.23% |
| 5 | Hybrid-Seq | neural_40 | 0.020478 | 40.00% |
| 6 | GRU | neural_80 | 0.020533 | 40.00% |
| 7 | Hybrid-Par | neural_40 | 0.020553 | 40.00% |
| 8 | LSTM | xgb_selected | 0.020700 | 56.43% |
| 9 | BASELINE_NAIVE | baseline | 0.025313 | 48.15% |

---

## 🥇 Overall Best

| Metric | Value |
|--------|-------|
| Model | **Ensemble-weighted_average** |
| Feature Set | all_models |
| wRMSE | 0.017904 |
| DirAcc | 54.67% |
| Run ID | 20260118_111753 |
| Data Range | 2023-11-20 → 2026-01-15 |

---

## 📖 Metrics

| Metric | Description |
|--------|-------------|
| wRMSE | Weighted Root Mean Squared Error (↓ lower is better) |
| DirAcc | Directional Accuracy (↑ higher is better) |

*Full details with period configurations available in CSV files.*

---

## Bootstrap Confidence Intervals (95%)

| Model | wRMSE (95% CI) | wMAE (95% CI) | DirAcc (95% CI) | N |
|-------|----------------|---------------|-----------------|---|
| Ensemble-weighted_average | 0.017904 [0.014979, 0.020914] | 0.014145 [0.011646, 0.017051] | 0.5467 [0.4267, 0.6533] | 75 |
| XGBoost | 0.019834 [0.016019, 0.023963] | 0.015627 [0.012810, 0.018919] | 0.5556 [0.4444, 0.6543] | 81 |
| LightGBM | 0.019897 [0.016028, 0.024063] | 0.015668 [0.012872, 0.018962] | 0.5432 [0.4198, 0.6420] | 81 |
| BASELINE_ZERO | 0.019984 [0.016107, 0.024175] | 0.015729 [0.012911, 0.019033] | 0.0123 [0.0000, 0.0373] | 81 |
| Hybrid-Seq | 0.020478 [0.016313, 0.024785] | 0.016182 [0.012917, 0.019459] | 0.4000 [0.2933, 0.5067] | 75 |
| GRU | 0.020533 [0.016364, 0.024882] | 0.016229 [0.012984, 0.019523] | 0.4000 [0.2933, 0.5067] | 75 |
| Hybrid-Par | 0.020553 [0.016365, 0.024909] | 0.016245 [0.013013, 0.019546] | 0.4000 [0.2933, 0.5067] | 75 |
| BASELINE_NAIVE | 0.025313 [0.021745, 0.028979] | 0.020068 [0.016762, 0.023692] | 0.4815 [0.3824, 0.5926] | 81 |
| LSTM | 0.037176 [0.034422, 0.040152] | 0.030268 [0.028174, 0.032678] | 0.5581 [0.5124, 0.5996] | 482 |

---

## Tomorrow Predictions (Next Trading Day)

| Rank | Model | Feature Set | Pred Return % |
|------|-------|-------------|---------------|
| 1 | XGBoost | xgb_selected | +0.0642% |
| 2 | Ensemble | all_models | +0.0279% |
| 3 | LightGBM | xgb_selected | -0.0085% |