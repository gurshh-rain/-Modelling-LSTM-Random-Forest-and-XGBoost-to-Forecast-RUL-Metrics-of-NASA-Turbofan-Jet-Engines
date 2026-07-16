# Predictive Maintenance: Remaining Useful Life (RUL) of NASA Turbofan Jet Engines

Predicts the Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS dataset, comparing Random Forest, XGBoost, LSTM, and an LSTM+XGBoost ensemble.

## Overview

This notebook tackles a predictive maintenance problem: given sensor readings and operational settings collected over an engine's operating cycles, estimate how many cycles remain before it fails. It progressively builds up model sophistication — starting from tree-based baselines, adding time-aware feature engineering, then a sequence-based deep learning model (LSTM), and finally an optimized ensemble of the two best approaches.

## Dataset

Uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001 subset. Expected files (place alongside the notebook):

- `train_FD001.txt` — training run-to-failure trajectories
- `test_FD001.txt` — test trajectories (truncated before failure)
- `RUL_FD001.txt` — true RUL values for each test engine

**Source:** [NASA Prognostics Center of Excellence Data Repository](https://data.nasa.gov/dataset/C-MAPSS-Jet-Engine-Simulated-Data/xaut-bemq) / commonly redistributed via Kaggle.

**Columns:** `unit number`, `time in cycles`, 3 operational settings, and 21 sensor measurements.

**Target:** `RUL` (Remaining Useful Life), computed as each engine's final cycle minus the current cycle, clipped at an upper bound of 125 (a standard practice for this dataset, since RUL degradation is roughly linear only near end-of-life).

## Requirements

```
pandas
matplotlib
seaborn
numpy
scipy
scikit-learn
xgboost
torch
```

Install with:
```bash
pip install pandas matplotlib seaborn numpy scipy scikit-learn xgboost torch
```

A CUDA-capable GPU is optional but recommended for the LSTM training section (falls back to CPU automatically).

## Workflow

1. **Load & label data** — Read raw space-delimited sensor logs, compute RUL per row per engine, clip at 125.
2. **Feature selection** — Correlation matrix against RUL; keep features with `|correlation| > 0.04`.
3. **Random Forest baseline**
   - `GridSearchCV` and `RandomizedSearchCV` (both with `GroupKFold` grouped by engine unit, to prevent leakage across an engine's own cycles)
   - Evaluated on held-out test engines
4. **XGBoost (raw features)** — Same grouped CV strategy, grid-searched over depth, learning rate, subsampling, etc.
5. **XGBoost + engineered features** — Adds rolling mean, rolling std, and trend (deviation from rolling mean) over a 20-cycle window per sensor, then re-tunes via randomized search.
6. **LSTM** — Sequence model (2-layer LSTM + fully connected head) trained on fixed-length (30-cycle) sliding windows per engine, using PyTorch.
7. **Ensemble** — Blends XGBoost and LSTM predictions; first a naive 50/50 blend, then a weighting optimized via `scipy.optimize.minimize_scalar` to minimize RMSE.

## Key Techniques

- **Group-aware cross-validation** (`GroupKFold` by `unit number`) to avoid training and validating on cycles from the same engine.
- **RUL clipping** at 125 cycles, standard for C-MAPSS to avoid penalizing models for early-life predictions where degradation isn't yet linear.
- **Rolling-window feature engineering** to give row-wise models (XGBoost) some sense of trend/history despite not natively processing sequences.
- **Sequence modeling with LSTM** to directly capture temporal degradation patterns.
- **Weighted ensembling** with a numerically optimized blend weight rather than a fixed split.

## Results Summary

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (GridSearchCV) | 12.32 | 17.33 | 0.813 |
| Random Forest (RandomizedSearchCV) | 12.33 | 17.43 | 0.810 |
| XGBoost (raw features) | 11.89 | 16.89 | 0.822 |
| XGBoost (+ rolling/trend features) | 11.86 | 16.47 | 0.831 |
| LSTM (30-cycle window) | 9.54 | 12.72 | 0.899 |
| Ensemble (50/50 LSTM + XGBoost) | 9.85 | 13.34 | 0.889 |
| **Ensemble (optimized weighting)** | **9.40** | **12.59** | **0.901** |

The optimized-weight ensemble of LSTM and XGBoost gives the best overall performance, with the LSTM alone as the strongest single model.

## Usage

1. Download the C-MAPSS FD001 files and place them alongside the notebook.
2. Run cells sequentially — later sections (LSTM, ensembling) depend on variables (`xgb_fe_prod`, `test_loader`, `model`) defined earlier.
3. Note: cell 34 references `RandomizedSearch` (incomplete name) for the XGBoost + engineered-features tuning step — confirm this resolves to `RandomizedSearchCV` before running, as it currently appears cut off.
4. Review the correlation heatmap, ensemble weight-search plot, and final metrics table to compare models.

## Notes / Caveats

- The notebook is written for the FD001 subset only (single operating condition, single fault mode); results won't directly generalize to FD002–FD004 subsets without adjustment.
- Test evaluation uses each test engine's **last available cycle only** (`test_last`), consistent with the standard C-MAPSS evaluation protocol.
- The LSTM section requires GPU/CPU-agnostic PyTorch setup (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`), and its data prep function (`prepare_data_for_lstm`) is truncated in the extracted source — verify this cell is complete before running end-to-end.
- Hyperparameter search grids are fairly wide (especially for XGBoost + engineered features), so full re-runs may take significant time; consider reducing `n_iter`/grid size for quick iteration.
