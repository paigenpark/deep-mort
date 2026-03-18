# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-Mort is a research project investigating why simple MLP deep learning models outperform traditional time-series approaches (Lee-Carter, Hyndman-Ullah, Coherent) for mortality rate forecasting. Uses data from the Human Mortality Database (HMD) and U.S. Mortality Database (USMDB).

## Active Development (2026 Revision)

Active work is in `code_2026/`, which is a revised version of the original `code/` directory for journal submission. Key changes in this revision include adding an expanding window evaluation scheme and uncertainty analysis. The original `code/` directory is kept for reference but should not be the focus of new work.

## Commands

### Environment Setup
```bash
pip install -r requirements.txt    # Python dependencies (TensorFlow 2.17, Keras 3.4)
# R: renv::restore()               # R dependencies via renv.lock
```

### Running the Full Pipeline
```bash
cd code && python start_to_finish.py
```
This runs data preparation (R + Python), benchmark models, DL training, and figure generation end-to-end. DL training is compute-intensive — cloud computing recommended.

### Running Individual Steps
- **Train DL models**: Run `code/train_dl_models.ipynb` (basic) or `code/train_dl_models_uncertainty.ipynb` (ensemble with uncertainty)
- **Evaluate uncertainty**: Run `code/evaluation_uncertainty.ipynb`
- **Generate figures**: Run `code/create_robust_figures.ipynb`
- **Benchmark models**: R scripts in `code/benchmark_models/` (lee-carter.r, hyndman-ullah.R, coherent.r)
- **Data prep**: R scripts in `code/data_preparation/`, then `code/data_preparation/split_data.py`

### Expanding Window Pipeline (code_2026/)
```bash
python code_2026/config.py                       # Generate config.json for R scripts
python code_2026/train_expanding_window.py        # Train DL ensembles for all cutoff years
Rscript code_2026/benchmark_models/lee-carter.r   # Lee-Carter for all cutoffs
Rscript code_2026/benchmark_models/hyndman-ullah.R # Hyndman-Ullah for all cutoffs
Rscript code_2026/benchmark_models/coherent.r     # Coherent for all cutoffs
python code_2026/evaluate_expanding_window.py     # Evaluate all models across windows
```

## Architecture

### DL Model Structure (defined in `code/training_functions.py` and `training_functions_uncertainty.py`)
- **Inputs**: Year (normalized), Age, Geography, Gender
- **Embedding layers**: Age (5D), Gender (5D), Geography (5D)
- **4 dense layers**: 128 units, tanh activation, batch normalization, 0.05 dropout
- **Skip connection**: Input features concatenated before final hidden layer
- **Two output variants**: Sigmoid (rate in [0,1]) or linear (log-rate)
- **Training**: MSE loss (basic) or Gaussian NLL loss (ensemble/uncertainty), batch size 256

### Uncertainty Quantification (ensemble models)
- Deep ensemble members output [μ, raw_variance]
- Uncertainty decomposed into aleatoric (average predicted variance) and epistemic (variance of ensemble means)

### Data Splits
- **Original (code/)**: Training: 1959–2005, Validation: 2005–2015, Final Test: 2015–2019
- **Expanding window (code_2026/)**: Configurable cutoffs [1985, 1990, 1995, 2000, 2005, 2010, 2015]. For each cutoff: train up to cutoff-5, val last 5 years, test cutoff+1 to 2019
- Year normalized as (year - 1959) / 60 (fixed across all windows), rates capped at 1.0

### Key Modules
- `code/training_functions.py` — Model architecture, data loading, training loops
- `code/training_functions_uncertainty.py` — Ensemble training with NLL loss
- `code/evaluation_functions.py` — Error metrics (MSE, RMSE, RRMSE)
- `code/start_to_finish.py` — End-to-end pipeline orchestration
- `code_2026/config.py` — Central configuration for expanding window pipeline
- `code_2026/data_utils.py` — Data loading and splitting for expanding windows
- `code_2026/train_expanding_window.py` — DL training across all cutoff years
- `code_2026/evaluate_expanding_window.py` — Multi-window evaluation and aggregation

### Languages
- **Python**: DL models, data splitting, evaluation, figures
- **R**: Data preparation from raw sources, benchmark demographic models
