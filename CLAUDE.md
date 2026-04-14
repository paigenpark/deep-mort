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

### Expanding Window Pipeline (code_2026/)
```bash
python code_2026/config.py                       # Generate config.json for R scripts
python code_2026/train_expanding_window.py        # Train DL ensembles for all cutoff years
Rscript code_2026/benchmark_models/lee-carter.r   # Lee-Carter for all cutoffs
Rscript code_2026/benchmark_models/hyndman-ullah.R # Hyndman-Ullah for all cutoffs
Rscript code_2026/benchmark_models/coherent.r     # Coherent for all cutoffs
python code_2026/evaluate_expanding_window.py     # Evaluate all models across windows
```

### Languages
- **Python**: DL models, data splitting, evaluation, figures
- **R**: Data preparation from raw sources, benchmark demographic models
