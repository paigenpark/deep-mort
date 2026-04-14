"""Central configuration for the expanding window evaluation pipeline."""

import json
import os

# --- Cutoff years for expanding window ---
CUTOFF_YEARS = [1985, 1990, 1995, 2000, 2005, 2009]
VALIDATION_WINDOW = 10  # years reserved from end of training for validation

# --- Data boundaries ---
MIN_YEAR = 1959
MAX_YEAR = 2019

# --- Year normalization (fixed across all windows) ---
YEAR_NORM_DIVISOR = 60  # (year - MIN_YEAR) / YEAR_NORM_DIVISOR

# --- Model hyperparameters ---
GEO_DIM = 90        # number of unique geographies (50 states + 40 countries)
ENSEMBLE_SIZE = 5
BATCH_SIZE = 256
EPOCHS_MEAN = 30     # Phase 1: MSE training
EPOCHS_VAR = 10      # Phase 2: variance head training

# --- Paths (relative to repo root) ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results_2026")

# Countries removed during data loading (same as split_data.py)
COUNTRIES_TO_REMOVE = [
    "CHL", "DEUTNP", "FRACNP", "GBRCENW", "GBR_NP",
    "HKG", "HRV", "KOR", "NZL_MA", "NZL_NM",
]


def export_config_json(output_path=None):
    """Export config values needed by R scripts as JSON."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "benchmark_models", "config.json",
        )
    config = {
        "cutoff_years": CUTOFF_YEARS,
        "validation_window": VALIDATION_WINDOW,
        "min_year": MIN_YEAR,
        "max_year": MAX_YEAR,
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {output_path}")
    return output_path


if __name__ == "__main__":
    export_config_json()
