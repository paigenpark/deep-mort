"""
Expanding window training pipeline for freeze-and-train ensemble models.

For each cutoff year:
  1. Load full data, filter geos with insufficient history
  2. Split into train/val/test
  3. Train ENSEMBLE_SIZE freeze-and-train ensemble members
  4. Save per-member and combined ensemble predictions

Supports resuming: skips cutoffs where output files already exist.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add paths for imports:
# - code_2026/ for config, data_utils, training_functions_freeze_uncertainty
# - code/ for training_functions
# - repo root for code.uncertainty_models (used by training_functions_freeze_uncertainty)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_repo_root, "code"))
sys.path.insert(0, _repo_root)

from config import (
    CUTOFF_YEARS, ENSEMBLE_SIZE, GEO_DIM, BATCH_SIZE,
    EPOCHS_MEAN, EPOCHS_VAR, RESULTS_DIR,
)
from data_utils import (
    load_full_data, split_by_cutoff, compute_steps_per_epoch,
    filter_geos_with_sufficient_data,
)
from training_functions import prep_data
from training_functions_freeze_uncertainty import run_freeze_ensemble_model
from code.uncertainty_models.training_functions_uncertainty import (
    predict_single_model,
    combine_ensemble_predictions,
)


def output_exists(results_dir, cutoff_year, ensemble_size):
    """Check if all output files for a cutoff already exist."""
    ensemble_file = os.path.join(
        results_dir,
        f"dl_freeze_ensemble_forecast_cutoff{cutoff_year}.txt",
    )
    if not os.path.exists(ensemble_file):
        return False
    for m in range(1, ensemble_size + 1):
        member_file = os.path.join(
            results_dir,
            f"dl_freeze_forecast_{m}_cutoff{cutoff_year}.txt",
        )
        if not os.path.exists(member_file):
            return False
    return True


def save_predictions(data_array, predictions, filepath, n_cols=5):
    """
    Save predictions alongside metadata columns.

    Args:
        data_array: original data with [geo, gender, year, age, rate]
        predictions: dict with keys 'mu', and optionally
                     'total_var', 'aleatoric_var', 'epistemic_var'
        filepath: output path
        n_cols: number of metadata columns from data_array
    """
    n = data_array.shape[0]
    mu = predictions["mu"].flatten()[:n]

    # Build output: geo, gender, year, age, mu, [total_var, aleatoric_var, epistemic_var]
    out = np.column_stack([
        data_array[:, :4],
        mu,
    ])

    if "total_var" in predictions:
        total_var = predictions["total_var"].flatten()[:n]
        aleatoric_var = predictions["aleatoric_var"].flatten()[:n]
        epistemic_var = predictions["epistemic_var"].flatten()[:n]
        out = np.column_stack([out, total_var, aleatoric_var, epistemic_var])

    np.savetxt(filepath, out)
    print(f"  Saved: {filepath}")


def train_cutoff(cutoff_year, full_data, results_dir):
    """Train ensemble for a single cutoff year."""
    print(f"\n{'='*60}")
    print(f"CUTOFF YEAR: {cutoff_year}")
    print(f"{'='*60}")

    # Filter geos with insufficient data for early cutoffs
    data, removed_geos = filter_geos_with_sufficient_data(
        full_data, cutoff_year
    )
    if removed_geos:
        print(f"  Removed {len(removed_geos)} geos with insufficient data")

    # Split data
    train_data, val_data, test_data = split_by_cutoff(data, cutoff_year)
    print(f"  Train: {train_data.shape[0]} rows "
          f"({int(train_data[:, 2].min())}-{int(train_data[:, 2].max())})")
    print(f"  Val:   {val_data.shape[0]} rows "
          f"({int(val_data[:, 2].min())}-{int(val_data[:, 2].max())})")
    print(f"  Test:  {test_data.shape[0]} rows "
          f"({int(test_data[:, 2].min())}-{int(test_data[:, 2].max())})")

    steps_per_epoch = compute_steps_per_epoch(train_data, BATCH_SIZE)
    print(f"  Steps per epoch: {steps_per_epoch}")

    # Prepare tf.data.Datasets
    dataset_train = prep_data(train_data, mode="train")
    dataset_val = prep_data(val_data, mode="test")

    # Prepare non-random datasets for prediction
    dataset_train_pred = prep_data(train_data, mode="not_random")
    dataset_test_pred = prep_data(test_data, mode="not_random")

    # Train ensemble members
    models = []
    for m in range(1, ENSEMBLE_SIZE + 1):
        print(f"\n  --- Ensemble member {m}/{ENSEMBLE_SIZE} ---")

        model, val_loss = run_freeze_ensemble_model(
            dataset_train, dataset_val, GEO_DIM,
            epochs_mean=EPOCHS_MEAN,
            epochs_var=EPOCHS_VAR,
            steps_per_epoch=steps_per_epoch,
            lograte=False,
        )
        models.append(model)
        print(f"  Member {m} val_loss (NLL): {val_loss:.6f}")

        # Save per-member predictions on test set
        mu, variance = predict_single_model(model, dataset_test_pred)
        save_predictions(
            test_data,
            {"mu": mu},
            os.path.join(
                results_dir,
                f"dl_freeze_forecast_{m}_cutoff{cutoff_year}.txt",
            ),
        )

        # Save per-member fitted values on train set
        mu_train, var_train = predict_single_model(model, dataset_train_pred)
        save_predictions(
            train_data,
            {"mu": mu_train},
            os.path.join(
                results_dir,
                f"dl_freeze_fitted_{m}_cutoff{cutoff_year}.txt",
            ),
        )

    # Combine ensemble predictions
    print("\n  Combining ensemble predictions...")

    # Test set ensemble
    ens_mu, ens_var, aleatoric, epistemic = combine_ensemble_predictions(
        models, dataset_test_pred
    )
    save_predictions(
        test_data,
        {
            "mu": ens_mu,
            "total_var": ens_var,
            "aleatoric_var": aleatoric,
            "epistemic_var": epistemic,
        },
        os.path.join(
            results_dir,
            f"dl_freeze_ensemble_forecast_cutoff{cutoff_year}.txt",
        ),
    )

    # Fitted ensemble
    ens_mu_train, ens_var_train, aleatoric_train, epistemic_train = (
        combine_ensemble_predictions(models, dataset_train_pred)
    )
    save_predictions(
        train_data,
        {
            "mu": ens_mu_train,
            "total_var": ens_var_train,
            "aleatoric_var": aleatoric_train,
            "epistemic_var": epistemic_train,
        },
        os.path.join(
            results_dir,
            f"dl_freeze_ensemble_fitted_cutoff{cutoff_year}.txt",
        ),
    )

    print(f"  Cutoff {cutoff_year} complete.")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading full dataset...")
    full_data, geos_key = load_full_data()
    print(f"Full data: {full_data.shape[0]} rows, "
          f"{len(np.unique(full_data[:, 0]))} geographies")

    # Save geos_key for reference
    np.save(os.path.join(RESULTS_DIR, "geos_key.npy"), geos_key)

    for cutoff in CUTOFF_YEARS:
        if output_exists(RESULTS_DIR, cutoff, ENSEMBLE_SIZE):
            print(f"\nSkipping cutoff {cutoff} — output files already exist.")
            continue
        train_cutoff(cutoff, full_data, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("All cutoffs complete.")


if __name__ == "__main__":
    main()
