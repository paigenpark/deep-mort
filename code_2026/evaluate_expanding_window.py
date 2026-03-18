"""
Multi-window evaluation for the expanding window pipeline.

For each cutoff year:
  - Load DL ensemble forecasts and benchmark forecasts
  - Load actual test data
  - Compute metrics: RMSE, MAE, NLL, 95% coverage, interval width
  - Aggregate across windows

Outputs CSV files with per-window and aggregate results.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure code_2026/ is on the path for config/data_utils/evaluation_functions
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from config import (
    CUTOFF_YEARS, RESULTS_DIR, DATA_DIR, MIN_YEAR, MAX_YEAR,
)
from data_utils import load_full_data, split_by_cutoff, filter_geos_with_sufficient_data
from evaluation_functions import calculate_error, calculate_error_by_category


# --- Uncertainty metrics ---

def compute_nll(actual, mu, variance):
    """Gaussian negative log-likelihood."""
    variance = np.maximum(variance, 1e-10)
    nll = 0.5 * np.log(variance) + 0.5 * (actual - mu) ** 2 / variance
    return np.mean(nll)


def compute_coverage(actual, lower, upper):
    """Fraction of actual values falling within [lower, upper]."""
    covered = (actual >= lower) & (actual <= upper)
    return np.mean(covered)


def compute_interval_width(lower, upper):
    """Mean width of prediction intervals."""
    return np.mean(upper - lower)


def load_dl_ensemble(results_dir, cutoff_year):
    """
    Load DL ensemble forecast for a cutoff year.
    Returns array with columns: [geo, gender, year, age, mu, total_var, aleatoric_var, epistemic_var]
    """
    filepath = os.path.join(
        results_dir,
        f"dl_freeze_ensemble_forecast_cutoff{cutoff_year}.txt",
    )
    if not os.path.exists(filepath):
        return None
    return np.loadtxt(filepath)


def load_benchmark(data_dir, model_name, cutoff_year):
    """
    Load benchmark forecast for a cutoff year.
    Returns array with columns matching the R output format:
    [geo, gender, year, age, mu, total_var, aleatoric_var, epistemic_var, lower_95, upper_95]
    """
    filepath = os.path.join(
        data_dir,
        f"{model_name}_forecast_cutoff{cutoff_year}.csv",
    )
    if not os.path.exists(filepath):
        return None
    return np.genfromtxt(filepath, delimiter=",")


def evaluate_single_window(actual_data, forecast_data, model_name,
                           cutoff_year, has_uncertainty=False):
    """
    Evaluate a single model on a single cutoff window.

    Args:
        actual_data: [geo, gender, year, age, rate]
        forecast_data: array with at least [geo, gender, year, age, mu]
        model_name: string label
        cutoff_year: int
        has_uncertainty: if True, expects variance columns

    Returns:
        dict of metrics
    """
    # Build arrays for calculate_error: [geo, gender, year, age, rate]
    forecast_for_error = np.column_stack([
        forecast_data[:, :4],
        forecast_data[:, 4],  # mu column
    ])

    mse, rmse, rrmse = calculate_error(forecast_for_error, actual_data)

    # MAE
    # Use aligned data for MAE computation
    common_keys = set(map(tuple, forecast_for_error[:, :4])) & set(
        map(tuple, actual_data[:, :4])
    )
    filtered_fc = np.array(
        [row for row in forecast_for_error if tuple(row[:4]) in common_keys]
    )
    filtered_actual = np.array(
        [row for row in actual_data if tuple(row[:4]) in common_keys]
    )
    mae = np.mean(np.abs(filtered_fc[:, 4] - filtered_actual[:, 4]))

    result = {
        "model": model_name,
        "cutoff_year": cutoff_year,
        "test_years": f"{cutoff_year + 1}-{MAX_YEAR}",
        "n_test_rows": actual_data.shape[0],
        "mse": mse,
        "rmse": rmse,
        "rrmse": rrmse,
        "mae": mae,
    }

    if has_uncertainty and forecast_data.shape[1] >= 6:
        mu = filtered_fc[:, 4].astype(float)
        actual_rates = filtered_actual[:, 4].astype(float)

        # Total variance is column 5
        fc_with_var = np.array(
            [row for row in forecast_data if tuple(row[:4]) in common_keys]
        )
        total_var = fc_with_var[:, 5].astype(float)

        result["nll"] = compute_nll(actual_rates, mu, total_var)

        # 95% prediction intervals
        std = np.sqrt(np.maximum(total_var, 0))
        lower = mu - 1.96 * std
        upper = mu + 1.96 * std
        result["coverage_95"] = compute_coverage(actual_rates, lower, upper)
        result["interval_width"] = compute_interval_width(lower, upper)

    return result


def evaluate_by_category(actual_data, forecast_data, feature_index,
                         category_name):
    """Compute RMSE by category (age, gender, geo)."""
    forecast_for_error = np.column_stack([
        forecast_data[:, :4],
        forecast_data[:, 4],
    ])
    _, rmses, _ = calculate_error_by_category(
        forecast_for_error, actual_data, feature_index
    )
    return {f"{category_name}_{int(k)}": v for k, v in rmses.items()}


def main():
    print("Loading full dataset...")
    full_data, _ = load_full_data()

    all_results = []
    benchmark_models = ["lc", "hu", "coherent"]

    for cutoff in CUTOFF_YEARS:
        print(f"\n--- Evaluating cutoff {cutoff} ---")

        # Get test data for this window
        data, removed_geos = filter_geos_with_sufficient_data(
            full_data, cutoff
        )
        _, _, test_data = split_by_cutoff(data, cutoff)

        if test_data.shape[0] == 0:
            print(f"  No test data for cutoff {cutoff}, skipping.")
            continue

        print(f"  Test data: {test_data.shape[0]} rows "
              f"({int(test_data[:, 2].min())}-{int(test_data[:, 2].max())})")

        # Evaluate DL ensemble
        dl_data = load_dl_ensemble(RESULTS_DIR, cutoff)
        if dl_data is not None:
            result = evaluate_single_window(
                test_data, dl_data, "DL_Ensemble", cutoff,
                has_uncertainty=True,
            )
            all_results.append(result)
            print(f"  DL_Ensemble RMSE: {result['rmse']:.6f}")
        else:
            print(f"  DL ensemble forecast not found for cutoff {cutoff}")

        # Evaluate benchmark models
        for bm in benchmark_models:
            bm_data = load_benchmark(DATA_DIR, bm, cutoff)
            if bm_data is not None:
                has_unc = bm_data.shape[1] >= 10  # has lower/upper columns
                result = evaluate_single_window(
                    test_data, bm_data, bm.upper(), cutoff,
                    has_uncertainty=has_unc,
                )
                all_results.append(result)
                print(f"  {bm.upper()} RMSE: {result['rmse']:.6f}")
            else:
                print(f"  {bm} forecast not found for cutoff {cutoff}")

    if not all_results:
        print("\nNo results to save.")
        return

    # Per-window results
    df = pd.DataFrame(all_results)
    per_window_path = os.path.join(RESULTS_DIR, "metrics_per_window.csv")
    df.to_csv(per_window_path, index=False)
    print(f"\nPer-window metrics saved to {per_window_path}")

    # Aggregate across windows (mean ± std of each metric)
    metric_cols = ["mse", "rmse", "rrmse", "mae", "nll",
                   "coverage_95", "interval_width"]
    existing_metrics = [c for c in metric_cols if c in df.columns]

    agg = df.groupby("model")[existing_metrics].agg(["mean", "std"])
    agg.columns = ["_".join(col) for col in agg.columns]
    agg_path = os.path.join(RESULTS_DIR, "metrics_aggregate.csv")
    agg.to_csv(agg_path)
    print(f"Aggregate metrics saved to {agg_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean ± std across windows)")
    print("=" * 70)
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        for metric in existing_metrics:
            if metric in model_df.columns and model_df[metric].notna().any():
                mean = model_df[metric].mean()
                std = model_df[metric].std()
                print(f"  {metric:>20s}: {mean:.6f} ± {std:.6f}")


if __name__ == "__main__":
    main()
