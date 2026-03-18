"""Data loading and splitting utilities for the expanding window pipeline."""

import csv
import numpy as np
import os
import sys

# Ensure config is importable when running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, MIN_YEAR, MAX_YEAR, COUNTRIES_TO_REMOVE, BATCH_SIZE,
)


def load_full_data(data_dir=None):
    """
    Load HMD + USMDB data and combine into a single array.

    Replicates the logic from data_preparation/split_data.py (lines 8-94):
    - Load usmdb.csv → state_data with geo indices 0..49
    - Load hmd.csv → country_data with geo indices 50+
    - Filter: age <= 99, valid rates, cap rates at 1.0

    Returns:
        combined: np.ndarray of shape (N, 5) with columns
                  [geo, gender, year, age, rate]
        geos_key: np.ndarray mapping geo names to indices
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # --- USMDB ---
    states = []
    genders = []
    state_rows = []

    with open(os.path.join(data_dir, "usmdb.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue
            state, gender, year, age, rate = row
            year = int(year)
            try:
                age = int(age)
            except ValueError:
                age = -1
            if state not in states:
                states.append(state)
            state_idx = states.index(state)
            if gender not in genders:
                genders.append(gender)
            gender_idx = genders.index(gender)
            try:
                rate = float(rate)
            except ValueError:
                rate = -1
            if rate > 1:
                rate = 1.0
            if age != -1 and rate != -1 and age <= 99:
                state_rows.append([state_idx, gender_idx, year, age, rate])

    state_data = np.array(state_rows)

    # --- HMD ---
    countries = []
    country_rows = []

    with open(os.path.join(data_dir, "hmd.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue
            country, gender, year, age, rate = row
            if country in COUNTRIES_TO_REMOVE:
                continue
            year = int(year)
            try:
                age = int(age)
            except ValueError:
                age = -1
            if country not in countries:
                countries.append(country)
            country_idx = countries.index(country)
            if gender not in genders:
                genders.append(gender)
            gender_idx = genders.index(gender)
            try:
                rate = float(rate)
            except ValueError:
                rate = -1
            if rate > 1:
                rate = 1.0
            if age != -1 and rate != -1 and age <= 99:
                country_rows.append([country_idx, gender_idx, year, age, rate])

    country_data = np.array(country_rows)

    # Offset country geo indices so they don't overlap with states
    country_data[:, 0] = country_data[:, 0] + len(states)

    # Build geos_key
    geos_list = states + countries
    geos_index = np.arange(len(geos_list))
    geos_key = np.column_stack((np.array(geos_list), geos_index))

    combined = np.vstack((state_data, country_data))

    return combined, geos_key


def split_by_cutoff(data, cutoff_year, validation_window=5):
    """
    Split data into train/val/test using an expanding window.

    Args:
        data: np.ndarray with columns [geo, gender, year, age, rate]
        cutoff_year: last year included in the training+validation portion
        validation_window: number of years at the end of training portion
                           reserved for validation

    Returns:
        train: years in [MIN_YEAR, cutoff_year - validation_window]
        val:   years in (cutoff_year - validation_window, cutoff_year]
        test:  years in (cutoff_year, MAX_YEAR]
    """
    years = data[:, 2]

    val_start = cutoff_year - validation_window

    train_mask = (years >= MIN_YEAR) & (years <= val_start)
    val_mask = (years > val_start) & (years <= cutoff_year)
    test_mask = (years > cutoff_year) & (years <= MAX_YEAR)

    return data[train_mask], data[val_mask], data[test_mask]


def compute_steps_per_epoch(train_data, batch_size=BATCH_SIZE):
    """Compute steps_per_epoch as ceil(n_train / batch_size)."""
    return int(np.ceil(train_data.shape[0] / batch_size))


def filter_geos_with_sufficient_data(data, cutoff_year, min_years=10):
    """
    Filter out geographies that have fewer than min_years of data
    up to the cutoff year. Important for early cutoffs (e.g. 1985).

    Returns:
        filtered_data: data with insufficient geos removed
        removed_geos: set of geo indices that were removed
    """
    train_portion = data[data[:, 2] <= cutoff_year]
    geos = np.unique(data[:, 0])
    removed_geos = set()

    for geo in geos:
        geo_data = train_portion[train_portion[:, 0] == geo]
        n_years = len(np.unique(geo_data[:, 2]))
        if n_years < min_years:
            removed_geos.add(geo)

    if removed_geos:
        mask = ~np.isin(data[:, 0], list(removed_geos))
        return data[mask], removed_geos
    return data, removed_geos
