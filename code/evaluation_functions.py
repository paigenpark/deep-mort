import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as plt
tfkl = tf.keras.layers

def calculate_mse_by_category_lc(forecasted_data, actual_data, feature_index):
    """
    Calculate the Mean Squared Error (MSE) between the forecasted and actual mortality rates.

    
    Args:
        forecasted_data (numpy.ndarray): A 2D array with 5 columns representing geo, gender, year, age, and forecasted mortality rate.
        actual_data (numpy.ndarray): A 2D array with 5 columns representing geo, gender, year, age, and actual mortality rate.
        
    Returns:
        float: The Mean Squared Error (MSE) between the forecasted and actual mortality rates by category.
    """

    # Ensure both arrays are sorted by geo, gender, year, and age
    forecasted_data = forecasted_data[np.lexsort((forecasted_data[:, 3], forecasted_data[:, 2], forecasted_data[:, 1], forecasted_data[:, 0]))]
    actual_data = actual_data[np.lexsort((actual_data[:, 3], actual_data[:, 2], actual_data[:, 1], actual_data[:, 0]))]

    # Find common geo/gender/year/age combinations between forecasted and actual rates
    common_keys = set(map(tuple, forecasted_data[:, :4])) & set(map(tuple, actual_data[:, :4]))

    # Filter both forecasted and actual rates based on common combinations
    filtered_forecasted = np.array([row for row in forecasted_data if tuple(row[:4]) in common_keys])
    filtered_actual = np.array([row for row in actual_data if tuple(row[:4]) in common_keys])

    categories = np.unique(filtered_forecasted[:, feature_index].astype(int))
    
    mses_by_category = {}

    for category in categories:
        forecasted = filtered_forecasted[filtered_forecasted[:, feature_index] == category]
        actual = filtered_actual[filtered_actual[:, feature_index] == category]

        forecasted_rates = forecasted[:, 4].astype(float)
        actual_rates = actual[:, 4].astype(float)

        mses_by_category[category] = np.mean((forecasted_rates - actual_rates) ** 2)
        
    return mses_by_category

def calculate_mse_by_category(val_data, model, feature_index):
    predictions = []
    true_values = []
    feature_values = []

    for X_batch, y_batch in val_data:
        preds_batch = model(X_batch, training=False)
        predictions.append(preds_batch)
        true_values.append(y_batch)
        feature_values.append(X_batch[feature_index])

    # concatenates the list of tensors created above 
    predictions = tf.concat(predictions, axis=0).numpy()
    true_values = tf.concat(true_values, axis=0).numpy()
    feature_values = tf.concat(feature_values, axis=0).numpy()

    unique_feature_values = np.unique(feature_values)

    mse_by_category = {}

    for value in unique_feature_values:
        value = value.astype(int)
        idx = np.where(feature_values == value)
        filtered_preds = predictions[idx]
        filtered_true_values = true_values[idx]

        mse = np.mean((filtered_preds - filtered_true_values) ** 2)

        mse_by_category[value] = mse

    return mse_by_category

def calculate_rmae_by_category(val_data, model, feature_index):
    predictions = []
    true_values = []
    feature_values = []

    for X_batch, y_batch in val_data:
        preds_batch = model(X_batch, training=False)
        predictions.append(preds_batch)
        true_values.append(y_batch)
        feature_values.append(X_batch[feature_index])

    # concatenates the list of tensors created above 
    predictions = tf.concat(predictions, axis=0).numpy()
    true_values = tf.concat(true_values, axis=0).numpy()
    feature_values = tf.concat(feature_values, axis=0).numpy()

    unique_feature_values = np.unique(feature_values)

    mse_by_category = {}

    for value in unique_feature_values:
        value = value.astype(int)
        idx = np.where(feature_values == value)
        filtered_preds = predictions[idx]
        filtered_true_values = true_values[idx]

        mse = np.mean(np.abs(filtered_preds - filtered_true_values)) / np.mean(filtered_true_values)

        mse_by_category[value] = mse

    return mse_by_category

def calculate_rmae_by_category_lc(forecasted_data, actual_data, feature_index):
    """
    Calculate the Mean Squared Error (MSE) between the forecasted and actual mortality rates.

    
    Args:
        forecasted_data (numpy.ndarray): A 2D array with 5 columns representing geo, gender, year, age, and forecasted mortality rate.
        actual_data (numpy.ndarray): A 2D array with 5 columns representing geo, gender, year, age, and actual mortality rate.
        
    Returns:
        float: The Mean Squared Error (MSE) between the forecasted and actual mortality rates by category.
    """

    # Ensure both arrays are sorted by geo, gender, year, and age
    forecasted_data = forecasted_data[np.lexsort((forecasted_data[:, 3], forecasted_data[:, 2], forecasted_data[:, 1], forecasted_data[:, 0]))]
    actual_data = actual_data[np.lexsort((actual_data[:, 3], actual_data[:, 2], actual_data[:, 1], actual_data[:, 0]))]

    # Find common geo/gender/year/age combinations between forecasted and actual rates
    common_keys = set(map(tuple, forecasted_data[:, :4])) & set(map(tuple, actual_data[:, :4]))

    # Filter both forecasted and actual rates based on common combinations
    filtered_forecasted = np.array([row for row in forecasted_data if tuple(row[:4]) in common_keys])
    filtered_actual = np.array([row for row in actual_data if tuple(row[:4]) in common_keys])

    categories = np.unique(filtered_forecasted[:, feature_index].astype(int))
    
    mses_by_category = {}

    for category in categories:
        forecasted = filtered_forecasted[filtered_forecasted[:, feature_index] == category]
        actual = filtered_actual[filtered_actual[:, feature_index] == category]

        forecasted_rates = forecasted[:, 4].astype(float)
        actual_rates = actual[:, 4].astype(float)

        mses_by_category[category] = np.mean(np.abs(forecasted_rates - actual_rates)) / np.mean(actual_rates)
        
    return mses_by_category


def calculate_relative_rmse(forecasted_rates, actual_rates):
    """
    Calculate the Relative Root Mean Squared Error (RRMSE) between the forecasted and actual mortality rates.
    
    Args:
        forecasted_rates (numpy.ndarray): A 2D array with 5 columns representing state, gender, year, age, and forecasted mortality rate.
        actual_rates (numpy.ndarray): A 2D array with 5 columns representing state, gender, year, age, and actual mortality rate.
        
    Returns:
        float: The Relative Root Mean Squared Error (RRMSE) between the forecasted and actual mortality rates.
    """

    # Ensure both arrays are sorted by geo, gender, year, and age
    forecasted_rates = forecasted_rates[np.lexsort((forecasted_rates[:, 3], forecasted_rates[:, 2], forecasted_rates[:, 1], forecasted_rates[:, 0]))]
    actual_rates = actual_rates[np.lexsort((actual_rates[:, 3], actual_rates[:, 2], actual_rates[:, 1], actual_rates[:, 0]))]

    # Find common geo/gender/year/age combinations between forecasted and actual rates
    common_keys = set(map(tuple, forecasted_rates[:, :4])) & set(map(tuple, actual_rates[:, :4]))

    # Filter both forecasted and actual rates based on common combinations
    filtered_forecasted = np.array([row for row in forecasted_rates if tuple(row[:4]) in common_keys])
    filtered_actual = np.array([row for row in actual_rates if tuple(row[:4]) in common_keys])

    # Extract the forecasted and actual mortality rates
    forecasted_values = filtered_forecasted[:, 4]
    actual_values = filtered_actual[:, 4]
    
    # Calculate the overall MSE
    overall_rrmse = np.sqrt(np.mean((forecasted_values - actual_values) ** 2))/actual_values

    # Filter for states and countries
    states_mask = np.isin(filtered_forecasted[:, 0], range(0, 50))
    countries_mask = np.isin(filtered_forecasted[:, 0], range(51, 87))

    # Calculate MSE for states
    states_forecasted_values = filtered_forecasted[states_mask, 4].astype(float)
    states_actual_values = filtered_actual[states_mask, 4].astype(float)
    states_rrmse = np.sqrt(np.mean((states_forecasted_values - states_actual_values) ** 2))/actual_values

    # Calculate MSE for countries
    countries_forecasted_values = filtered_forecasted[countries_mask, 4].astype(float)
    countries_actual_values = filtered_actual[countries_mask, 4].astype(float)
    countries_rrmse = np.sqrt(np.mean((countries_forecasted_values - countries_actual_values) ** 2))/actual_values
    
    results = []
    results.append((states_rrmse, countries_rrmse, overall_rrmse))
    
    return results 

