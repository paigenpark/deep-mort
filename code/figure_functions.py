import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as plt
tfkl = tf.keras.layers

# function to plot actual mortality rates for given year, gender, geo against model predictions 
def plot_mort_predictions(geo, year, age_range, genders, data, dl_models, model_names, lc_predictions, geos_key):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()

    for idx, gender in enumerate(genders):
        # Filter the actual data for the specific state, year, and gender
        actual_data = data[(data[:, 0] == geo) & (data[:, 2] == year) & (data[:, 1] == gender)]
        ages = actual_data[:, 3]  # Age column
        actual_rates = actual_data[:, 4]  # Mortality rate column

        # Prepare the plot
        ax = axes[idx]
        ax.plot(ages, actual_rates, label='Actual Rates', marker='o', linestyle='-', color='black')


        # Generate predictions from each model
        for model_name in model_names:
            if model_name == 'Lee-Carter':
                # Filter predictions for the specific geo and gender
                lc_pred = lc_predictions[(lc_predictions[:, 0] == geo) & 
                                                (lc_predictions[:, 1] == gender) &
                                                (lc_predictions[:, 2] == year)]       

                # Plot Lee-Carter predictions
                ax.plot(age_range, lc_pred[:, 4], label='Lee-Carter Model', linestyle='--', marker='x')

            else:
                # DL models
                model = dl_models[model_name]
                predictions = []

                for age in age_range:
                    input_features = (tf.convert_to_tensor([(year - 1959) / 60], dtype=tf.float32),  # Normalized year
                                  tf.convert_to_tensor([age], dtype=tf.float32),  # Age
                                  tf.convert_to_tensor([geo], dtype=tf.float32),  # Geography
                                  tf.convert_to_tensor([gender], dtype=tf.float32))  # Gender
                    # Predict using the model
                    pred = model.predict(input_features)
                    predictions.append(pred[0][0])  # Append predicted rate

                # Plot model predictions
                ax.plot(age_range, predictions, label=f'{model_name} Model', linestyle='--', marker='x')

        # Title and labels
        ax.set_title(f'Mortality Rates: {"Male" if gender == 0 else "Female"} in {year} for {geos_key[geo]}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Mortality Rate')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    # plot difference from actual rates by age 
def plot_pred_diff_by_age(geo, year, age_range, genders, data, dl_models, model_names, lc_predictions, geos_key):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()

    for idx, gender in enumerate(genders):
        # Filter the actual data for the specific state, year, and gender
        actual_data = data[(data[:, 0] == geo) & (data[:, 2] == year) & (data[:, 1] == gender)]
        ages = actual_data[:, 3]  # Age column
        actual_rates = actual_data[:, 4]  # Mortality rate column

        # Prepare the plot
        ax = axes[idx]
        ax.axhline(y=0, color='black', linestyle='-', label='Actual Rates')

        # Generate predictions from each model
        for model_name in model_names:
            if model_name == 'Lee-Carter':
                # Filter predictions for the specific geo and gender
                lc_pred = lc_predictions[(lc_predictions[:, 0] == geo) & 
                                        (lc_predictions[:, 1] == gender) &
                                        (lc_predictions[:, 2] == year)]
               
                lc_diff = lc_pred[:, 4] - actual_rates

                # Plot Lee-Carter predictions
                ax.plot(age_range, lc_diff, label='Lee-Carter Model Difference', linestyle='--', marker='x')

            else:
                # DL models
                model = dl_models[model_name]
                predictions = []

                for age in age_range:
                    input_features = (tf.convert_to_tensor([(year - 1959) / 60], dtype=tf.float32),  # Normalized year
                                  tf.convert_to_tensor([age], dtype=tf.float32),  # Age
                                  tf.convert_to_tensor([geo], dtype=tf.float32),  # Geography
                                  tf.convert_to_tensor([gender], dtype=tf.float32))  # Gender
                    # Predict using the model
                    pred = model.predict(input_features)
                    predictions.append(pred[0][0])  # Append predicted rate

                dl_diff = np.array(predictions) - actual_rates 

                # Plot model predictions
                ax.plot(age_range, dl_diff, label=f'{model_name} Model Differences', linestyle='--', marker='x')

        # Title and labels
        ax.set_title(f'Difference in Predictions vs Actual: {"Male" if gender == 0 else "Female"} in {year} for {geos_key[geo]}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Difference in Mortality Rate')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    # plot difference by year 
def plot_pred_diff_by_year(geo, year_range, age, genders, data, dl_models, model_names, lc_predictions, geos_key):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()

    for idx, gender in enumerate(genders):
        # Filter the actual data for the specific state, year, and gender
        actual_data = data[(data[:, 0] == geo) & (data[:, 3] == age) & (data[:, 1] == gender)]
        actual_rates = actual_data[:, 4]  # Mortality rate column

        # Prepare the plot
        ax = axes[idx]
        ax.axhline(y=0, color='black', linestyle='-', label='Actual Rates')

        # Generate predictions from each model
        for model_name in model_names:
            if model_name == 'Lee-Carter':
                # Filter predictions for the specific geo and gender
                lc_pred = lc_predictions[(lc_predictions[:, 0] == geo) & 
                                        (lc_predictions[:, 1] == gender) &
                                        (lc_predictions[:, 3] == age)]
               
                lc_diff = lc_pred[:, 4] - actual_rates

                # Plot Lee-Carter predictions
                ax.plot(year_range, lc_diff, label='Lee-Carter Model Difference', linestyle='--', marker='x')

            else:
                # DL models
                model = dl_models[model_name]
                predictions = []

                for year in year_range:
                    input_features = (tf.convert_to_tensor([(year - 1959) / 60], dtype=tf.float32),  # Normalized year
                                  tf.convert_to_tensor([age], dtype=tf.float32),  # Age
                                  tf.convert_to_tensor([geo], dtype=tf.float32),  # Geography
                                  tf.convert_to_tensor([gender], dtype=tf.float32))  # Gender
                    # Predict using the model
                    pred = model.predict(input_features)
                    predictions.append(pred[0][0])  # Append predicted rate

                dl_diff = np.array(predictions) - actual_rates 

                # Plot model predictions
                ax.plot( year_range, dl_diff, label=f'{model_name} Model Differences', linestyle='--', marker='x')

        # Title and labels
        ax.set_title(f'Difference in Predictions vs Actual: For Sex {"Male" if gender == 0 else "Female"}, Age {age}, and US State {geos_key[geo]}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Difference in Mortality Rate')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    # plot mse by category 


def plot_mse_section(mse_by_category1, mse_by_category2, mse_by_category3, category_labels, plot_type, start, end, x_max):
    category_labels_dict = {int(num): label for label, num in category_labels}
    # Filter the MSE dictionary for keys in the range start to end (inclusive)
    filtered_mse1 = {k: v for k, v in mse_by_category1.items() if start <= k <= end}
    filtered_mse2 = {k: v for k, v in mse_by_category2.items() if start <= k <= end}
    filtered_mse3 = {k: v for k, v in mse_by_category3.items() if start <= k <= end}
    
    # Sort the filtered dictionary by the MSE values in decreasing order
    sorted_mse = sorted(filtered_mse1.items(), key=lambda item: item[1], reverse=True)
    
    # Extract the keys and values for plotting
    categories = [item[0] for item in sorted_mse]
    mse_values1 = [item[1] for item in sorted_mse]
    mse_values_model2 = [filtered_mse2.get(cat, 0) for cat in categories]
    mse_values_model3 = [filtered_mse3.get(cat, 0) for cat in categories]

    category_labels_mapped = [category_labels_dict.get(cat, str(cat)) for cat in categories]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 12))
    indices = np.arange(len(categories))
    bar_width = 0.3
    
    ax.barh(indices, mse_values1, height=bar_width, color='skyblue', label='Combined Model')
    ax.barh(indices + bar_width, mse_values_model2, height=bar_width, color='salmon', label='Separate Model')
    ax.barh(indices + 2 * bar_width, mse_values_model3, height=bar_width, color='lightgrey', label='Lee-Carter Model')  # Faded grey color
    
    # Add labels and adjust y-axis
    ax.set_yticks(indices + bar_width)
    ax.set_yticklabels(category_labels_mapped, fontsize=12)  # Set the label in the middle of the bars
    ax.set_xlabel('MSE', fontsize=12)
    ax.set_ylabel(f'{plot_type}', fontsize=12)
    ax.set_title(f'MSE Comparison for {plot_type}', fontsize=14)
    ax.legend()
    
    # Set x-axis limit for zoomed view
    ax.set_xlim(0, x_max)
    
    fig.tight_layout()
    return fig, ax

