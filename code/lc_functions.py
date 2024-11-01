# set up lee-carter function

def lee_carter(mx_matrix):
    """
    Run the Lee-Carter model on age-specific mortality data.
    
    Args:
        mx_matrix (numpy.ndarray): A 2D array of age-specific mortality rates. rows = age, columns = years
        
    Returns:
        tuple: A tuple containing the estimated parameters (ax, bx, kt) and the fitted mortality rates.
    """
    # set 0s to small number so log resutls are valid
    mx_matrix[mx_matrix <= 0] = 1e-9

    # ax are averages over time of log(mx) 
    ax = np.mean(np.log(mx_matrix), axis=1) # time axis = 1
    ax = ax.reshape(-1, 1) # reshape ax into column vector
    
    # center and log mx
    centered_mx = np.log(mx_matrix) - ax
    
    # SVD on centered matrix
    U, S, Vt = np.linalg.svd(centered_mx, full_matrices=False)

    # extract right and left singular vectors (bx and kt)
    bx = U[:, 0]
    kt = Vt[0, :]

    # normalize bx to sum to 1 and kt to sum to 0
    bx = bx / np.sum(bx)
    kt = kt - np.mean(kt)

    # estimate fitted mortality 
    fitted_mort = np.exp(ax + np.outer(bx, kt))

    return (ax, bx, kt), fitted_mort

# set up function to run multiple models on all years in training data

def lee_carter_geo_gender(data):

    geos = np.unique(data[:, 0])
    genders = np.unique(data[:, 1])

    results = {}

    # fit LC model to each geography and gender seperately 
    for geo in geos:
        for gender in genders:
            mask = (data[:, 0] == geo) & (data[:, 1] == gender)
            geo_gender_data = data[mask]

            # extract ages and years
            years = np.unique(geo_gender_data[:, 2])
            ages = np.unique(geo_gender_data[:, 3])

            m_x = np.zeros((len(ages), len(years)))

            for i, age in enumerate(ages):
                for j, year in enumerate(years):
                    mask = (geo_gender_data[:, 3] == age) & (geo_gender_data[:, 2] == year)
                    #m_x[i,j] = geo_gender_data[mask, 4]
                    selected_data = geo_gender_data[mask, 4]

                    # Ensure we handle different cases for selected_data
                    if selected_data.size == 0:
                        # No data available for this age and year
                        m_x[i, j] = np.nan  # Assign NaN or some default value
                    elif selected_data.size == 1:
                        # Exactly one value, the expected case
                        m_x[i, j] = selected_data[0]
                    else:
                        # More than one value, choose an aggregation method
                        print("more than 1 value")
                        m_x[i, j] = np.mean(selected_data)  # Or use np.median, np.min, etc.

            # Debugging
             # Check for NaN or infinite values
            if np.isnan(m_x).any() or np.isinf(m_x).any():
                print(f"Skipping Geo: {geo}, Gender: {gender} due to NaN or infinite values in m_x")
                continue

            try:
                params, fitted_mort = lee_carter(m_x)
            except np.linalg.LinAlgError as e:
                print(f"SVD did not converge for Geo: {geo}, Gender: {gender}. Error: {str(e)}")
                continue

            params, fitted_mort = lee_carter(m_x)
    
            # Store the results for the current geo and gender
            results[(geo, gender)] = {
                'params': params,
                'fitted_mortality': fitted_mort
            }
    
    return results


def lee_carter_forecast(results, h, start_year, ages, drift=True):
    """
    Perform the forecasting step of the Lee-Carter method using a random walk with drift.
    
    Args:
        results (dict): A dictionary containing the estimated parameters (ax, bx, kt) for each state and gender combination.
        h (int): The number of future periods to forecast.
        start_year (int): The starting year of the forecast.
        ages (numpy.ndarray): A 1D array of ages corresponding to the rows of the mortality matrix.
        drift (bool, optional): Whether to include a drift term in the random walk. Default is True.
        
    Returns:
        numpy.ndarray: A 2D array with 5 columns representing state, gender, year, age, and forecasted mortality rate.
    """
    
    forecasts = []
    
    for geo, gender in results.keys():
        ax, bx, kt = results[(geo, gender)]['params']
        
        # Estimate the drift term (slope in kt)
        if drift:
            drift_term = (kt[-1] - kt[0]) / (len(kt) - 1)
        else:
            drift_term = 0
        
        # Forecast future kt values using a random walk with drift
        kt_forecast = np.zeros(h)
        kt_forecast[0] = kt[-1]
        for i in range(1, h):
            kt_forecast[i] = kt_forecast[i-1] + drift_term + np.random.normal(0, 1)
        
        # Forecast future mortality rates
        ax_matrix = np.repeat(ax, h).reshape(-1, h)
        bx_matrix = np.repeat(bx, h).reshape(-1, h)
        kt_matrix = np.repeat(kt_forecast, len(ax)).reshape(h, -1).T
        mortality_forecast = np.exp(ax_matrix + bx_matrix * kt_matrix)

        # Clipping forecasted mortality rates to a maximum of 1
        mortality_forecast = np.clip(mortality_forecast, 0, 1)

        # Create a 2D array with geo, gender, year, age, and forecasted mortality rate
        for i in range(h):
            year = start_year + i
            for j, age in enumerate(ages):
                forecasts.append([geo, gender, year, age, mortality_forecast[j, i]])

    # Convert forecasts to a NumPy array
    forecasts = np.array(forecasts)

    # Sort the forecasts array based on the first four columns
    sorted_indices = np.lexsort((forecasts[:, 3], forecasts[:, 2], forecasts[:, 1], forecasts[:, 0]))
    forecasts = forecasts[sorted_indices]

    
    return forecasts

def calculate_mse(forecasted_rates, actual_rates):
    """
    Calculate the Mean Squared Error (MSE) between the forecasted and actual mortality rates.
    
    Args:
        forecasted_rates (numpy.ndarray): A 2D array with 5 columns representing state, gender, year, age, and forecasted mortality rate.
        actual_rates (numpy.ndarray): A 2D array with 5 columns representing state, gender, year, age, and actual mortality rate.
        
    Returns:
        float: The Mean Squared Error (MSE) between the forecasted and actual mortality rates.
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
    overall_mse = np.mean((forecasted_values - actual_values) ** 2)

    # Filter for states and countries
    states_mask = np.isin(filtered_forecasted[:, 0], range(0, 50))
    countries_mask = np.isin(filtered_forecasted[:, 0], range(51, 87))

    # Calculate MSE for states
    states_forecasted_values = filtered_forecasted[states_mask, 4].astype(float)
    states_actual_values = filtered_actual[states_mask, 4].astype(float)
    states_mse = np.mean((states_forecasted_values - states_actual_values) ** 2)

    # Calculate MSE for countries
    countries_forecasted_values = filtered_forecasted[countries_mask, 4].astype(float)
    countries_actual_values = filtered_actual[countries_mask, 4].astype(float)
    countries_mse = np.mean((countries_forecasted_values - countries_actual_values) ** 2)

    # # Plotting the Actual vs. Forecasted Mortality Rates for States
    # plt.figure(figsize=(12, 6))
    # plt.scatter(states_actual_values, states_forecasted_values, alpha=0.6)
    # plt.plot([min(states_actual_values), max(states_actual_values)], 
    #          [min(states_actual_values), max(states_actual_values)], 
    #          color='red', linestyle='--', linewidth=2)
    # plt.title('Actual vs Forecasted Mortality Rates for States')
    # plt.xlabel('Actual Mortality Rate')
    # plt.ylabel('Forecasted Mortality Rate')
    # plt.grid(True)
    # plt.show()

    # # Plotting the Actual vs. Forecasted Mortality Rates for Countries
    # plt.figure(figsize=(12, 6))
    # plt.scatter(countries_actual_values, countries_forecasted_values, alpha=0.6)
    # plt.plot([min(countries_actual_values), max(countries_actual_values)], 
    #          [min(countries_actual_values), max(countries_actual_values)], 
    #          color='red', linestyle='--', linewidth=2)
    # plt.title('Actual vs Forecasted Mortality Rates for Countries')
    # plt.xlabel('Actual Mortality Rate')
    # plt.ylabel('Forecasted Mortality Rate')
    # plt.grid(True)
    # plt.show()
    
    results = []
    results.append((states_mse, countries_mse, overall_mse))
    
    return results 

def run_lc_model(train_data, test_data):
    lc_output = lee_carter_geo_gender(train_data)
    predictions = lee_carter_forecast(lc_output, h=10, start_year=2006, ages=range(0, 100))
    test_mse = calculate_mse(predictions, test_data)

    return lc_output, predictions, test_mse