import csv
import numpy as np
import os as os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# loading in USMDB data
data = []
ages = []
states = []
genders = []

with open("../../data/usmdb.csv", "r") as file:
    reader = csv.reader(file,delimiter=',')
    for row_index, row in enumerate(reader):
        if row_index == 0:
            print(row)
        if row_index >= 1:
            state, gender, year, age, rate = row
            year = int(year)
            try:
                age = int(age)
            except:
                age = -1
            if state not in states:
                states.append(state)
            state = states.index(state)
            if gender not in genders:
                genders.append(gender)
            gender = genders.index(gender)
            try:
                rate = float(rate)
            except:
                rate = -1
            if rate > 1:
                rate = 1
            if age != -1 and rate != -1 and age <= 99:
                data.append([state, gender, year, age, rate])

state_data = np.array(data)

# loading in HMD data
data = []
ages = []
countries = []
genders = []
countries_to_remove = ["CHL", 'DEUTNP', 'FRACNP', 'GBRCENW', 'GBR_NP', 'HKG', 'HRV', 'KOR', 'NZL_MA', 'NZL_NM']


with open("../../data/hmd.csv", "r") as file:
    reader = csv.reader(file,delimiter=",")
    for row_index, row in enumerate(reader):
        if row_index == 0:
            print(row)
        if row_index >= 1:
            country, gender, year, age, rate = row
            if country in countries_to_remove:
                continue
            year = int(year)
            try:
                age = int(age)
            except:
                age = -1
            if age not in ages and age != -1 and age <= 99:
                ages.append(age)
            if country not in countries:
                countries.append(country)
            country = countries.index(country)
            if gender not in genders:
                genders.append(gender)
            gender = genders.index(gender)
            try:
                rate = float(rate)
            except:
                rate = -1
            if rate > 1:
                rate = 1
            if age != -1 and rate != -1 and age <= 99:
                data.append([country, gender, year, age, rate])

country_data = np.array(data)

# getting unique values for geographic location column 
country_data[:,0] = country_data[:,0] + 50

# Below, I create a joint list of populations and their 
# corresponding numeric code that identifies them in the data
geos_list = states + countries
geos_index = np.arange(len(geos_list))
geos_key = np.column_stack((np.array(geos_list), geos_index))
np.save('../../data/geos_key.npy', geos_key)

# create combined data
combined = np.vstack((state_data, country_data))

    # ##### State Splits #####
    # # training set includes data from years 1959-2000
    # training_index = np.logical_and(state_data[:, 2] >= 1959, state_data[:, 2] <= 2000)
    # state_training = state_data[training_index, :]
    # np.savetxt('../../data/state_training.txt', state_training)

    # # Calibration set: near-future years for learning recalibration mapping
    # calibration_index = np.logical_and(state_data[:, 2] > 2000, state_data[:, 2] <= 2005)
    # state_calibration = state_data[calibration_index, :]
    # np.savetxt('../../data/state_calibration.txt', state_calibration)

    # # Evaluation set: further-future years for testing calibration
    # evaluation_index = np.logical_and(state_data[:, 2] > 2005, state_data[:, 2] <= 2010)
    # state_evaluation = state_data[evaluation_index, :]
    # np.savetxt('../../data/state_evaluation.txt', state_evaluation)

    # # Keep the combined test set for backwards compatibility
    # test_index = np.logical_and(state_data[:, 2] > 2000, state_data[:, 2] <= 2009)
    # state_test = state_data[test_index, :]
    # np.savetxt('../../data/state_test.txt', state_test)

    # # final test set that I'm not touching until the very end is 2010-2019
    # final_test_index = np.logical_and(state_data[:, 2] > 2009, state_data[:, 2] <= 2019)
    # state_final_test = state_data[final_test_index, :]
    # np.savetxt('../../data/state_final_test.txt', state_final_test)

##### Country Splits #####
training_index = np.logical_and(country_data[:, 2] >= 1959, country_data[:, 2] <= 2000)
country_training = country_data[training_index, :]
np.savetxt('../../data/country_training.txt', country_training)

# Calibration set: near-future years for learning recalibration mapping
calibration_index = np.logical_and(country_data[:, 2] > 2000, country_data[:, 2] <= 2005)
country_calibration = country_data[calibration_index, :]
np.savetxt('../../data/country_calibration.txt', country_calibration)

# Evaluation set: further-future years for testing calibration
evaluation_index = np.logical_and(country_data[:, 2] > 2005, country_data[:, 2] <= 2010)
country_evaluation = country_data[evaluation_index, :]
np.savetxt('../../data/country_evaluation.txt', country_evaluation)

# Keep the combined test set for backwards compatibility
test_index = np.logical_and(country_data[:, 2] > 2000, country_data[:, 2] <= 2009)
country_test = country_data[test_index, :]
np.savetxt('../../data/country_test.txt', country_test)

final_test_index = np.logical_and(country_data[:, 2] > 2010, country_data[:, 2] <= 2019)
country_final_test = country_data[final_test_index, :]
np.savetxt('../../data/country_final_test.txt', country_final_test)

# #### Combined Splits #####
# # split combined data
# training_index = np.logical_and(combined[:, 2] >= 1959, combined[:, 2] <= 2000)
# combined_training = combined[training_index, :]
# np.savetxt('../../data/combined_training.txt', combined_training)

# # Calibration set: near-future years for learning recalibration mapping
# calibration_index = np.logical_and(combined[:, 2] > 2000, combined[:, 2] <= 2005)
# combined_calibration = combined[calibration_index, :]
# np.savetxt('../../data/combined_calibration.txt', combined_calibration)

# # Evaluation set: further-future years for testing calibration
# evaluation_index = np.logical_and(combined[:, 2] > 2005, combined[:, 2] <= 2009)
# combined_evaluation = combined[evaluation_index, :]
# np.savetxt('../../data/combined_evaluation.txt', combined_evaluation)

# # Keep the combined test set for backwards compatibility
# test_index = np.logical_and(combined[:, 2] > 2000, combined[:, 2] <= 2009)
# combined_test = combined[test_index, :]
# np.savetxt('../../data/combined_test.txt', combined_test)

# final_test_index = np.logical_and(combined[:, 2] > 2009, combined[:, 2] <= 2019)
# combined_final_test = combined[final_test_index, :]
# np.savetxt('../../data/combined_final_test.txt', combined_final_test)

# Print split sizes
# print("\n=== DATA SPLIT SUMMARY ===")
# print(f"\nState data:")
# print(f"  Training (1959-2000):     {state_training.shape[0]:>8,} rows")
# print(f"  Calibration (2001-2005):  {state_calibration.shape[0]:>8,} rows")
# print(f"  Evaluation (2006-2009):   {state_evaluation.shape[0]:>8,} rows")
# print(f"  Final test (2010-2019):   {state_final_test.shape[0]:>8,} rows")
# print(f"\nCountry data:")
# print(f"  Training (1959-2000):     {country_training.shape[0]:>8,} rows")
# print(f"  Calibration (2001-2005):  {country_calibration.shape[0]:>8,} rows")
# print(f"  Evaluation (2006-2009):   {country_evaluation.shape[0]:>8,} rows")
# print(f"  Final test (2010-2019):   {country_final_test.shape[0]:>8,} rows")
# print(f"\nCombined data:")
# print(f"  Training (1959-2000):     {combined_training.shape[0]:>8,} rows")
# print(f"  Calibration (2001-2005):  {combined_calibration.shape[0]:>8,} rows")
# print(f"  Evaluation (2006-2009):   {combined_evaluation.shape[0]:>8,} rows")
# print(f"  Final test (2010-2019):   {combined_final_test.shape[0]:>8,} rows")







