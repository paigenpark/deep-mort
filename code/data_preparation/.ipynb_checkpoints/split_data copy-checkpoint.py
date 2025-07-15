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
            # get rid of years, ages, not in health data and other cleaning
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
np.save('../../data/geos_key_new.npy', geos_key)

# create combined data
combined = np.vstack((state_data, country_data))

##### State Splits #####
# training set includes data from years 1959-2005
training_index = np.logical_and(state_data[:, 2] >= 1959, state_data[:, 2] <= 2005)
state_training = state_data[training_index, :]
np.savetxt('../../data/state_training.txt', state_training)

# test set 1 includes years 2005-2015
test_index = np.logical_and(state_data[:, 2] > 2005, state_data[:, 2] <= 2015)
state_test = state_data[test_index, :]
np.savetxt('../../data/state_test.txt', state_test)

# final test set that I'm not touching until the very end is 2015-2019
final_test_index = np.logical_and(state_data[:, 2] > 2015, state_data[:, 2] <= 2019)
state_final_test = state_data[final_test_index, :]
np.savetxt('../../data/state_final_test.txt', state_final_test)

##### Country Splits #####
training_index = np.logical_and(country_data[:, 2] >= 1959, country_data[:, 2] <= 2005)
country_training = country_data[training_index, :]
np.savetxt('../../data/country_training_new.txt', country_training)

test_index = np.logical_and(country_data[:, 2] > 2005, country_data[:, 2] <= 2015)
country_test = country_data[test_index, :]
np.savetxt('../../data/country_test_new.txt', country_test)

final_test_index = np.logical_and(country_data[:, 2] > 2015, country_data[:, 2] <= 2019)
country_final_test = country_data[final_test_index, :]
np.savetxt('../../data/country_final_test_new.txt', country_final_test)

#### Combined Splits #####
# split combined data
training_index = np.logical_and(combined[:, 2] >= 1959, combined[:, 2] <= 2005)
combined_training = combined[training_index, :]
np.savetxt('../../data/combined_training_new.txt', combined_training)

test_index = np.logical_and(combined[:, 2] > 2005, combined[:, 2] <= 2015)
combined_test = combined[test_index, :]
np.savetxt('../../data/combined_test_new.txt', combined_test)

final_test_index = np.logical_and(combined[:, 2] > 2015, combined[:, 2] <= 2019)
combined_final_test = combined[final_test_index, :]
np.savetxt('../../data/combined_final_test_new.txt', combined_final_test) 








