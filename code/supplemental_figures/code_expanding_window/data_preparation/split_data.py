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



##### Country Splits (expanding window) #####
train_end_years = [1979, 1989, 1999, 2009]
for train_end in train_end_years:
    test_end = train_end + 10

    training_index = np.logical_and(country_data[:, 2] >= 1959, country_data[:, 2] <= train_end)
    country_training = country_data[training_index, :]
    np.savetxt(f'../../data/country_training_1959_{train_end}.txt', country_training)

    test_index = np.logical_and(country_data[:, 2] > train_end, country_data[:, 2] <= test_end)
    country_test = country_data[test_index, :]
    np.savetxt(f'../../data/country_test_{train_end + 1}_{test_end}.txt', country_test)









