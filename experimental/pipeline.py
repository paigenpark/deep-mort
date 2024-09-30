import tensorflow as tf
import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt
tfkl = tf.keras.layers

class Pipeline:
    def __init__(self, paths):
        self.geos = []
        self.genders = []
        self.data = self.load_data(paths)      

    def load_data(self, paths):
        all_data = []
        
        for path in paths:
            data, self.geos, self.genders = self.load_file(path, self.geos, self.genders)
            all_data.append(data)

        combined_data = np.vstack(all_data)
        return combined_data

    def load_file(self, path, geos, genders):
        data = []
        ages = []

        with open(path, "r") as file:
            reader = csv.reader(file,delimiter=',')
            for row_index, row in enumerate(reader):
                if row_index == 0:
                    print(row)
                if row_index >= 1:
                    geo, gender, year, age, rate = row
                    year = int(year)
                    try:
                        age = int(age)
                    except:
                        age = -1
                    if geo not in geos:
                        geos.append(geo)
                    geo = geos.index(geo)
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
                        data.append([geo, gender, year, age, rate])

        return np.array(data), geos, genders


