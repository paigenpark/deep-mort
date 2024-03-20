import csv
import pandas as pd
import numpy as np

year = '1993'

df = pd.read_csv('../../data/brfss_data/brfss' + year + '.csv', encoding='cp1252', low_memory=False)
print(df.columns)
print(df["x.ageg5yr"])
print(df['cdem.01'])
print(df.shape)


income = df['income']
race = df['race']
state = df['x.state']
age = df['x.ageg5yr']
sex = df['sex']
height = df['height']
weight = df['weight']

# brfss_out.to_csv('brfss'+year+'clean.csv')

