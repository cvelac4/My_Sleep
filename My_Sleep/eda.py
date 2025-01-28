#eda.py
import numpy as np
import pandas as pd
import value

data = pd.read_csv('cmu-sleep.csv')
print(data.head(5))
print(data.info())
#print(data.describe())

#Check for unic values
print(data.nunique())

data = data.replace(' ', '0')

for column in data.columns:
    print(f'\n{column}')
    print(data[column].unique())
