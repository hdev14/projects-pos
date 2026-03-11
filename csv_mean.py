import pandas as pd

data = pd.read_csv('example.csv')
mean_value = data['column_name'].mean()

print(mean_value)