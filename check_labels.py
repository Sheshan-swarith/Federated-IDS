import pandas as pd

y = pd.read_csv("data/processed/y.csv", header=None)
print(y[0].value_counts())
