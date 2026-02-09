import pandas as pd

df = pd.read_csv("data/raw/NSL_KDD.csv", header=None)
labels = df.iloc[:, -1].astype(str)

print("Top 30 labels and their counts:")
print(labels.value_counts().head(30))
print(sorted(labels.unique()))
