import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df = pd.read_csv("data/raw/NSL_KDD.csv", header=None)

X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1].astype(int)

# One-hot encode categorical features
X = pd.get_dummies(X)
X = X.values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Label mapping to 5 classes
def map_label(x):
    if x == 21:
        return 0   # Normal
    elif x in [18, 19, 20]:
        return 1   # DoS
    elif x in [15, 16, 17]:
        return 2   # Probe
    elif x in [11, 12, 13, 14]:
        return 3   # R2L
    else:
        return 4   # U2R

y = y_raw.apply(map_label).values

# Save processed dataset
pd.DataFrame(X).to_csv("data/processed/X.csv", index=False, header=False)
pd.DataFrame(y).to_csv("data/processed/y.csv", index=False, header=False)

np.savez("data/processed/dataset.npz", X=X, y=y)

print("Preprocessing completed")
print("Class distribution:")
print(pd.Series(y).value_counts())
