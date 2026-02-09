import pandas as pd
import os

NUM_CLIENTS = 5

X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv")

# Fix column names
X.columns = [f"f{i}" for i in range(X.shape[1])]
y.columns = ["label"]

data = pd.concat([X, y], axis=1)

# Non-IID: sort by label
data_sorted = data.sort_values(by="label").reset_index(drop=True)

os.makedirs("federated/client_data", exist_ok=True)

# Pandas-based splitting
splits = [data_sorted.iloc[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

for i, client_df in enumerate(splits):
    path = f"federated/client_data/client_{i}"
    os.makedirs(path, exist_ok=True)
    client_df.to_csv(f"{path}/data.csv", index=False)
    print(f"Client {i} samples:", len(client_df))

print("Client datasets created")
