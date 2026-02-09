import pandas as pd
import os

X = pd.read_csv("data/processed/X.csv", header=None)
y = pd.read_csv("data/processed/y.csv", header=None)

data = pd.concat([X, y], axis=1)

os.makedirs("federated/clients", exist_ok=True)

client_labels = {
    0: [0],        # Mostly Normal
    1: [1],        # Mostly DoS
    2: [2],        # Mostly Probe
    3: [3, 4],     # Rare attacks
    4: [0, 1, 2]   # Mixed
}

for cid, labels in client_labels.items():
    path = f"federated/clients/client_{cid}"
    os.makedirs(path, exist_ok=True)

    subset = data[data.iloc[:, -1].isin(labels)]

    subset.to_csv(f"{path}/data.csv", index=False, header=False)

    print(f"Client {cid} labels {labels} samples:", len(subset))
