import pandas as pd

for cid in range(5):
    df = pd.read_csv(f"federated/clients/client_{cid}/data.csv", header=None)
    labels = df.iloc[:, -1]
    print(f"\nClient {cid} label distribution:")
    print(labels.value_counts())
