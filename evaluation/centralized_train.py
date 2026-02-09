import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.ids_model import IDSNet
from sklearn.metrics import accuracy_score

X = pd.read_csv("data/processed/X.csv", header=None).values
y = pd.read_csv("data/processed/y.csv", header=None).values.flatten()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

model = IDSNet(X.shape[1], 5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Train
for epoch in range(5):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

# Evaluate
with torch.no_grad():
    preds = torch.argmax(model(X), dim=1)

acc = accuracy_score(y, preds.numpy())
print("Centralized Accuracy:", acc)
