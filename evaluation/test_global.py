import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from models.ids_model import IDSNet
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load full dataset
dfX = pd.read_csv("data/processed/X.csv", header=None)
dfy = pd.read_csv("data/processed/y.csv", header=None)

X = torch.tensor(dfX.values, dtype=torch.float32)
y = torch.tensor(dfy.values.squeeze(), dtype=torch.long)

# Load model
input_dim = X.shape[1]
model = IDSNet(input_dim, num_classes=5)
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(X)
    _, preds = torch.max(outputs, 1)

# Accuracy
acc = (preds == y).sum().item() / len(y)
print("Global Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y, preds)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, preds, digits=4))

# Plot confusion matrix
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix - Federated Non-IID IDS")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
