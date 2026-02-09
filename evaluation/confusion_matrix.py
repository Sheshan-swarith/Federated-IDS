import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from models.ids_model import IDSNet
from torch.utils.data import DataLoader, TensorDataset

# Load model
model = IDSNet(input_dim=145)
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# Load dataset
X = pd.read_csv("data/processed/X.csv", header=None)
y = pd.read_csv("data/processed/y.csv", header=None)

X = torch.tensor(X.values, dtype=torch.float32)
y = y.values.flatten()

loader = DataLoader(TensorDataset(X, torch.tensor(y)), batch_size=512)

all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in loader:
        preds = model(xb).argmax(1).numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Federated IDS")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from models.ids_model import IDSNet

# Load full dataset
X = pd.read_csv("data/processed/X.csv", header=None).values
y = pd.read_csv("data/processed/y.csv", header=None).values.flatten()

X = torch.tensor(X, dtype=torch.float32)

# Load trained global model
model = IDSNet(input_dim=X.shape[1], num_classes=5)
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(X)
    _, preds = torch.max(outputs, 1)

y_true = y
y_pred = preds.numpy()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Federated IDS")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
