import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.ids_model import IDSNet

# -------------------------------
# Attack Settings
# -------------------------------
ATTACKER_ID = None
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Load Client Dataset
# -------------------------------
def load_client_data(cid):
    path = f"federated/clients/client_{cid}/data.csv"
    df = pd.read_csv(path, header=None)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # ⚠ Label Flipping Attack
    if int(cid) == ATTACKER_ID:
      print("⚠ Malicious client performing label flipping attack")
      y = np.random.randint(0, NUM_CLASSES, size=len(y)).astype(np.int64)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return loader


# -------------------------------
# Flower Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val).to(DEVICE)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        self.model.to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1):
            for X, y in self.trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        self.model.to(DEVICE)

        correct = 0
        total = 0
        loss_total = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in self.testloader:
                X, y = X.to(DEVICE), y.to(DEVICE)

                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss_total += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = loss_total / len(self.testloader)
        accuracy = correct / total

        print(f"Client Evaluation -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return float(avg_loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


# -------------------------------
# Start Client
# -------------------------------
def main():
    cid = sys.argv[1]
    print(f"Starting client {cid}")

    trainloader = load_client_data(cid)
    testloader = trainloader  # same data for evaluation

    model = IDSNet(input_dim=145, num_classes=5)

    client = FlowerClient(model, trainloader, testloader)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )


if __name__ == "__main__":
    main()
