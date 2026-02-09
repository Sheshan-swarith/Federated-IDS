import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import numpy as np
import torch

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from models.ids_model import IDSNet


# -------- MEDIAN AGGREGATION FUNCTION --------
def aggregate_median(results):
    weights = [parameters_to_ndarrays(r.parameters) for _, r in results]

    median_weights = []
    for layer in zip(*weights):
        median_weights.append(np.median(np.array(layer), axis=0))

    return ndarrays_to_parameters(median_weights)


# -------- CUSTOM STRATEGY --------
class MedianStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        print("Using Median Aggregation (Defense Enabled)")

        aggregated_params = aggregate_median(results)

        # Save global model
        ndarrays = parameters_to_ndarrays(aggregated_params)

        model = IDSNet(input_dim=145, num_classes=5)
        state_dict = dict(zip(model.state_dict().keys(), ndarrays))
        model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

        torch.save(model.state_dict(), "global_model_median.pth")
        print("Saved defended global model")

        return aggregated_params, {}


# -------- START SERVER --------
if __name__ == "__main__":

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
    )


    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
