import os
import sys

# Set the PYTHONPATH to the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmm_code.gmm_gcnn import GMMGCNN
from gmm_code.utils import create_forecasting_dataset, create_dataset, create_knn_graph

file_path = os.path.join(os.path.dirname(__file__), "lab2_NOAA", "NOA_109_data.npy")
timeseries_data = np.load(file=file_path)
print(
    f"The dataset contains {timeseries_data.shape[1]} measurements over {timeseries_data.shape[0]} stations."
)

pred_horizen = 1
obs_window = 4
n_stations = timeseries_data.shape[0]
n_timestamps = timeseries_data.shape[1]


def train_epoch_gcnn(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for sample in loader:
        x = sample[0].reshape(n_stations, obs_window)
        y = sample[1].reshape(n_stations, -1)
        out = model(shift_operator, x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        break
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch_gcnn(model, loader, criterion):
    model.eval()
    total_loss = 0

    for sample in loader:
        x = sample[0].reshape(n_stations, obs_window)
        y = sample[1].reshape(n_stations, -1)

        # print(shift_operator.shape)
        # print(x.shape)

        out = model(shift_operator, x)
        # print(out)
        loss = criterion(out, y)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_gcnn(model, num_epochs, criterion, train_loader, test_loader):
    # TODO: Check loss function!
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    start_time = time.time()
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # Model training
        train_loss = train_epoch_gcnn(model, train_loader, optimizer, criterion)

        # Model validation
        val_loss = evaluate_epoch_gcnn(model, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(
                "epoch:",
                epoch,
                "\t training loss:",
                np.round(train_loss, 4),
                "\t validation loss:",
                np.round(val_loss, 4),
            )

    elapsed_time = time.time() - start_time
    print(f"Model training took {elapsed_time:.3f} seconds")

    return train_losses, val_losses


if __name__ == "__main__":
    import time

    split = [
        0.3,
        0.2,
        0.5,
    ]

    dataset = create_forecasting_dataset(
        timeseries_data,
        splits=split,
        pred_horizen=pred_horizen,
        obs_window=obs_window,
        in_sample_mean=False,
    )

    size = 10

    def _create_missing_dataset(data, remaining_frac=0.8):
        # Clone the data to avoid modifying the original
        remaining_data = data.clone()
        num_samples, num_nodes, num_features = remaining_data.shape

        # Calculate the number of values to keep as non-NaN
        total_values = num_samples * num_nodes * num_features
        num_values_to_keep = int(remaining_frac * total_values)

        # Create a flat mask for the entire data
        mask = torch.rand(total_values) < remaining_frac

        # Reshape the mask to match the data shape
        mask = mask.reshape(num_samples, num_nodes, num_features)

        # Apply the mask, setting unmasked values to NaN
        remaining_data[~mask] = torch.nan

        return remaining_data

    data_fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    for frac in data_fractions:

        train_set = dataset["trn"]
        val_set = dataset["val"]
        test_set = dataset["tst"]

        X_train = torch.Tensor(train_set["data"][:size])
        # print(X_train.shape)
        y_train = torch.Tensor(train_set["labels"][:size])

        X_val = torch.Tensor(val_set["data"][:size])
        y_val = torch.Tensor(val_set["labels"][:size])

        X_test = torch.Tensor(test_set["data"][:size])
        y_test = torch.Tensor(test_set["labels"][:size])

        X_train = _create_missing_dataset(X_train, frac)
        X_val = _create_missing_dataset(X_val, frac)
        X_test = _create_missing_dataset(X_test, frac)

        A = np.load(
            file=os.path.join(
                os.path.dirname(__file__), "lab2_NOAA/NOA_109_original_adj.npy"
            )
        )

        k = 10
        G = create_knn_graph(A, k)

        A = torch.Tensor(nx.to_numpy_array(G))

        def normalize_adjacency_matrix(A):
            D = torch.diag(torch.sum(A, axis=1) ** (-0.5))
            return torch.mm(torch.mm(D, A), D)

        # ! Need normalized adjacency matrix, otherwhise powers in the order explode to -1
        A = normalize_adjacency_matrix(A)

        shift_operator = A.clone().detach()
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

        train_dataset = create_dataset(X_train, y_train, edge_index)
        val_dataset = create_dataset(X_val, y_val, edge_index)
        test_dataset = create_dataset(X_test, y_test, edge_index)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = GMMGCNN(
            obs_size=obs_window,
            pred_size=pred_horizen,
            hid_sizes=[16, 16],
            num_components=5,
            all_features=X_train[0, :, :],
            all_A=A,
            order=2,
        )

        # print(X_train.shape)

        train_gcnn(model, 10, torch.nn.MSELoss(), train_loader, test_loader)
