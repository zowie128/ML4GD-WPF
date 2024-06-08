from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmmgcn import GMMGCN


timeseries_data = np.load(file="gmm_code/lab2_NOAA/NOA_109_data.npy")
print(
    f"The dataset contains {timeseries_data.shape[1]} measurements over {timeseries_data.shape[0]} stations."
)

pred_horizen = 1
obs_window = 4
n_stations = timeseries_data.shape[0]
n_timestamps = timeseries_data.shape[1]


# ! From lab
def create_forecasting_dataset(
    graph_signals,
    splits: list,
    pred_horizen: int,
    obs_window: int,
    in_sample_mean: bool,
):

    T = graph_signals.shape[1]
    max_idx_trn = int(T * splits[0])
    max_idx_val = int(T * sum(splits[:-1]))
    split_idx = np.split(np.arange(T), [max_idx_trn, max_idx_val])

    data_dict = {}
    data_type = ["trn", "val", "tst"]

    if in_sample_mean:
        in_sample_means = graph_signals[:, :max_idx_trn].mean(axis=1, keepdims=True)
        data = graph_signals - in_sample_means
        data_dict["in_sample_means"] = in_sample_means
    else:
        data = graph_signals

    for i in range(3):

        split_data = data[:, split_idx[i]]
        data_points = []
        targets = []

        for j in range(len(split_idx[i])):
            try:
                targets.append(
                    split_data[
                        :, list(range(j + obs_window, j + obs_window + pred_horizen))
                    ]
                )
                data_points.append(split_data[:, list(range(j, j + obs_window))])
            except:
                break

        data_dict[data_type[i]] = {
            "data": np.stack(data_points, axis=0),
            "labels": np.stack(targets, axis=0),
        }

    print("dataset has been created.")
    print("-------------------------")
    print(f"{data_dict['trn']['data'].shape[0]} train data points")
    print(f"{data_dict['val']['data'].shape[0]} validation data points")
    print(f"{data_dict['tst']['data'].shape[0]} test data points")

    return data_dict


# ! From Lab
def create_knn_graph(A, k):
    rows, cols = A.shape
    G = nx.Graph()

    for i in range(rows):
        # Get the indices and values of the k-nearest neighbors
        indices = np.nonzero(A[i])[0]
        row_data = [(idx, A[i][idx]) for idx in indices]
        row_data.sort(key=lambda x: x[1])
        nearest_neighbors = row_data[:k]

        for j, weight in nearest_neighbors:
            # if weight != 0:
            G.add_edge(i, j, weight=weight)

    return G


def create_dataset(x, y, edge_index):
    dataset = []
    for idx, x in enumerate(x):
        x_ = torch.tensor(x, dtype=torch.float)
        y_ = torch.tensor(y[idx], dtype=torch.float).squeeze()
        data = (x_, y_, edge_index)
        dataset.append(data)
    return dataset


def train_epoch_gcnn(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for sample in loader:
        x = sample[0].reshape(n_stations, obs_window)
        y = sample[1].reshape(n_stations, -1)
        out = model(shift_operator, x)
        # print(out)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch_gcnn(model, loader, criterion):
    model.eval()
    total_loss = 0

    for sample in loader:
        x = sample[0].reshape(n_stations, obs_window)
        y = sample[1].reshape(n_stations, -1)
        out = model(shift_operator, x)
        loss = criterion(out, y)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_gcn(model, num_epochs, criterion, train_loader, test_loader):
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

        if epoch % 1 == 0:
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

    size = 100

    def _create_missing_dataset(data, remaining_frac=0.8):
        remaining_data = data.clone()  # Clone the data to avoid modifying the original

        num_samples, num_nodes, num_features = remaining_data.shape

        # Determine the number of features to keep
        num_features_to_keep = int(remaining_frac * num_features)

        # Create a mask to zero out features
        feature_mask = torch.ones(num_features, dtype=torch.bool)

        if remaining_frac < 1.0:
            # Randomly select indices to zero out
            zero_out_indices = torch.randperm(num_features)[num_features_to_keep:]
            feature_mask[zero_out_indices] = False

        # Apply the mask to each sample and each node's features
        for sample in range(num_samples):
            for node in range(num_nodes):
                remaining_data[sample, node] = (
                    remaining_data[sample, node] * feature_mask
                )

        return remaining_data

    data_fractions = [1.0, 0.6, 0.4, 0.1, 0.01, 0.001]
    for frac in data_fractions:

        train_set = dataset["trn"]
        val_set = dataset["val"]
        test_set = dataset["tst"]

        X_train = torch.Tensor(train_set["data"][:size])
        y_train = torch.Tensor(train_set["labels"][:size])

        X_val = torch.Tensor(val_set["data"][:size])
        y_val = torch.Tensor(val_set["labels"][:size])

        X_test = torch.Tensor(test_set["data"][:size])
        y_test = torch.Tensor(test_set["labels"][:size])

        X_train = _create_missing_dataset(X_train, frac)
        X_val = _create_missing_dataset(X_val, frac)
        X_test = _create_missing_dataset(X_test, frac)

        A = np.load(file="gmm_code/lab2_NOAA/NOA_109_original_adj.npy")

        k = 10
        G = create_knn_graph(A, k)

        A = torch.Tensor(nx.to_numpy_array(G))

        shift_operator = torch.tensor(A, dtype=torch.float32)
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

        train_dataset = create_dataset(X_train, y_train, edge_index)
        val_dataset = create_dataset(X_val, y_val, edge_index)
        test_dataset = create_dataset(X_test, y_test, edge_index)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = GMMGCN(
            obs_size=obs_window,
            pred_size=pred_horizen,
            hid_sizes=[16, 16],
            num_components=5,
            all_features=X_train[0, :, :],
            all_A=A,
        )

        train_gcn(model, 10, torch.nn.MSELoss(), train_loader, test_loader)
