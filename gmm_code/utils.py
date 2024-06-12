import numpy as np
import torch
import torch.nn.functional as F

import networkx as nx


# FOR GMM Implementation
# ! Taken from the paper.
def ex_relu(mu, sigma):
    is_zero = sigma == 0
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (
        torch.div(torch.exp(torch.div(-w * w, 2)), np.sqrt(2 * np.pi))
        + torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2))))
    )
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


##################################################
# FOR TESTING ON NOAA DATASET FROM LAB


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
        x_ = x.clone().detach()
        y_ = y[idx].clone().detach()
        data = (x_, y_, edge_index)
        dataset.append(data)
    return dataset
