import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNNLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super(GCNNLayer, self).__init__()
        self.order = order
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features, order))
        # use Xavier initialization to match variance of input with output
        nn.init.xavier_uniform_(self.weights)

    def forward(self, shift, features):
        batch_size = features.size(0)
        output_dim = self.weights.size(1)
        shift_powers = [torch.matrix_power(shift, k).float() for k in range(self.order)]

        tensor_list = []
        for i in range(batch_size):
            out = torch.zeros((features.size(1), output_dim), device=features.device)
            for k in range(self.order):
                weighted_features = torch.matmul(features[i], self.weights[:, :, k])
                shift_k = shift_powers[k]
                shifted_features = torch.matmul(shift_k, weighted_features)
                out += shifted_features
            tensor_list.append(out)
        return torch.stack(tensor_list, dim=0)


# Inputs must be sized [num_nodes, obs_size] and outputs will be [num_nodes, pred_size]
class GCNN(nn.Module):
    def __init__(self, hid_sizes, order):
        super(GCNN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer of size obs_size
        self.layers.append(GCNNLayer(1, hid_sizes[0], order))
        # num_hid hidden layers of size hid_size
        for i in range(len(hid_sizes) - 1):
            self.layers.append(GCNNLayer(hid_sizes[i], hid_sizes[i + 1], order))
        # fully connected layer to get  output of dim pred_size
        self.layers.append(nn.Linear(hid_sizes[-1], 1))

    # Forward sample by sample, no batches are yet supported
    def forward(self, shift, features):
        temp = features
        for layer in self.layers[:-1]:
            # use relu activation function
            temp = F.relu(layer(shift, temp))
        return self.layers[-1](temp)
