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
        out = torch.zeros(features.size(0), self.weights.size(1))
        # compute order hop shift
        for k in range(self.order):
            out += torch.matrix_power(shift, k).mm(features.mm(self.weights[:, :, k]))
        return out


# Inputs must be sized [num_nodes, obs_size] and outputs will be [num_nodes, pred_size]
class GCNN(nn.Module):
    def __init__(self, obs_size, pred_size, hid_sizes, order):
        super(GCNN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer of size obs_size
        self.layers.append(GCNNLayer(obs_size, hid_sizes[0], order))
        # num_hid hidden layers of size hid_size
        for i in range(len(hid_sizes) - 1):
            self.layers.append(GCNNLayer(hid_sizes[i], hid_sizes[i + 1], order))
        # fully connected layer to get  output of dim pred_size
        self.layers.append(nn.Linear(hid_sizes[-1], pred_size))

    # Forward sample by sample, no batches are yet supported
    def forward(self, shift, features):
        temp = features
        for layer in self.layers[:-1]:
            # use relu activation function
            temp = F.relu(layer(shift, temp))
        return self.layers[-1](temp)
