import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # use Xavier initialization to match variance of input with output
        nn.init.xavier_uniform_(self.weights)

    def forward(self, shift, features):
        return shift.mm(features.mm(self.weights))


# input (num_nodes, obs_size) matrix
# shift operator (num_nodes, num_nodes) matrix
# output (num_nodes, pred_size) matrix
class GCN(nn.Module):
    def __init__(self, obs_size, pred_size, hid_sizes):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer of size obs_size
        self.layers.append(GCNLayer(obs_size, hid_sizes[0]))
        # num_hid hidden layers of size hid_size
        for i in range(len(hid_sizes)-1):
            self.layers.append(GCNLayer(hid_sizes[i], hid_sizes[i+1]))
        # fully connected layer to get  output of dim pred_size
        self.layers.append(nn.Linear(hid_sizes[-1], pred_size))

    def forward(self, shift, features):
        temp = features
        for layer in self.layers[-1]:
            # use relu activation function
            temp = F.relu(layer(shift, temp))
        return self.layers[-1](shift, temp)
