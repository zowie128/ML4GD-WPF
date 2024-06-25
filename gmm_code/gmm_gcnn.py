import os
import sys

# Set the PYTHONPATH to the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gcnn import GCNNLayer
from gmm_code.gmm_gcn import GMMGCNLayer
from gmm_code.utils import ex_relu

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMGCNNLayer(GMMGCNLayer):
    """This extends the GMMGCNLayer, adding order q. Forward pass LMW becomes Sum over q of L^qMW."""

    def __init__(
        self,
        in_features,
        out_features,
        num_components,
        all_features,
        all_A,
        order,
        device="cpu",
    ) -> None:
        super(GMMGCNNLayer, self).__init__(
            in_features, out_features, num_components, all_features, all_A, device
        )
        self.all_A = all_A
        self.order = order

    def forward(self, shift, features):
        batch_size = features.size(0)
        # output_dim = self.weights.size(1)
        # shift_powers = [torch.matrix_power(shift, k).float() for k in range(self.order)]

        tensor_list = []
        for i in range(batch_size):
            out = self._forward(shift, features[i])
            tensor_list.append(out)
        return torch.stack(tensor_list, dim=0)

    def _forward(self, shift, features):
        x_imp = features.repeat(self.num_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.sigma)
        # M
        mean_mat = torch.where(
            x_isnan, self.mu.repeat((features.size(0), 1, 1)).permute(1, 0, 2), x_imp
        )
        # S
        var_mat = torch.where(
            x_isnan,
            variances.repeat((features.size(0), 1, 1)).permute(1, 0, 2),
            torch.zeros(size=x_imp.size(), device=self.device, requires_grad=True),
        )

        # M^kW
        transform_x = torch.matmul(mean_mat, self.weight)
        # S^k (W * W)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            # First:
            # LM^kW
            # conv_x.append(torch.spmm(shift, component_x))

            # Becomes:
            # sum over Q of (L^q M^k W)
            out = torch.zeros(component_x.size(0), self.weight.size(1))
            # compute order hop shift
            for q in range(self.order):
                out += (
                    torch.spmm(torch.matrix_power(shift, q), component_x)
                ) / self.order
            conv_x.append(out)

        for component_covs in transform_covs:
            # First:
            # (L*L) S^k (W * W)
            # conv_covs.append(torch.spmm(self.A2, component_covs))

            # Becomes:
            # sum over Q of ( (L^q * L^q) S^k (W * W) )
            out = torch.zeros(component_covs.size(0), self.weight.size(1))
            # compute order hop shift
            for q in range(self.order):
                # A2 explodes to zero
                A2 = torch.mul(
                    torch.matrix_power(self.all_A, q), torch.matrix_power(self.all_A, q)
                ).to(self.device)

                out += (torch.spmm(A2, component_covs)) / self.order

            conv_covs.append(out)

        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        # ReLU[N(M, S)]
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self._calc_responsibility(mean_mat, variances)
        # ReLU[(LXW)]
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        # Check for NaNs
        if torch.isnan(expected_x).any():
            print("NaN detected in expected_x:")
        return expected_x


class GMMGCNN(nn.Module):

    def __init__(
        self, obs_size, pred_size, hid_sizes, num_components, all_features, all_A, order
    ):
        super(GMMGCNN, self).__init__()
        self.layers = nn.ModuleList()

        # First layer is a GCNN that incorporates the GMM.
        self.layers.append(
            GMMGCNNLayer(
                obs_size, hid_sizes[0], num_components, all_features, all_A, order
            )
        )

        # Later layers are regular GCNs.
        # num_hid hidden layers of size hid_size
        for i in range(len(hid_sizes) - 1):
            self.layers.append(GCNNLayer(hid_sizes[i], hid_sizes[i + 1], order))
        # fully connected layer to get  output of dim pred_size
        self.layers.append(nn.Linear(hid_sizes[-1], pred_size))

    def forward(self, shift, features):
        # print(f"forward in: {features}")
        temp = features
        # No relu for the first layer, it is a GMM layer.
        temp = self.layers[0].forward(shift, features)
        # print(f"after gmmgcnn layer: {temp}")
        for layer in self.layers[1:-1]:
            # use relu activation function
            temp = F.relu(layer(shift, temp))
        # Last layer no relu
        return self.layers[-1](temp)
