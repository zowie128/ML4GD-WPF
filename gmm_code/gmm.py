import os
import sys

# Set the PYTHONPATH to the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gcn import GCNLayer
from gcnn import GCNNLayer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture

device = torch.device("cpu")


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


class GMMGCNLayer(nn.Module):

    def __init__(
        self, in_features, out_features, num_components, all_features, all_A
    ) -> None:
        super(GMMGCNLayer, self).__init__()

        self.num_components = num_components
        self.in_features = in_features
        self.out_features = out_features

        # All feature data, used for initialization of the GMM.
        self.all_features = all_features
        self.A2 = torch.mul(all_A, all_A).to(device)

        # Initialize the weights
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        # Initialize GMM and its parameters (we're going to learn these)
        self.gmm = self._init_gmm(self.all_features, self.num_components)
        self.pi = nn.Parameter(torch.FloatTensor(np.log(self.gmm.weights_))).to(device)
        self.mu = nn.Parameter(torch.FloatTensor(self.gmm.means_).to(device))
        self.sigma = torch.FloatTensor(np.log(self.gmm.covariances_)).to(device)

    def _init_gmm(self, features, K):
        # Simply impute the values once for initialization of the gmm.
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputed_x = imputer.fit_transform(features)
        gmm = GaussianMixture(n_components=K, covariance_type="diag").fit(imputed_x)
        return gmm

    # ! Taken from the paper.
    def _calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (
            (-1 / 2)
            * torch.sum(
                torch.pow(mean_mat - self.mu.unsqueeze(1), 2) / variances.unsqueeze(1),
                2,
            )
            - (dim / 2) * np.log(2 * np.pi)
            - (1 / 2) * torch.sum(self.sigma)
        )
        log_prob = self.pi.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    # ! Taken from the paper.
    def forward(self, shift, features):
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
            torch.zeros(size=x_imp.size(), device=device, requires_grad=True),
        )

        # M^kW
        transform_x = torch.matmul(mean_mat, self.weight)
        # S^k (W * W)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            # LM^kW
            conv_x.append(torch.spmm(shift, component_x))
        for component_covs in transform_covs:
            # (L*L) S^k (W * W)
            conv_covs.append(torch.spmm(self.A2, component_covs))

        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        # ReLU[N(M, S)]
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self._calc_responsibility(mean_mat, variances)
        # ReLU[(LXW)]
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x


class GMMGCN(nn.Module):
    def __init__(
        self,
        obs_size,
        pred_size,
        hid_sizes,
        num_components,
        all_features,
        all_A,
    ):
        super(GMMGCN, self).__init__()
        self.layers = nn.ModuleList()

        # First layer is a incorporates the GMM.
        self.layers.append(
            GMMGCNLayer(obs_size, hid_sizes[0], num_components, all_features, all_A)
        )

        # Later layers are regular GCNs.
        # num_hid hidden layers of size hid_size
        for i in range(len(hid_sizes) - 1):
            self.layers.append(GCNLayer(hid_sizes[i], hid_sizes[i + 1]))
        # fully connected layer to get  output of dim pred_size
        self.layers.append(nn.Linear(hid_sizes[-1], pred_size))

    def forward(self, shift, features):
        temp = features
        # No relu for the first layer, it is a GMM layer.
        temp = self.layers[0].forward(shift, features)
        for layer in self.layers[1:-1]:
            # use relu activation function
            temp = F.relu(layer(shift, temp))
        # Last layer no relu
        return self.layers[-1](temp)


class GMMGCNNLayer(GMMGCNLayer):
    """This extends the GMMGCNLayer, adding order q. Forward pass LMW becomes Sum over q of L^qMW."""

    def __init__(
        self, in_features, out_features, num_components, all_features, all_A, order
    ) -> None:
        super(GMMGCNNLayer, self).__init__(
            in_features, out_features, num_components, all_features, all_A
        )
        self.all_A = all_A
        self.order = order

    def forward(self, shift, features):
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
            torch.zeros(size=x_imp.size(), device=device, requires_grad=True),
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
                A2 = torch.mul(
                    torch.matrix_power(self.all_A, q), torch.matrix_power(self.all_A, q)
                ).to(device)

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
            self.layers.append(GCNLayer(hid_sizes[i], hid_sizes[i + 1]))
        # fully connected layer to get  output of dim pred_size
        self.layers.append(nn.Linear(hid_sizes[-1], pred_size))

    def forward(self, shift, features):
        temp = features
        # No relu for the first layer, it is a GMM layer.
        temp = self.layers[0].forward(shift, features)
        for layer in self.layers[1:-1]:
            # use relu activation function
            temp = F.relu(layer(shift, temp))
        # Last layer no relu
        return self.layers[-1](temp)
