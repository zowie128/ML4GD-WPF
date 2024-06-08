import os
import sys

# Set the PYTHONPATH to the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gcn import GCNLayer

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

        self.K = num_components
        self.in_features = in_features
        self.out_features = out_features

        # All feature data, used for initialization of the GMM.
        self.all_features = all_features
        self.A2 = torch.mul(all_A, all_A).to(device)

        # Initialize the weights
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        # Initialize GMM and its parameters (we're going to learn these)
        self.gmm = self._init_gmm(self.all_features, self.K)
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
        x_imp = features.repeat(self.K, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.sigma)
        mean_mat = torch.where(
            x_isnan, self.mu.repeat((features.size(0), 1, 1)).permute(1, 0, 2), x_imp
        )
        var_mat = torch.where(
            x_isnan,
            variances.repeat((features.size(0), 1, 1)).permute(1, 0, 2),
            torch.zeros(size=x_imp.size(), device=device, requires_grad=True),
        )

        transform_x = torch.matmul(mean_mat, self.weight)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            conv_x.append(torch.spmm(shift, component_x))
        for component_covs in transform_covs:
            conv_covs.append(torch.spmm(self.A2, component_covs))
        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self._calc_responsibility(mean_mat, variances)
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x


class GMMGCN(nn.Module):
    def __init__(
        self, obs_size, pred_size, hid_sizes, num_components, all_features, all_A
    ):
        super(GMMGCN, self).__init__()
        self.layers = nn.ModuleList()

        # First layer is a incorporates the GMM.
        self.layers.append(
            GMMGCNLayer(obs_size, hid_sizes[0], num_components, all_features, all_A)
        )

        # Later layers are regular.
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
        # This is a linear layer??
        return self.layers[-1](temp)


class GMMGCNTrainer:
    def __init__(self, model, lr=0.01, weight_decay=5e-4):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, train_loader, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                features, A, targets = data
                features, A, targets = (
                    features.to(device),
                    A.to(device),
                    targets.to(device),
                )

                print(features.shape)
                print(A.shape)
                print(targets.shape)

                self.optimizer.zero_grad()
                outputs = self.model(A, features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        print(f"Test Loss: {total_loss/len(test_loader)}")


if __name__ == "__main__":
    pass

    # # Remove data for testing
    # def _create_missing_dataset(data, remaining_frac=0.8):
    #     remaining_data = data.clone()  # Clone the data to avoid modifying the original

    #     num_features = remaining_data.shape[1]
    #     num_nodes = remaining_data.shape[0]

    #     # Determine the number of features to keep
    #     num_features_to_keep = int(remaining_frac * num_features)

    #     # Create a mask to zero out features
    #     feature_mask = torch.ones(num_features, dtype=torch.bool)

    #     if remaining_frac < 1.0:
    #         # Randomly select indices to zero out
    #         zero_out_indices = torch.randperm(num_features)[num_features_to_keep:]
    #         feature_mask[zero_out_indices] = False

    #     # Apply the mask to each node's features
    #     for node in range(num_nodes):
    #         remaining_data[node] = remaining_data[node] * feature_mask

    #     return remaining_data

    #     # Data set with missing data

    # data_fractions = [1.0, 0.6, 0.4, 0.1]
    # for frac in data_fractions:
    #     missing_data_train = _create_missing_dataset(X_train, frac)
    #     missing_data_test = _create_missing_dataset(X_test, frac)

    #     print(missing_data_train.shape)

    #     all_A = to_dense_adj(data.edge_index)[0]

    #     train_A = all_A[data.train_mask][:, data.train_mask]
    #     test_A = all_A[data.test_mask][:, data.test_mask]

    #     print(train_A.shape)

    #     train_dataset = TensorDataset(missing_data_train, train_A, y_train)
    #     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    #     test_dataset = TensorDataset(missing_data_test, test_A, y_test)
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #     model = GMMGCN(
    #         obs_size=cora_dataset.num_features,
    #         pred_size=cora_dataset.num_classes,
    #         hid_sizes=[16, 16],
    #         num_components=5,
    #         all_features=missing_data_train,
    #         all_A=all_A,
    #     )

    #     trainer = GMMGCNTrainer(model=model)
    #     trainer.train(train_loader=train_loader, epochs=100)
    #     trainer.evaluate(test_loader=test_loader)
