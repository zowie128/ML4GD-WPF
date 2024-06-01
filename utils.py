import numpy as np
import torch


def normalize_adjacency(A):
    I = torch.eye(A.size(0))
    A_hat = A + I
    D = torch.diag(torch.sum(A_hat, dim=1))
    D_inv_sqrt = torch.diag(torch.pow(D.diag(), -0.5))
    S = torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)
    return S


def create_param_prod_graph(s, S_N, S_T):
    return torch.tensor(sum([s[i, j] * np.kron(torch.matrix_power(S_T, i), np.matrix_power(S_N, j))
                             for i in range(2) for j in range(2)]))
