import numpy as np
import torch


def normalize_adjacency(A):
    I = torch.eye(A.size(0))
    A_hat = A + I
    D = torch.diag(torch.sum(A_hat, dim=1))
    D_inv_sqrt = torch.diag(torch.pow(D.diag(), -0.5))
    S = torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)
    return S

def build_parametric_product_graph(S_0, S_1, h_00, h_01, h_10, h_11):
    I_0 = np.eye(S_0.shape[1])
    I_1 = np.eye(S_1.shape[1])

    S_kron_II = torch.from_numpy(np.kron(I_0, I_1))
    S_kron_SI = torch.from_numpy(np.kron(S_0, I_1))
    S_kron_IS = torch.from_numpy(np.kron(I_0, S_1))
    S_kron_SS = torch.from_numpy(np.kron(S_0, S_1)).double()

    S = h_00 * S_kron_II + \
        h_01 * S_kron_IS + \
        h_10 * S_kron_SI + \
        h_11 * S_kron_SS
    return S
