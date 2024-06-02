import numpy as np

class ProductGraph:
    
    def __init__(self, adjacency_matrix, S_T):
        self.S = adjacency_matrix
        self.S_T = S_T
        
    def generate_Kronecker(self):
        return np.kron(self.S_T, self.S)
        
    def generate_Cartesian(self):
        I_N = np.eye(self.S.shape[0])
        I_T = np.eye(self.S_T.shape[0])
        return np.kron(self.S_T, I_N) + np.kron(I_T, self.S)
        
    def generate_Strong(self):
        return self.generate_Cartesian() + self.generate_Kronecker()
        
    def generate_parametric(self):
        I_T = np.eye(self.S_T.shape[0])
        I_N = np.eye(self.S.shape[0])
        return np.kron(I_T, I_N) + self.generate_Strong()
