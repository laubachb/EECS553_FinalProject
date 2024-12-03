import numpy as np
import torch
from torch.utils.data import Dataset

class SMDataset_withClusters(Dataset):
    # Initialize a dataset based on X, y data and column clustered labels
    def __init__(self, X, y, cluster_assignment):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.cluster_assignment = torch.from_numpy(cluster_assignment)

class SMDataset(Dataset):
    # Initialize a dataset based on X, y 
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)