import numpy as np
import torch
import random
from torch.utils.data import Dataset

def augmentation(x_index, cluster_assignment, reduced_cluster, number):
    # f_label = cluster_assignment
    # f_cluster = reduced_cluster
    random.shuffle(reduced_cluster)


    return

class SMDataset_withClusters(Dataset):
    # Initialize a dataset based on X, y data and column clustered labels
    def __init__(self, X, y, cluster_assignment):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.cluster_assignment = cluster_assignment

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idex):
        if torch.is_tensor(index):
            index = index.tolist()
        X_index = self.X[index]
        y_index = self.y[index]
        cluster_assignment = self.cluster_assignment # f_label in source code
        reduced_cluster = list(set(cluster_assignment)) # f_cluster in source code
        number = len(reduced_cluster) // 2
        x_tilde1, cluster_remain = augmentation(X_index, cluster_assignment, reduced_cluster, number) # cluster_remain is f_remain in source code
        return

class SMDataset(Dataset):
    # Initialize a dataset based on X, y 
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X)