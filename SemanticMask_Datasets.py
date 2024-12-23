import numpy as np
import torch
import random
from torch.utils.data import Dataset

def augmentation(x_index, cluster_assignment, reduced_cluster, number, prob_pert):
    # f_label = cluster_assignment
    # f_cluster = reduced_cluster
    random.shuffle(reduced_cluster)
    cluster = reduced_cluster[:number]
    not_cluster = reduced_cluster[number:]
    mask = []
    for i in cluster:
        m_index = np.where(cluster_assignment==i)[0]
        m = np.random.binomial(1,prob_pert,len(m_index))>0
        m_index=m_index[m]  # select some 0's to be flipped to 1's
        mask.extend(m_index)
    augmented_x = x_index.clone()
    for i in range(len(x_index)):
        if i in mask:
            augmented_x[i]=0

    return augmented_x, not_cluster

def augmentation_description(x_index, cluster_assignment, reduced_cluster, number):
    # f_label = cluster_assignment
    # f_cluster = reduced_cluster
    random.shuffle(reduced_cluster)
    cluster = reduced_cluster[:number]
    not_cluster = reduced_cluster[number:]
    mask = []
    for i in cluster:
        if i == 1:
            prob_pert=0.3
        else:
            prob_pert=0.5
        m_index = np.where(cluster_assignment==i)[0]
        m = np.random.binomial(1,prob_pert,len(m_index))>0
        m_index=m_index[m]  # select some 0's to be flipped to 1's
        mask.extend(m_index)
    augmented_x = x_index.clone()
    for i in range(len(x_index)):
        if i in mask:
            augmented_x[i]=0

    return augmented_x, not_cluster

def augmentation_position(x_index, cluster_assignment, reduced_cluster, number, prob_pert):
    # f_label = cluster_assignment
    # f_cluster = reduced_cluster
    random.shuffle(reduced_cluster)
    cluster = reduced_cluster[:number]
    not_cluster = reduced_cluster[number:]
    mask = []
    for i in cluster:
        m_index = np.where(cluster_assignment==i)[0]
        m = np.random.binomial(1,prob_pert,len(m_index))>0
        m_index=m_index[m]  # select some 0's to be flipped to 1's
        mask.extend(m_index)
    augmented_x = x_index.clone()
    for i in range(len(x_index)):
        if i in mask:
            augmented_x[i]=0
    m_new = 1 * (x_index != augmented_x)
    return augmented_x, not_cluster, m_new

class SMDataset_withClusters(Dataset):
    # Initialize a dataset based on X, y data and column clustered labels
    def __init__(self, X, y, cluster_assignment, pm = 0.4):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.cluster_assignment = cluster_assignment
        self.pm = pm

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        X_index = self.X[index]
        y_index = self.y[index]
        pm = self.pm
        cluster_assignment = self.cluster_assignment # f_label in source code
        reduced_cluster = list(set(cluster_assignment)) # f_cluster in source code
        number = len(reduced_cluster) // 2
        x_tilde1, cluster_remain = augmentation(X_index, cluster_assignment, reduced_cluster, number, pm) # cluster_remain is f_remain in source code
        x_tilde2, cluster_remain = augmentation(X_index, cluster_assignment, cluster_remain, number, pm)
        x = [x_tilde1,x_tilde2]
        return x, y_index

class SMDataset_Description(Dataset):
    # Initialize a dataset based on X, y data and column clustered labels
    def __init__(self, X, y, cluster_assignment):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.cluster_assignment = cluster_assignment

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        X_index = self.X[index]
        y_index = self.y[index]
        cluster_assignment = self.cluster_assignment # f_label in source code
        reduced_cluster = list(set(cluster_assignment)) # f_cluster in source code
        number = len(reduced_cluster) // 2
        x_tilde1, cluster_remain = augmentation_description(X_index, cluster_assignment, reduced_cluster, number) # cluster_remain is f_remain in source code
        x_tilde2, cluster_remain = augmentation_description(X_index, cluster_assignment, cluster_remain, number)
        x = [x_tilde1,x_tilde2]
        return x, y_index

class SMDataset_Position(Dataset):

    def __init__(self, X, y, cluster_assignment, pm = 0.4):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.cluster_assignment = cluster_assignment
        self.pm = pm

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        X_index = self.X[index]
        y_index = self.y[index]
        pm = self.pm
        cluster_assignment = self.cluster_assignment # f_label in source code
        reduced_cluster = list(set(cluster_assignment)) # f_cluster in source code
        number = len(reduced_cluster) // 2
        x_tilde1, cluster_remain, m1 = augmentation_position(X_index, cluster_assignment, reduced_cluster, number, pm) # cluster_remain is f_remain in source code
        x_tilde2, cluster_remain, m2 = augmentation_position(X_index, cluster_assignment, cluster_remain, number, pm)
        x = [x_tilde1,x_tilde2]
        m = [m1, m2]
        return x, y_index, m

class SMDataset(Dataset):
    # Initialize a dataset based on X, y 
    def __init__(self, X, y, pm = 0.4):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.pm = pm
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        X_index = self.X[index]
        y_index = self.y[index]
        return X_index,y_index