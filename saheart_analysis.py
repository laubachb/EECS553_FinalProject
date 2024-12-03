import numpy as np
from SemanticMask_Datasets import SMDataset_labeled, SMDataset_unlabeled

# Load in SA Heart Data from .npy files
# Load in cluster assignments generated using the following files:
# - saheart_generate_clusters.py
X_train = np.load('data/X_train_saheart.npy')   
y_train = np.load('data/y_train_saheart.npy')  
X_valid = np.load('data/X_valid_saheart.npy')    
y_valid = np.load('data/y_valid_saheart.npy')
X_test = np.load('data/X_test_saheart.npy')   
y_test = np.load('data/y_test_saheart.npy')
cluster_assignment = np.load('data/saheart_clusters.npy')

# Create torch dataset objects for train and test sets using the following files:
# - SemanticMask_Datasets.py
data_train_sm = SMDataset_labeled(X_train,y_train,cluster_assignment)
data_train = SMDataset_unlabeled(X_train, y_train)
data_test = SMDataset_unlabeled(X_test, y_test)
data_validation = SMDataset_unlabeled(X_valid, y_valid)

# Load in datasets using torch
train_dataset = torch.utils.data.DataLoader(dataset=data_train,batch_size=151)
test_dataset = torch.utils.data.DataLoader(dataset=data_test,batch_size=75)
validation_dataset = torch.utils.data.DataLoader(dataset=data_validation,batch_size=256)

# Fit NN model using the following files:

# Hyperparameters
temperature = 0.01
epochs = 1000