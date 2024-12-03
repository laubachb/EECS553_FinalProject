import numpy as np

X_train = np.load('data/X_train_saheart.npy')   
y_train = np.load('data/y_train_saheart.npy')  
X_valid = np.load('data/X_valid_saheart.npy')    
y_valid = np.load('data/y_valid_saheart.npy')
X_test = np.load('data/X_test_saheart.npy')   
y_test = np.load('data/y_test_saheart.npy')

cluster_assignment = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0])  