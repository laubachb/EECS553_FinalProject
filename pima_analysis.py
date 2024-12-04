import numpy as np
import torch
import train_pima
import torch.optim as optimize
from SemanticMask_Datasets import SMDataset_withClusters, SMDataset
import evaluation
import prep_data

'''
Generate and load Pima files to data tab - comment out if already loaded
x_train, y_train, x_valid, y_valid, x_test, y_test = prep_data.process_dataset('pima')
np.save('pima_data/X_train_pima.npy', x_train)  
np.save('pima_data/y_train_pima.npy', y_train)    #all zeros 
np.save('pima_data/X_valid_pima.npy', x_valid)    
np.save('pima_data/y_valid_pima.npy', y_valid)    #all zeros 
np.save('pima_data/X_test_pima.npy', x_test)   
np.save('pima_data/y_test_pima.npy', y_test)  
'''
# Load Pima data files 
X_train = np.load('pima_data/X_train_pima.npy')   
y_train = np.load('pima_data/y_train_pima.npy')  
X_valid = np.load('pima_data/X_valid_pima.npy')    
y_valid = np.load('pima_data/y_valid_pima.npy')
X_test = np.load('pima_data/X_test_pima.npy')   
y_test = np.load('pima_data/y_test_pima.npy')
cluster_assignment = np.load('pima_data/pima_clusters.npy')

# Create torch dataset objects for train and test sets using the following files:
# - SemanticMask_Datasets.py
data_train_sm = SMDataset_withClusters(X_train, y_train, cluster_assignment)
data_train = SMDataset(X_train, y_train)
data_test = SMDataset(X_test, y_test)
data_validation = SMDataset(X_valid, y_valid)

# Load in datasets using torch
train_dataset = torch.utils.data.DataLoader(dataset=data_train,batch_size=151)
test_dataset = torch.utils.data.DataLoader(dataset=data_test,batch_size=75)
validation_dataset = torch.utils.data.DataLoader(dataset=data_validation,batch_size=256)

# Fit NN model using the following files:
# - train.py
# - Loss.py

# Generate and fit model with standard SemanticMask
net = train_pima.Encoder()
optimizer = optimize.Adam(net.parameters(), lr = 0.001)
trainloader_SemanticMask = torch.utils.data.DataLoader(data_train_sm,batch_size=151)  
net, training_loss = train_pima.train_dnn(net,0.01,1000,optimizer,trainloader_SemanticMask)
auroc = evaluation.evaluate(net, train_dataset, validation_dataset, test_dataset)
print(auroc)
