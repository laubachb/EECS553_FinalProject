import numpy as np
import torch
import train_heart
import torch.optim as optimize
from SemanticMask_Datasets import SMDataset_withClusters, SMDataset
import evaluation
import prep_data


#Generate and load heart files to data tab - comment out if already loaded
# x_train, y_train, x_valid, y_valid, x_test, y_test = prep_data.process_dataset('heart')
# np.save('heart_data/X_train_heart.npy', x_train)  
# np.save('heart_data/y_train_heart.npy', y_train)    #all zeros 
# np.save('heart_data/X_valid_heart.npy', x_valid)    
# np.save('heart_data/y_valid_heart.npy', y_valid)    #all zeros 
# np.save('heart_data/X_test_heart.npy', x_test)   
# np.save('heart_data/y_test_heart.npy', y_test)  


# Load heart data files 
X_train = np.load('heart_data/X_train_heart.npy')   
y_train = np.load('heart_data/y_train_heart.npy')  
X_valid = np.load('heart_data/X_valid_heart.npy')    
y_valid = np.load('heart_data/y_valid_heart.npy')
X_test = np.load('heart_data/X_test_heart.npy')   
y_test = np.load('heart_data/y_test_heart.npy')
cluster_assignment = np.load('heart_data/heart_clusters.npy')

# Create torch dataset objects for train and test sets using the following files:
# - SemanticMask_Datasets.py
list_auroc = []
list_pr_auc = []
for pm in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
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
    net = train_heart.Encoder()
    optimizer = optimize.Adam(net.parameters(), lr = 0.001)
    trainloader_SemanticMask = torch.utils.data.DataLoader(data_train_sm,batch_size=151)  
    net, training_loss = train_heart.train_dnn(net,0.01,1000,optimizer,trainloader_SemanticMask)


    auroc, pr_auc = evaluation.evaluate(net, train_dataset, validation_dataset, test_dataset)
    list_auroc.append(auroc)
    list_pr_auc.append(pr_auc)
print(list_auroc)
print(list_pr_auc)
