import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from numpy.linalg import det


def get_features(model, dataloader):
    total = 0
    model.eval()
    
    for batch_idx, (img, label) in enumerate(dataloader):
        # do not pass to cuda 
        # set up the initial batch
        if batch_idx == 0:
            f = model(img)  # Forward pass
            train_feature = f.detach().numpy()  # Convert to NumPy
        # subsequent batch
        else:
            f = model(img)
            train_feature = np.concatenate((train_feature, f.detach().numpy()))
    return train_feature


def mahalanobis_distance(train_feature,e):
    centerpoint = np.mean(train_feature , axis=0)  
    p1 = e
    p2 = centerpoint
    # covariance matrix
    covariance  = np.cov(train_feature, rowvar=False)
    if det(covariance) != 0:
        # covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    else:
        covariance_pm1 = np.linalg.pinv(covariance)
    return (p1-p2).T.dot(covariance_pm1).dot(p1-p2)


def evaluate(net,trainloader,validloader,testloader):
    net.eval()  
    train_feature= get_features(net,trainloader)
    auroc_max = 0
    with torch.no_grad():         
        for batch_idx, (images, label) in enumerate(validloader):
            # do not pass to cuda 
            if batch_idx == 0:
                embedding = net(images)
                embedding = np.array(embedding)
            else:
                emb = net(images)
                embedding = np.concatenate((embedding,np.array(emb)))
        #print(embedding.shape)
        distances = []
        for e in embedding:
            distance = mahalanobis_distance(train_feature,e)
            distances.append(distance)
        distances = np.array(distances)
        
        for batch_idx, (images, label) in enumerate(testloader):
            # do not pass to cuda 
            if batch_idx == 0:
                embedding = net(images)
                embedding = np.array(embedding)
                labels = np.array(label)
            else:
                emb = net(images)
                embedding = np.concatenate((embedding,np.array(emb)))
                labels = np.concatenate((labels,label))
        #print(embedding.shape)
        distances_test = []
        for e in embedding:
            distance_test = mahalanobis_distance(train_feature,e)
            distances_test.append(distance_test)
        distances_test = np.array(distances_test)

        for percentile in range(85,86):
            y_true = []
            y_pred = []
            total_correct = 0
            #print("percentile:",percentile)
            cutoff = np.percentile(distances,percentile)
            pred = distances_test > cutoff
            pred = pred.astype(int)
            for i in labels:
                y_true.append(i.item())
            for i in pred:
                y_pred.append(i.item())
            pred = torch.tensor(pred)
            
            labels = torch.tensor(labels)
            total_correct += torch.sum(pred == labels).item()

            cm = confusion_matrix(y_true,y_pred)
            
            accuracy = total_correct / len(testloader.dataset)
            
            # AUROC score 
            AUROC = roc_auc_score(y_true, distances_test)
       
            # AUC-PR score 
            PR_AUC = average_precision_score(y_true, distances_test)
            
    return AUROC, PR_AUC