import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score


def extract_features(model, dataloader):
    model.eval()
    train_features = []

    with torch.no_grad():
        for data, _ in dataloader:
            f = model(data)[0].detach().numpy()
            train_features.append(f)

    return np.concatenate(train_features, axis=0)

def mahalanobis_distance(train_features, embeddings):
    centerpoint = np.mean(train_features, axis=0)
    covariance = np.cov(train_features, rowvar=False)

    # check determinant to handle singular matrix
    if np.linalg.det(covariance) != 0:
        covariance_inv = np.linalg.inv(covariance)
    else: 
        covariance_inv = np.linalg.pinv(covariance)
        
    diff = embeddings - centerpoint
    return np.dot(np.dot(diff.T, covariance_inv), diff)
    
    
def evaluate_position(model, trainloader, validloader, testloader):
    model.eval()

    # extract training data features
    train_features = extract_features(model, trainloader)

    # extract validation data features
    valid_features = extract_features(model, validloader)
    distances = np.array([mahalanobis_distance(train_features, e) for e in valid_features])
    
    # extract test data features
    test_features = []
    labels = []

    with torch.no_grad():
        for test_data, label in testloader:
            f = model(test_data)[0].detach().numpy()
            test_features.append(f)
            labels.extend(label.numpy())

    test_features = np.concatenate(test_features, axis=0)
    labels = np.array(labels)
    test_distances = np.array([mahalanobis_distance(train_features, e) for e in test_features])

    # Calculate metrics
    percentile = 85
    cutoff = np.percentile(distances, percentile)
    y_pred = (test_distances > cutoff).astype(int)

    accuracy = np.sum(y_pred == labels) / len(labels)
    cm = confusion_matrix(labels, y_pred)
    AUROC = roc_auc_score(labels, test_distances)
    PR_AUC = average_precision_score(labels, test_distances)

    return AUROC, PR_AUC
    
    
    
    
    
    
