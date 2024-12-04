import torch
import torch.nn as nn
import torch.nn.functional as func
from Loss import SupConLoss
from tqdm import tqdm  

# Set up the NN layers to be versitile based on the size of the training data
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(13, 128), # Hidden layer with X, input and 128 output
            nn.ReLU(),  # activation function
            nn.Linear(128, 64),  # Hidden layer with 128 input, and 64 output
            nn.ReLU(),  # activation function
            )

    def forward(self, x):
        x = self.encoder(x)  # forward propigation
        return x
    

def train_dnn(FF_nn,temperature,epochs, optimizer, trainloader_SCL):
    FF_nn.train()
    criterion = SupConLoss(temperature)  # loss from https://arxiv.org/pdf/2004.11362
    iterative_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X_train, labels in trainloader_SCL:

            X_train = torch.cat([X_train[0], X_train[1]], dim=0).to('cpu')
            Y_shape = labels.shape[0]
            features = FF_nn.encoder(X_train)
            cluster_1, cluster_2 = torch.split(features, [Y_shape, Y_shape], dim=0)
            features_1 = torch.cat([cluster_1.unsqueeze(1), cluster_2.unsqueeze(1)], dim=1)
            features_1 = func.normalize(features_1, dim=2)
            loss = criterion(features_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()/X_train.shape[0]
        iterative_loss.append(epoch_loss)
 
    return FF_nn, iterative_loss
