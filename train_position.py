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
            nn.Linear(9, 128), # Hidden layer with X, input and 128 output
            nn.ReLU(),  # activation function
            nn.Linear(128, 64),  # Hidden layer with 128 input, and 64 output
            nn.ReLU(),  # activation function
            )
        self.projector = nn.Sequential(
            nn.Linear(64,9),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)  # forward propigation
        m_predict = self.projector(x)
        return x
    

def train_dnn_position(FF_nn,temperature,epochs, optimizer, trainloader_SCL):
    FF_nn.train()
    criterion = SupConLoss(temperature)  # loss from https://arxiv.org/pdf/2004.11362
    iterative_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X_train, labels, mask in trainloader_SCL:

            X_train = torch.cat([X_train[0], X_train[1]], dim=0).to('cpu')
            Y_shape = labels.shape[0]
            mask = torch.cat([mask[0], mask[1]], dim=0).to('cpu')
            features, m_predict = net(X_train)
            m_predict = m_predict.to(torch.float32)

            mask_criteria = nn.MSELoss(reduction='mean')
            mask_loss = mask_criteria(m_predict.cpu(), mask)

            features = net.encoder(X_train)
            features_11, features_21 = torch.split(features, [Y_shape, Y_shape], dim=0)
            features_1 = torch.cat([features_11.unsqueeze(1), features_21.unsqueeze(1)], dim=1)
            features_1 = func.normalize(features_1, dim=2)
            contrastive_loss = criterion(features_1)
            loss = 0.5 * mask_loss + contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()/X_train.shape[0]
        iterative_loss.append(epoch_loss)
 
    return FF_nn, iterative_loss
