from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np

data_x,data_y = np.load('data.npy')
data_Y = torch.from_numpy(data_y)
# Concatenate second order polynomial feature for x and store the result in data_X


def plotlosses(model,epoch, loss,name):
    plt.plot(data_x,data_y,'r.',label='Data')
    plt.plot(data_x,model(data_X).detach().numpy(),'b.',label='Prediction')
    plt.legend(loc='lower right')
    plt.title(f'EPOCH : {epoch} | LOSS : {loss}')
    plt.savefig(f'{name}_{epoch}.png')
    plt.close()

def train(X, Y, model, loss_fn, optim, max_iter, name=''):
    loss_list = []
    for epoch in range(max_iter):
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if epoch % 1000 == 0:
            plotlosses(model,epoch, loss.data.item(),name)
            loss_list += [loss.data.item()]
    return loss_list

# Set a manual seed of 100 for the torch module

class NLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(2,1).double())
        self.b1 = nn.Parameter(torch.randn(1).double())
        self.w2 = nn.Parameter(torch.randn(1,1).double())
        self.b2 = nn.Parameter(torch.randn(1).double())
    
    def forward(self, X):
        # Implement the evaluation function


nl_model = NLModel()

loss_function = nn.MSELoss()

nl_optim = torch.optim.SGD(nl_model.parameters(), lr=0.02)

loss_list = train(data_X,data_Y,nl_model,loss_function,nl_optim,10001,'non_linear')

# Save the the weights in lexicographic order AS PER THE PDF in 'weights.npy'

np.save('losses', loss_list)