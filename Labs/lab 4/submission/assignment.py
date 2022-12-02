
import numpy as np 
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    NO EDITS REQUIRED IN THIS FUNCTION. 
"""
def train(args, Xtrain, Ytrain, Xval, Yval, model ):
    """
      tr_dataset : Num training samples * feature_dimension
      Trains for fixed number of epochs
      Keeps track of training loss and validation accuracy
    """
    tr_dataset = data_utils.TensorDataset(Xtrain, Ytrain)
    loader = data_utils.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataset = data_utils.TensorDataset(Xval, Yval)
    eval_loader = data_utils.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    # build model
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),\
                           lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    val_accs = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch[0])
            label = batch[1]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        losses.append(total_loss)
        val_acc = evaluate(eval_loader, model)
        val_accs.append(val_acc)
        print("Epoch ", epoch, "Loss: ", total_loss, "Val Acc.: ", val_acc)
    return val_accs, losses

"""
    NO EDITS REQUIRED HERE.
"""
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

"""
    NO EDITS REQUIRED IN THIS FUNCTION. 
"""
def set_seed(x=4):
    # Set random seeds
    seed = x
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

"""
    NO EDITS REQUIRED IN THIS FUNCTION. 
"""
def plot(val_accs, losses):
    """
        You can use this function to visualize progress of
        the training loss and validation accuracy  
    """
    plt.figure(figsize=(14,6))

    plt.subplot(1, 2, 1)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Val Accuracy", fontsize=18)
    plt.plot(val_accs)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(1, 2, 2)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Train Loss", fontsize=18)
    plt.plot(losses)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)



class ModelClass(torch.nn.Module):
    def __init__(self, args,input_dim):
        super(ModelClass, self).__init__()
        """
            Initialize the linear layer and any hyper-parameters here 
            You can add optional layers like relu as per you discretion 
            Any model parameter should be hard-coded here, do not add it to args 
        """
        self.args  = args
        self.input_dm=input_dim
        ######### TODO: Your code starts here ###############
        self.linear_svm = torch.nn.Linear(self.input_dm, 1)
        # self.linear
        self.sigmoid = torch.nn.Sigmoid()
        ######### Your code ends here  ################

    def forward(self, data):
      """
          Implement forard pass here
          Input: data is the batched data of shape (BATCH_SIZE * num_features)
          Output: Returns pred of shape BATCH_SIZE * 1 
          Note: pred will be autograd tensor
      """
      if self.args.model_type == "svm": 
        ##### TODO: Your code starts here######  
        pred = self.linear_svm(data)
        ######Your code ends here##########
      elif  self.args.model_type == "nll":
        ##### TODO: Your code starts here######  
        z = self.linear_svm(data)
        pred  = self.sigmoid(z)
        ######Your code ends here##########
      elif  self.args.model_type == "ranking":
        ##### TODO: our code starts here######  
        pred = self.linear_svm(data)
        ######Your code ends here##########
      else: 
        raise NotImplementedError()
      return pred

    def loss(self, pred, label):
      """
        Input : pred  : tensor of shape BATCH_SIZE*1
                label : tensor of shape BATCH_SIZE*1
        Output: returns single valued autograd tensor 
      """

      label = label.detach().clone()
      if self.args.model_type == "svm": 
        ##### TODO: Your code starts here#########
        label[label==0]=-1
        pred = pred.squeeze()
        l = torch.sum(torch.maximum(torch.zeros_like(label), 1-label*(pred)))/label.shape[0]
        ######Your code ends here##########
      elif  self.args.model_type == "nll":
        ##### TODO: Your code starts here###### 
        pred = pred.squeeze()
        l = torch.nn.BCELoss()(pred.type(torch.float64),label.type(torch.float64))
        ######Your code ends here##########
      elif  self.args.model_type == "ranking":
        ##### TODO: Your code starts here######  

        pred = pred[torch.argsort(label)]
        number_of_zeros = pred.shape[0]- torch.sum(label)
        number_of_zeros = int(number_of_zeros)
        l0 = pred[:number_of_zeros]
        l1 = pred[number_of_zeros:]
        p = torch.reshape(l1, (-1,1))
        n = torch.reshape(l0, (1,-1))
        l = torch.sum(torch.maximum(torch.zeros_like(n-p),n-p))
        ######Your code ends here##########
      else:
        raise NotImplementedError()
      
      return l

def evaluate(loader, model):
    """
        Testing module used for evaluation of validation data/ test data 
        Input:  loader : DataLoader object 
                model : model object
        Output: eval_score: the evaluation score for the model. 
                            Single valued torch tensor. 
                            "requires_grad=True" not necessary, since this is not used for backprop
        evaluation depends on model prediction and ground truth labels
        NOTE: In this function, you first need to use the model to predict the socres for each input (pred)
              Then as per the model_type, convert the (real values) pred to binary predictions
              Finally, use binary predictions, and binary labels to obtain evaluaiton score
    """
    model.eval() # This enables the evaluation mode for the model
    pList = []
    nList = []
    eval_score = 0
    total_len =0
    for data in loader:
      with torch.no_grad():
        #NOTE: pred = model(data)
        #NOTE: label = pred[1]
        pred = model(data[0])
        label = data[1].detach().clone()
        pred = pred.squeeze()

        if model.args.model_type == "svm": 
          ##### TODO: Your code starts here###### 
          label[label==0] =-1
          eval_score += torch.sum(pred*label>=0)
          total_len += pred.shape[0]
          ######Your code ends here##########
        elif model.args.model_type == "nll":
          ##### TODO: Your code starts here######  
          pred[pred<0.5]=0
          pred[pred>=0.5]=1

          eval_score += torch.sum((pred==label))
          total_len += pred.shape[0]
          ######Your code ends here##########
        elif  model.args.model_type == "ranking":
          ##### TODO: Your code starts here######  
          pred = pred.squeeze()
          pred = pred[torch.argsort(label)]
          number_of_zeros = pred.shape[0]- torch.sum(label)
          number_of_zeros = int(number_of_zeros)
          l0 = pred[:number_of_zeros]
          l1 = pred[number_of_zeros:]
          pList += l1.tolist()
          nList += l0.tolist()
          ######Your code ends here##########
        else: 
          raise NotImplementedError()

        #TODO (optional) If you want to add common code here ####
        ######Your code ends here##########

    
    if model.args.model_type!= "ranking":
      eval_score = eval_score/total_len
    else:
      p = torch.tensor(pList)
      n = torch.tensor(nList)
      p = torch.reshape(p, (-1,1))
      n = torch.reshape(n, (1,-1))
      eval_score = torch.sum((p-n)>0)

    return eval_score


if __name__ == '__main__':
    set_seed(4)
      
    df=pd.read_csv('./dataset.csv') ############ give path of the csv file and it will return a pandas dataframe
    ########### then you will have to split the dataframe into training and validation set where Xtrain denotes input samples and
    ########Ytrain denotes ouptut lables in training set and similar notation is used for validation set. 
    df = df.drop(df.columns[0], axis=1)
    x = df[df.columns[0:80]].to_numpy()
    y = df["Output"].to_numpy()

 
    Xtrain, Xval, Ytrain, Yval = train_test_split(x, y, test_size=0.3, random_state=42)


    Xtrain = torch.from_numpy(Xtrain)
    Ytrain = torch.from_numpy(Ytrain)

    Xval = torch.from_numpy(Xval)
    Yval = torch.from_numpy(Yval)

    Xtrain = Xtrain.float()
    Ytrain = Ytrain.long()
    Xval = Xval.float()
    Yval = Yval.long()
    args = {'batch_size': 128,
            'epochs': 100, 
            'opt': 'adam',
            'weight_decay': 5e-3,
            'lr': 0.01,
            'model_type': 'nll'} 

    args = objectview(args)

    input_dim = Xtrain.shape[1]
    args.model_type = 'ranking'  # Can change the model type here
    my_model = ModelClass(args, input_dim)

    val_accs, losses =  train(args, Xtrain, Ytrain, Xval, Yval, my_model)
    plot(val_accs, losses)
