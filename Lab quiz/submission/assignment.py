import numpy as np
import torch
import pickle as pkl
import torch.nn as nn
import matplotlib.pyplot as plt 
torch.manual_seed(0)


def get_confusion_matrix(true_y_classes_array, predicted_y_classes_array):
  
    unique_classes = np.unique(true_y_classes_array)
    # For a binary class the above will give me [0 1] numpy array
    # so top-left of confusion matrix will start from 0 i.e. 'True Negative'

    # But the challenge here asks that the top left will be 'True Positive'
    # Hence I need to reverse the above numpy array
    unique_classes = unique_classes[::-1]
    # print('reversed unique', unique_classes) # will convert the above array to [1 0]

    # initialize a matrix with zero values that will be the final confusion matrix
    # For the binary class-label dataset, this confusion matrix will be a 2*2 square matrix
    confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)))

    for i in range(len(unique_classes)):
        for j in range(len(unique_classes)):
            confusion_matrix[i, j] = np.sum((true_y_classes_array == unique_classes[j]) & (predicted_y_classes_array == unique_classes[i]))

    return confusion_matrix

def train_val_split(X:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, train_pc):
    """This function splits the training dataset into train and validation datasets

    Args:
        X (_type_): The input torch 2D tensor
        y1 (_type_): classification target vector tensor
        y2 (_type_): regression target vector tensor
        train_pc (_type_): float \in (0, 1)
    
    Returns:
        X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val
    """

    X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = None, None, None, None, None, None

    ## Start TODO
    idxs = torch.randperm(X.shape[0])
    X_temp = X[idxs]
    y1_temp = y1[idxs]
    y2_temp = y2[idxs]

    train_len = int(train_pc*X.shape[0])
    X_trn = X_temp[:train_len]
    y1_trn = y1_temp[:train_len]
    y2_trn = y2_temp[:train_len]

    X_val = X_temp[train_len:]
    y1_val = y1_temp[train_len:]
    y2_val = y2_temp[train_len:]

    ## End TODO

    assert X_trn.shape[0] + X_val.shape[0] == X.shape[0]
    return  X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val

def accuracy(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the accuracy of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        acc: float between (0, 1)
    """
    acc = None

    ## Start TODO
    acc = torch.sum(preds==targets)/float(len(preds))
    ## End TODO

    return acc

def precision(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the precision of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        precision: float between (0, 1)
    """
    precision = None

    ## Start TODO
    p = preds.numpy()
    t = targets.numpy()
    C = get_confusion_matrix(t,p)
    TP = C[0][0]
    FP = C[0][1]
    TN = C[1][1]
    FN = C[1][0]
    precision = TP / (TP+FP)

    ## End TODO

    return precision

def recall(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the recall of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        recall: float between (0, 1)
    """
    recall = None

    ## Start TODO
    p = preds.numpy()
    t = targets.numpy()
    C = get_confusion_matrix(t,p)
    TP = C[0][0]
    FP = C[0][1]
    TN = C[1][1]
    FN = C[1][0]

    
    recall = TP/(TP+FN)
    ## End TODO

    return recall

def f1_score(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the F1-Score of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        f1_score: float between (0, 1)
    """
    f1 = None

    ## Start TODO
    p = preds.numpy()
    t = targets.numpy()
    C = get_confusion_matrix(t,p)
    TP = C[0][0]
    FP = C[0][1]
    TN = C[1][1]
    FN = C[1][0]

    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    f1 = (2 * (precision * recall)) / (precision + recall )

    ## End TODO

    return f1

def mean_squared_error(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the mse of the model predictions

    Args:
        preds (_type_): vector of regression predictions tensor
        targets (_type_): vector of ground truth regression targets tensor
    Returns:
        mse: float
    """
    mse = None

    ## Start TODO
    N = targets.shape[0]
    mse = torch.sum((targets-preds)**2)/N

    ## End TODO

    return mse

def mean_absolute_error(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the mae of the model predictions

    Args:
        preds (_type_): vector of regression predictions tensor
        targets (_type_): vector of ground truth regression targets tensor
    Returns:
        mae: float between
    """
    mae = None

    ## Start TODO
    N = targets.shape[0]
    mae = torch.sum(torch.abs(targets- preds))/N
    ## End TODO

    return mae


def predict_labels(model:nn.Module, X_tst:torch.Tensor):
    """This function makes the predictions for the multi-task model. 

    Args:
        model (nn.Module): trained torch model
        X_tst (torch.Tensor): test Tensor
    Returns:
        y1_preds: a tensor vector containing classificatiopn predictions
        y2_preds: a tensor vector containing regression predictions
    """
    y1_preds, y2_preds = torch.zeros(X_tst.shape[0]), torch.zeros(X_tst.shape[0])

    ## start TODO
    for i,x in enumerate(X_tst):
        with torch.no_grad():
            y1, y2 = model(x)
            y1_preds[i] = torch.argmax(y1)
            y2_preds[i] = y2.item()


    ## End TODO

    assert len(y1_preds.shape) == 1 and len(y2_preds.shape) == 1
    assert y1_preds.shape[0] == X_tst.shape[0] and y2_preds.shape[0] == X_tst.shape[0]
    assert len(torch.where(y1_preds == 0)[0]) + len(torch.where(y1_preds == 1)[0]) == X_tst.shape[0], "y1_preds should only contain classification targets"
    return y1_preds, y2_preds


class nn_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(nn_model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax()
        # self.

    def forward(self,x):
        y = self.linear1(x)
        y = self.relu1(y)
        y2 = self.linear2(y)
        y1 = self.linear3(y)
        y1 = self.relu2(y1)
        y1 = self.softmax(y1)

        return y1, y2


if __name__ == "__main__":

    # Load the dataset
    with open("dataset_train.pkl", "rb") as file:
        dataset_trn = pkl.load(file)
        X_trn, y1_trn, y2_trn = dataset_trn
        X_trn, y1_trn, y2_trn = torch.Tensor(X_trn), torch.Tensor(y1_trn), torch.Tensor(y2_trn)
    with open("dataset_test.pkl", "rb") as file:
        X_tst = pkl.load(file)
        X_tst = torch.Tensor(X_tst)
    
    X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = train_val_split(X=X_trn, y1=y1_trn, y2=y2_trn, train_pc=0.7)

    # print(X_trn)
    # exit(0)

    input_size = 28
    hidden_size = 56
    output_size = 1
    model = nn_model(input_size, output_size,hidden_size)
    ## start TODO
    # Your model definition, model training, validation etc goes here

    gamma = 0.5
    lr = 0.001
    epochs = 100
    loss1 = nn.BCELoss()
    loss2 = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    N = len(X_trn)
    accuracies = []
    f1_scores =[]
    mses =[]
    maes =[]

    for epoch in range(epochs):
        running_loss=0
        for i, (x, y1, y2) in enumerate(zip(X_trn, y1_trn, y2_trn)):
            y1_pred, y2_preds= model(x)
            # print(y1_pred)
            z = torch.zeros(2)
            z[int(y1)]=1.0
            optimizer.zero_grad()
            loss = loss1(y1_pred, z) + gamma*loss2(y2_preds,y2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch%30==0:
            optimizer.param_groups[0]['lr']*=0.5

        # validation step
        y1_preds, y2_preds = predict_labels(model, X_tst=X_val)
        acc = accuracy(y1_preds, y1_val)
        prec = precision(y1_preds, y1_val)
        rec = recall(y1_preds, y1_val)
        f1 = f1_score(y1_preds, y1_val) 
        mse = mean_squared_error(y2_preds, y2_val)
        mae = mean_absolute_error(y2_preds, y2_val)

        # UNCOMMENT THIS PART TO GET A RUNNING PLOT 

        # accuracies.append(acc)
        # f1_scores.append(f1)
        # mses.append(mse)
        # maes.append(mae)

        # plt.figure(figsize=(20,10))
        # plt.subplot(1,4,1)
        # plt.plot(range(epoch+1), accuracies)
        # plt.title("Accuracy vs epoch")
        # plt.xlabel("epochs")
        # plt.ylabel("Accuracy")

        # plt.subplot(1,4,2)
        # plt.plot(range(epoch+1), f1_scores)
        # plt.title("f1_score vs epoch")
        # plt.xlabel("epochs")
        # plt.ylabel("f1_score")

        # plt.subplot(1,4,3)
        # plt.plot(range(epoch+1), mses)
        # plt.title("MSE vs epoch")
        # plt.xlabel("epochs")
        # plt.ylabel("MSE")

        # plt.subplot(1,4,4)
        # plt.plot(range(epoch+1), maes)
        # plt.title("MAE vs epoch")
        # plt.xlabel("epochs")
        # plt.ylabel("MAE")

        # plt.savefig("plt.png")
        # plt.clf()
        
        # PRINTING THE PERFORMANCE METRICS
        print(40*"-"+f"EPOCH {epoch}"+40*"-")
        print(f"{torch.sum(y1_preds==y1_val).item()}/{len(y1_val)}")
        print(f"running loss = {running_loss/N}")
        print(f"accuracy = {acc}")
        print(f"f1 score = {f1}")
        print(f"MSE = {mse}")
        print(f"mae = {mae}")

        
    ## END TODO

    y1_preds, y2_preds = predict_labels(model, X_tst=X_tst)
    
    # You can test the metrics code -- accuracy, precision etc. using training data for correctness of your implementation

    # dump the outputs
    with open("output.pkl", "wb") as file:
        pkl.dump((y1_preds, y2_preds), file)
    with open("model.pkl", "wb") as file:
        pkl.dump(model, file)
