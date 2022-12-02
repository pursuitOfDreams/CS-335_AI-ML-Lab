import numpy as np
import torch
import pickle as pkl
import torch.nn as nn

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
    y1_preds, y2_preds = None, None

    ## start TODO



    ## End TODO

    assert len(y1_preds.shape) == 1 and len(y2_preds.shape) == 1
    assert y1_preds.shape[0] == X_tst.shape[0] and y2_preds.shape[0] == X_tst.shape[0]
    assert len(torch.where(y1_preds == 0)[0]) + len(torch.where(y1_preds == 1)[0]) == X_tst.shape[0], "y1_preds should only contain classification targets"
    return y1_preds, y2_preds


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
    

    model = None
    ## start TODO
    # Your model definition, model training, validation etc goes here


    ## END TODO

    y1_preds, y2_preds = predict_labels(model, X_tst=X_tst)
    
    # You can test the metrics code -- accuracy, precision etc. using training data for correctness of your implementation

    # dump the outputs
    with open("output.pkl", "wb") as file:
        pkl.dump((y1_preds, y2_preds), file)
    with open("model.pkl", "wb") as file:
        pkl.dump(model, file)
