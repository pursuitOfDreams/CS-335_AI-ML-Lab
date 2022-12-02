import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import random
import pandas as pd
import sys

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    torch.manual_seed(seed)

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

if __name__ == "__main__":

    roll_number = sys.argv[1]
    name = sys.argv[2]

    marks = [-1, -1, 0, 1e3, 0]
    q_comments = ['']*5
    isImported = True
    
    try:
        import assignment
    except:
        isImported = False
        q_comments = ['Error importing functions']*5

    # question 1
    if isImported:

        set_seed()

        """# Logistic Regression Test cases"""
        from assignment import EarlyStoppingModule, LogisticRegression, LinearRegression

        moons_dataset = make_moons(n_samples=100, noise=0.05, random_state=100)
        moons_X_test, moons_y_test = moons_dataset

        def accuracy(yhat, y):
            yhat = yhat > 0.5
            return torch.sum(yhat == torch.Tensor(y)).item() / len(y)

        try:
            args = {'batch_size': 128,
                'max_epochs': 500, 
                'lr': 1e-3,
                'patience' : 10,
                'delta' : 1e-4,
                'temp' : 1.0,
                'model_type': 'logistic-moons'} 
            args = objectview(args)
            es_moons = assignment.EarlyStoppingModule(args)
            moons_model = es_moons.load_best_model()
            if moons_model is None:
                q_comments[0] += "Model file not uploaded with submission"
            moons_y_hat = moons_model.forward(torch.Tensor(moons_X_test))
            moons_acc = accuracy(moons_y_hat, moons_y_test)
            marks[0] = moons_acc
            q_comments[0] += f"Accuracy = {moons_acc} on pvt test dataset"
        except Exception as e:
            q_comments[0] += str(e).replace(",", " " )
        try:
            args = {'batch_size': 128,
                    'max_epochs': 500, 
                    'lr': 1e-3,
                    'patience' : 10,
                    'delta' : 1e-4,
                    'temp' : 1.0,
                    'model_type': 'logistic-titanic'}
            args = objectview(args)
            with open("/mnt/infonas/data/nlokesh/vcnet-base/varying-coefficient-net-with-functional-tr/Lab6-Grading/titanic_tst.pkl", "rb") as file:
                tit_X_test, tit_y_test = pickle.load(file)
                tit_X_test, tit_y_test = torch.Tensor(tit_X_test), torch.Tensor(tit_y_test)
            
            es_tit = assignment.EarlyStoppingModule(args)
            tit_model = es_tit.load_best_model()
            if tit_model is None:
                q_comments[1] += "Model file not uploaded with submission"
            tit_y_hat = tit_model.forward(torch.Tensor(tit_X_test))
            tit_acc = accuracy(tit_y_hat, tit_y_test)
            marks[1] = tit_acc
            q_comments[1] += f"Accuracy = {tit_acc} on pvt test dataset"
        except Exception as e:
            q_comments[1] += str(e).replace(",", " " )

        
        size = 6
        rnd_X, rnd_y = torch.eye(size), torch.arange(size)
        bsz = 2

        try:
            all_recvd = torch.zeros(size)
            comment = ""
            marks_2 = 2
            for idx, (mini_x, mini_y) in enumerate(assignment.minibatch(rnd_X, rnd_y, bsz)):
                if mini_x.shape[0] != bsz:
                    marks_2 -= 1
                    comment += "batch size is inconsistent."

                for x, y in zip(mini_x, mini_y):
                    all_recvd[y] = 1
                    if torch.where(x == 1)[0] == y:
                        pass
                    else:
                        comment += "X, y mismatch in minibatching"
                        marks_2 = 1
            if torch.sum(all_recvd) != size:
                comment += "minibatch does not pass over all samples in the dataset"
                marks_2 -= 1
            q_comments[2] += comment
            marks[2] = max(marks_2, 0)
            q_comments[2] += f"marks - {marks_2}/2"
        except Exception as e:
            q_comments[2] += str(e).replace(",", " " )

        """# Linear Regression"""

        def split_data(X:torch.Tensor, y:torch.Tensor, split_per=0.6):
            """Splits the dataset into train and test.
            """
            shuffled_idxs = torch.randperm(X.shape[0])
            trn_idxs = shuffled_idxs[0:int(split_per * X.shape[0])]
            tst_idxs = shuffled_idxs[int(split_per * X.shape[0]):]
            return X[trn_idxs], y[trn_idxs], X[tst_idxs], y[tst_idxs]

        def mse_loss(yhat, y):
            return torch.mean(torch.square(y-yhat))

        data = pd.read_csv('/mnt/infonas/data/nlokesh/vcnet-base/varying-coefficient-net-with-functional-tr/Lab6-Grading/dataset.csv', index_col=0)
        X = (data.iloc[:,:-1].to_numpy())
        y = (data.iloc[:,-1].to_numpy())
        X, y = torch.Tensor(X), torch.Tensor(y)

        X_train, y_train, X_test, y_test = split_data(X, y,)


        try:
            args = {'batch_size': 128,
                    'max_epochs': 500, 
                    'lr': 1e-3,
                    'patience' : 10,
                    'delta' : 1e-4,
                    'model_type': 'linear'} 
            args = objectview(args)
            es = assignment.EarlyStoppingModule(args)
            linear_model = es.load_best_model()
            if linear_model is None:
                q_comments[3] += "Model file not uploaded with submission"
            linear_yhat = linear_model.forward(X_test)
            lr_loss = mse_loss(linear_yhat,y_test)  #High values = good performance
            marks[3] = float(lr_loss.item())
            q_comments[3] += f"mse_loss = {lr_loss.item()}"
        except Exception as e:
            q_comments[3] += str(e).replace(",", " " )

        """##Early Stopping"""
        try:
            test_args = {'batch_size': 128,
                    'max_epochs': 500, 
                    'lr': 1e-3,
                    'patience' : 10,
                    'delta' : 1e-4,
                    'temp' : 1.0,
                    'model_type': 'DummyTest'} 

            marks_4 = 2
            test_args = objectview(test_args)
            es = assignment.EarlyStoppingModule(test_args)

            #Checking 1. if early stopping is keeping track of the correct best score
            #         2. should_stop_now variable is at the correct state

            val_scores = torch.cat([torch.arange(1,10,0.5),  torch.arange(1,10,0.5).flip(0)])
            epoch_idx = 0
            while not es.check(val_scores[epoch_idx],None,epoch_idx):
                epoch_idx+=1

            if not (es.best_score == val_scores[18] and es.should_stop_now):
                marks_4 -= 1
                q_comments[4] += "Failed to record the best validation score"


            #Checking if first score is the best score overall, then it should be stored correctly 
            val_scores = torch.arange(1,10,0.5).flip(0)
            epoch_idx = 0
            while not es.check(val_scores[epoch_idx],None,epoch_idx):
                epoch_idx+=1

            if not (es.best_score == val_scores[0] and es.should_stop_now):
                marks_4 -= 1
                q_comments[4] += "Failed to record the best validation score when the best score is in the first epoch"
            marks[4] = marks_4
            q_comments[4] += f"marks - {marks_4}/2"
        except Exception as e:
            q_comments[4] += str(e).replace(",", " " )

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(q_comments)
        f.write(f"{roll_number},{name},{marks_string}, {q_comments_string}"+"\n")
