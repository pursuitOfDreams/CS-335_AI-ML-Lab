import importlib
import torch
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import numpy 
import os 
import random
import sys
import numpy as np
import copy

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def RankingLoss(pred, label):
    arg_sorted = torch.argsort(label)
    pred = pred[arg_sorted]
    label = label[arg_sorted]
    idx = len(label) - torch.sum(label)
    pred_n = torch.reshape(pred[:idx], (1, -1))
    pred_p = torch.reshape(pred[idx:], (-1, 1))
    loss = torch.sum(torch.maximum(torch.zeros(pred_p.shape[0], pred_n.shape[1]), pred_n - pred_p))
    return loss



class MyModelClass(torch.nn.Module):
    def __init__(self, args, input_dim):
        super(MyModelClass, self).__init__()
        self.args  = args
        self.input_dm=input_dim
        self.lin = torch.nn.Linear(input_dim, 1)
    
    def forward(self, data):
        try:
            pred = self.lin(data)
            return pred 
        except: 
            pred = self.lin(data[0])
            return pred 
    
    def loss(self, pred, label):
        pred = pred.unsqueeze(-1)
        pred1 = torch.nn.Sigmoid()(copy.deepcopy(pred))
        label = label.unsqueeze(-1)
        if self.args.model_type == "ranking":

            l1 =  RankingLoss(pred, label)
            label = label.flatten()
            l2 =  RankingLoss(pred, label)
            label = label.unsqueeze(-1)
            pred = pred.flatten()
            l3 =  RankingLoss(pred, label)
            label = label.flatten()
            pred = pred.flatten()
            l4 =  RankingLoss(pred, label)
            return np.array([l1,l2,l3,l4])
        elif self.args.model_type == "nll":
            #return nn.BCELoss()(pred,label)
            l1 =  - torch.mean( label*torch.log(pred) + (1-label)*torch.log((1-pred)))
            l2 =  - torch.mean( label*torch.log(pred1) + (1-label)*torch.log((1-pred1)))
           
            l3 = -1*(torch.sum(torch.mul(label, torch.log(pred)) +
                       torch.mul(1-label, torch.log(1-pred))))/len(label)
            l4 = -1*(torch.sum(torch.mul(label, torch.log(pred1)) +
                       torch.mul(1-label, torch.log(1-pred1))))/len(label)
 
            label = label.flatten()
            
            l5 =  - torch.mean( label*torch.log(pred) + (1-label)*torch.log((1-pred)))
            l6 =  - torch.mean( label*torch.log(pred1) + (1-label)*torch.log((1-pred1)))
            l7 = -1*(torch.sum(torch.mul(label, torch.log(pred)) +
                       torch.mul(1-label, torch.log(1-pred))))/len(label)
            l8 = -1*(torch.sum(torch.mul(label, torch.log(pred1)) +
                       torch.mul(1-label, torch.log(1-pred1))))/len(label)
            
            label = label.unsqueeze(-1)
            pred = pred.flatten()
            l9 =  - torch.mean( label*torch.log(pred) + (1-label)*torch.log((1-pred)))
            l10 =  - torch.mean( label*torch.log(pred1) + (1-label)*torch.log((1-pred1)))
            l11 = -1*(torch.sum(torch.mul(label, torch.log(pred)) +
                       torch.mul(1-label, torch.log(1-pred))))/len(label)
            l12 = -1*(torch.sum(torch.mul(label, torch.log(pred1)) +
                       torch.mul(1-label, torch.log(1-pred1))))/len(label)

            label = label.flatten()
            pred = pred.flatten()
            l13 =  - torch.mean( label*torch.log(pred) + (1-label)*torch.log((1-pred)))
            l14 =  - torch.mean( label*torch.log(pred1) + (1-label)*torch.log((1-pred1)))
            l15 = -1*(torch.sum(torch.mul(label, torch.log(pred)) +
                       torch.mul(1-label, torch.log(1-pred))))/len(label)
            l16 = -1*(torch.sum(torch.mul(label, torch.log(pred1)) +
                       torch.mul(1-label, torch.log(1-pred1))))/len(label)
            return np.array([l13,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l1,l14,l15,l16])
        elif self.args.model_type == "svm":
            #TODO
            label[label==0] = -1
            loss = 1 - label*pred
            loss = torch.clamp(loss,min=0)
            l1 = torch.sum(loss)
            l2 = torch.mean(loss)
           
            label = label.flatten()
            #label[label==0] = -1
            loss = 1 - label*pred
            loss = torch.clamp(loss,min=0)
            l3 = torch.sum(loss)
            l4 = torch.mean(loss)
            
            label = label.unsqueeze(-1)
            pred = pred.flatten()
            #label[label==0] = -1
            loss = 1 - label*pred
            loss = torch.clamp(loss,min=0)
            l5 = torch.sum(loss)
            l6 = torch.mean(loss)
            
            label = label.flatten()
            pred = pred.flatten()
            loss = 1 - label*pred
            loss = torch.clamp(loss,min=0)
            l7 = torch.sum(loss)
            l8 = torch.mean(loss)
            
            L = 1 - (2*label-1)*pred
            L [L < 0]=0
            l9=torch.sum(L)
            l10=torch.mean(L)
            return np.array([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10])


def MyEvaluate(args,loader, model):
    model.eval()
    if args.model_type == "ranking":
        eval_score = 0
        data_size = 0
        N = []
        P = []
        for data in loader:
            with torch.no_grad():
                pred = model(data[0])
                label = data[1]
                N.append(pred[label == 0])
                P.append(pred[label == 1])
        N = torch.cat(N)
        P = torch.reshape(torch.cat(P), [1,-1])
        eval_score = ((P - N) > 0).sum()
    elif args.model_type == "nll":
        data = next(iter(loader))
        with torch.no_grad():
            pred = model(data[0])
            pred = torch.squeeze(pred)
            pred[pred>0] = 1
            pred[pred<=0] = 0
            eval_score = torch.mean((pred == data[1]).float())
    elif args.model_type == "svm":
        data = next(iter(loader))
        with torch.no_grad():
            label = data[1]
            label[label==0] = -1
            pred = model(data[0])
            pred = torch.squeeze(pred)
            pred = torch.sign(pred)
            eval_score = torch.mean((pred == label).float())
    return eval_score


def load_data():
    df= pd.read_csv('./dataset.csv') 
    df = df.drop(df.columns[0], axis=1)
    x = df[df.columns[0:80]].to_numpy()
    y = df["Output"].to_numpy()

    _, Xval, _, Yval = train_test_split(x, y, test_size=0.35, random_state=1)

    Xval = torch.from_numpy(Xval)
    Yval = torch.from_numpy(Yval)

    Xval = Xval.float()
    Yval = Yval.long()

    return [Xval, Yval]


def checker(pred, label, students_class, MyModel, MyAns, l1, eval_loader,args):
    feedback = ""
    Marks = 0
    #print(label)
    # pred = torch.tensor([0,0,0.1,0.2])
    # label = torch.tensor([1,0,1,0])
    isCorrect = False
    l2 = 90
    try:
        #print("1")
        #print(label)
        l2 = students_class.loss(pred, label)
        isCorrect = True
    except Exception as e:
        try:
            pred = pred.unsqueeze(1)
            ##print(label)
            #print("2")
            l2 = students_class.loss(pred, label)
            isCorrect = True
        except Exception as v:
            e = v
            try:
                label = label.unsqueeze(1)
                #print("3")
                l2 = students_class.loss(pred, label)
                isCorrect = True
            except Exception as b:
                e = b
                e = (str(e)).replace('\n', '\t').replace(',', ';')
                feedback += f"incorrect implementation of {args.model_type} loss {e}. "
    
    if isCorrect:
        #if abs(l1-l2)>1e-1:
        if all(abs((l1-l2.item()))>1e-1):
            feedback += "incorrect implementation of {:s} loss expected: {:.6f} received: {:.6f}. ".format(args.model_type,l1[0], l2)
        else:
            Marks += 10
            feedback += f"correct implementation of {args.model_type} loss: +10. "
        
    
    isCorrect = False
    try:
        stuAns = assignment.evaluate(eval_loader, MyModel)
        isCorrect = True
    except Exception as e:
        e = (str(e)).replace('\n', '\t').replace(',', ';')
        feedback += f"incorrect implementation of {args.model_type} evaluate {e}. "
    
    if isCorrect:
        if abs(MyAns-stuAns) > 1e-1:
            feedback += f"incorrect implementation of {args.model_type} evaluate expected:{MyAns} received:{stuAns}. "
        else:
            Marks += 10
            feedback += f"correct implementation of {args.model_type}  evaluate: +10. "
        
    #feedback+=str(Marks)
    return Marks, feedback



if __name__ == "__main__":
    roll_number = sys.argv[1]
    name = sys.argv[2]

    args = {'batch_size': 1280,
            'epochs': 100, 
            'opt': 'adam',
            'weight_decay': 5e-3,
            'lr': 0.01,
            'model_type': 'ranking'} 
    
    data_dim = 222 
    
    random.seed(999)
    
    args = objectview(args)
    
    Val = load_data()
    Xval = Val[0]
    Yval = Val[1]
    eval_dataset = data_utils.TensorDataset(Xval, Yval)
    
    pred = torch.tensor([random.uniform(0,1) for i in range(data_dim)])
    label = torch.tensor([random.randint(0,1) for i in range(data_dim)])
    
    #print(label) 
    
    total_feedback = ""
    total_Marks = 0
    
    try:
        import assignment
    
        students_class = assignment.ModelClass(args, data_dim)
        #for mt  in ["nll"]: 
        for mt  in ["ranking", "nll", "svm"]: 
            args.model_type =  mt
            eval_loader = data_utils.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
            MyModel = MyModelClass(args, 80)
            MyAns = MyEvaluate(args, eval_loader, MyModel)
            l1 = MyModel.loss(pred, label)
    
            m, f= checker(pred, label, students_class, MyModel, MyAns, l1, eval_loader, args)
            total_feedback += f
            total_Marks += m

    except Exception as e:
        e = (str(e)).replace('\n', '\t').replace(',', ';')
        total_feedback += f"Error in Model initialization {e}: 0"
        #print(feedback)
        #quit()


    with open('grades.csv','a') as f:
        #marks_string = ",".join([str(mark) for mark in marks])
        #q_comments_string = ",".join(q_comments)
        f.write(f"{roll_number},{name},{total_Marks},{total_feedback}\n")     
    
