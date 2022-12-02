import shutil
import torch
import random
import pandas as pd
import sys
from pathlib import Path
import os
from sklearn import metrics
import numpy as np
import os
from glob import glob
import sklearn.metrics as skm
import pickle as pkl

this_dir = Path("/home/development/nihars/lokesh/cs337_grading/Quiz")

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    torch.manual_seed(seed)

def eval_q2():
  input = [0]*100 + [1]*  200
  random.shuffle(input)
  y_true = torch.tensor(input)
  random.shuffle(input)
  y_pred = torch.tensor(input)

  m=0
  c=""
  try:
    acc_pred = assgn.accuracy(y_pred,y_true) 
    if abs(acc_pred - skm.accuracy_score(y_true,y_pred) )<1e-3:
      m+=1
      c+=f"Accuracy implementation correct. 1 mark."
    else:
      c+= f"Accuracy implementation incorrect. expected:{ skm.accuracy_score(y_true,y_pred)} receied:{acc_pred}"
  except: 
    c+=f"Error running function accuracy"

  try:
    precision_pred = assgn.precision(y_pred,y_true) 
    if abs(precision_pred - skm.precision_score(y_true,y_pred) )<1e-3:
      m+=1
      c+=f"precision implementation correct. 1 mark."
    else:
      c+= f"precision implementation incorrect. expected:{ skm.precision_score(y_true,y_pred)} receied:{precision_pred}"
  except: 
    c+=f"Error running function precision"

  try:
    recall_pred = assgn.recall(y_pred,y_true) 
    if abs(recall_pred - skm.recall_score(y_true,y_pred) )<1e-3:
      m+=1
      c+=f"recall implementation correct. 1 mark."
    else:
      c+= f"recall implementation incorrect. expected:{ skm.recall_score(y_true,y_pred)} receied:{recall_pred}"
  except: 
    c+=f"Error running function recall"

  try:
    f1_pred = assgn.f1_score(y_pred,y_true) 
    if abs(f1_pred - skm.f1_score(y_true,y_pred) )<1e-3:
      m+=1
      c+=f"f1 implementation correct. 1 mark."
    else:
      c+= f"f1 implementation incorrect. expected:{ skm.f1_score(y_true,y_pred)} received:{f1_pred}"
  except: 
    c+=f"Error running function f1"

  y_true = torch.rand(300)
  y_pred = torch.rand(300)
  try:
    mae_pred = assgn.mean_absolute_error(y_pred,y_true) 
    if abs(mae_pred - skm.mean_absolute_error(y_true,y_pred) )<1e-3:
      m+=1
      c+=f"mean_absolute_error implementation correct. 1 mark."
    else:
      c+= f"mean_absolute_error implementation incorrect. expected:{ skm.mean_absolute_error(y_true,y_pred)} receied:{mae_pred}"
  except: 
    c+=f"Error running function mean_absolute_error"

  try:
    mse_pred = assgn.mean_squared_error(y_pred,y_true) 
    if abs(mse_pred - skm.mean_squared_error(y_true,y_pred) )<1e-3:
      m+=1
      c+=f"mean_squared_error implementation correct. 1 mark."
    else:
      c+= f"mean_squared_error implementation incorrect. expected:{ skm.mean_squared_error(y_true,y_pred)} receied:{mse_pred}"
  except: 
    c+=f"Error running function mean_squared_error"

  return m,c

def eval_q3():
    m, c = 2, ""
    X = torch.eye(5)
    y1 = torch.arange(5)
    y2 = torch.arange(5)

    X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = assgn.train_val_split(X=X, y1=y1, y2=y2, train_pc=0.6)

    if len(X_trn) != 3:
        c += f"Expected 3 records in Train but received {len(X_trn)} records"
        m = 0
    if len(X_val) != 2:
        c += f"Expected 2 records in val but received {len(X_val)} records"
        m = 0
    for v1, v2, v3 in zip(X_trn, y1_trn, y2_trn):
        if v2 != v3:
            c += "y1 and y2 is inconsistent"
            m = 0
        if torch.argmax(v1) != v2:
            c += "features and lebels are inconsistent"
            m = 0
    return m, c
        

if __name__ == "__main__":

    roll_number = sys.argv[1]
    name = sys.argv[2]

    marks = [0, 0, 0, 0]
    comments = ["", "", "", ""]
    set_seed()

    isImported = True
    
    try:
        import assignment as assgn
    except:
        isImported = False
        q_comments = ['Error importing functions']*3

    # question 1
    if isImported:
        try:
            with open("output.pkl", "rb") as file:
                stud_out = pkl.load(file)
            with open("dataset_test_ours.pkl", "rb") as file:
                our_preds = pkl.load(file)
            try:
                acc = skm.accuracy_score(stud_out[0].detach().cpu(), our_preds[1])
                marks[0] = acc
                comments[0] = f"Accuracy score for classification task: {acc}"
            except Exception as e:
                comments[0] += f"Exception occured: {str(e)}"
            
            try:
                mse = skm.mean_squared_error(stud_out[1].detach().cpu(), our_preds[2])
                marks[1] = mse
                comments[1] = f"mse score for regression task: {mse}"
            except Exception as e:
                comments[1] += f"Exception occured: {str(e)}"

        except Exception as e:
            comments[0] += f"Exception occured: {str(e)}"
        
        try:
            set_seed()
            m, c = eval_q2()
            marks[2] = m
            comments[2] += c
        except Exception as e:
            comments[2] += f"Exception occured: {str(e)}"
    
    try:
        m, c = eval_q3()
        marks[3] = m
        comments[3] += c
    except Exception as e:
        comments[3] += f"Exception occured: {str(e)}"
    

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(comments)
        f.write(f"{roll_number},{name},{marks_string}, {q_comments_string}"+"\n")
