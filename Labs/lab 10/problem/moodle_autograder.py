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

this_dir = Path("/mnt/infonas/data/nlokesh/vcnet-base/varying-coefficient-net-with-functional-tr/Lab-10")

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    torch.manual_seed(seed)

def evaluate_q1():
    mses = []
    def eval_file(labelFile):
        data = np.loadtxt(f"q1/{labelFile}.txt")    
        scores = np.loadtxt(f"{labelFile}.txt")
        mse = metrics.mean_squared_error(data, scores)
        return mse
    
    for out_file in ["output00", "output01", "output02"]:
        try:
            mse = eval_file(out_file)
            mses.append(mse)
        except:
            mses.append(-1)
    return mses

def evaluate_q2():
    accs = []
    def eval_file(labelFile):
        data = np.loadtxt(f"q2/{labelFile}.txt")    
        scores = np.loadtxt(f"{labelFile}.txt")
        acc = metrics.accuracy_score(data, scores)
        return acc

    for out_file in ["output00", "output01"]:
        try:
            acc = eval_file(out_file)
            accs.append(acc)
        except:
            accs.append(0)
        
    return accs

def walk_file(folder, file):
    root_directory = Path(folder)
    for path_object in root_directory.glob('**/*'):
        if path_object.is_file():
            if str(path_object.name) == file:
                return path_object
    return None
    

if __name__ == "__main__":

    roll_number = sys.argv[1]
    name = sys.argv[2]

    Q1 = eval(sys.argv[3])
    Q1_c = sys.argv[4]

    Q2 = eval(sys.argv[5])
    Q2_c = sys.argv[6]

    roll_dir = Path(sys.argv[7])

    marks = [-1, -1, -1, 0, 0]
    qcomments = [Q1_c, Q2_c]
    
    if Q1 == 1:
        try:
            os.system("rm *.txt")
            Q1_dir = (roll_dir / "Q1").absolute()
            out_files = [walk_file(Q1_dir, "output00.txt"), walk_file(Q1_dir, "output01.txt"), walk_file(Q1_dir, "output02.txt")]
            
            for file in out_files:
                if file is not None:
                    shutil.copy(file, Path(file).name)

        except Exception as e:
            Q1_c += str(e).replace(",", " " )
    else:
        pass

    mses = evaluate_q1()
    marks[0] = mses[0]; marks[1] = mses[1]; marks[2] = mses[2]
    qcomments[0] += f"AUC on pvt test cases: {str(mses)}".replace(",", " " )
    
    if Q2 == 1:
        try:
            os.system("rm *.txt")
            Q1_dir = (roll_dir / "Q2").absolute()
            out_files = [walk_file(Q1_dir, "output00.txt"), walk_file(Q1_dir, "output01.txt")]

            for file in out_files:
                if file is not None:
                    shutil.copy(file, Path(file).name)
        
        except Exception as e:
            Q2_c += str(e).replace(",", " " )
    else:
        pass
    
    accs = evaluate_q2()
    marks[3] = accs[0]; marks[4] = accs[1]
    qcomments[1] += f"AUC on pvt test cases: {str(accs)}".replace(",", " " )
    
    # Delete all the output files
    os.system("rm *.txt")

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(qcomments)
        f.write(f"{roll_number},{name},{marks_string}, {q_comments_string}"+"\n")
