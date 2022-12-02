import torch
import random
import sys
from pathlib import Path
import os
from sklearn import metrics
import numpy as np
import os
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

def score(queries_embed:torch.Tensor, corpus_embed:torch.Tensor) -> torch.Tensor:
    """Computes the similarity score between each query and corpus pair.
    Since this function will be autograded, make sure that you stick to the formula mentioned.

    Args:
        queries_embed (torch.Tensor): This is a Tensor of shape (num_queries, embedding_dim)
        corpus_embed (torch.Tensor): This is a Tensor of shape (num_corpus, embedding_dim)

    Returns:
        torch.Tensor: This is a tensor of shape (num_queries, num_corpus)
    """
    scores = torch.zeros(queries_embed.shape[0], corpus_embed.shape[0])
    ### TODO
    scores = -torch.nn.ReLU()((queries_embed[:,None,:] - corpus_embed[None,:,:])).sum(-1)
    ### END TODO
    assert scores.shape == (queries_embed.shape[0], corpus_embed.shape[0])
    return scores

def ranking_loss(scores: torch.Tensor, ground_truth:torch.Tensor, margin:float) -> float:
    """Implements the ranking loss formula mentioned in the Question paper.

    Args:
        scores (torch.Tensor): This is Tensor of predicted similarity scores of shape (num_queries, num_corpus)
        ground_truth (torch.Tensor): This is Tensor of ground truth binary labels 
                            of shape (num_queries, num_corpus). Label 1 means the corresponding pair is relevant
        margin (float): This is a floating point value.

    Returns:
        torch.tensor: The computed Loss. This should be torch.tensor which allows autograd.
    """
    loss = 0

    ## TODO
    for qidx in range(scores.shape[0]):
      sc = scores[qidx]
      gt = ground_truth[qidx]
      predPos = sc[gt>0.5]
      predNeg = sc[gt<0.5]
      loss+= (torch.nn.ReLU()(margin+ predNeg[:,None] - predPos[None,:])).sum()

    ## End TODO

    assert len(loss.shape) == 0
    return loss

def average_precision(scores:torch.Tensor, ground_truth:torch.Tensor) -> float:
    """Implements the average precision formula mentioned in the Question paper.
    Since this function will be autograded, make sure that you stick to the formukla mentioned.

    Args:
        scores (torch.Tensor): This is Tensor of predicted similarity scores of shape (num_queries, num_corpus)
        ground_truth (torch.Tensor): This is Tensor of ground truth binary labels 
                            of shape (num_queries, num_corpus). Label 1 means the corresponding pair is relevant

    Returns:
        float: The computed evaluation metric \in [0,1].
    """
    avg_precision = float(0)
    ## TODO
    all_ap = []
    for qidx in range(scores.shape[0]):
      all_ap.append(skm.average_precision_score(ground_truth[qidx], scores[qidx]))
    avg_precision = sum(all_ap)/len(all_ap)
    ## END TODO
    assert isinstance(avg_precision, float) == True
    return avg_precision 

def eval_q1():
    margin=0.1
    scores = torch.rand(100,63)
    ground_truth = torch.randint(0,2,(100,63))

    m=0
    c=""
    try:
        rl_pred = assgn.ranking_loss(scores, ground_truth, margin) 
        if abs(rl_pred - ranking_loss(scores, ground_truth, margin) ) < 1:
            m+=3
            c+=f"ranking_loss implementation correct. 3 mark."
        elif abs(rl_pred - ranking_loss(scores, ground_truth, margin) ) < 1000:
            m+=2
            c+=f"ranking_loss implementation correct. 2 mark."
        else:
            c+= f"ranking_loss implementation incorrect. expected:{ ranking_loss(scores, ground_truth, margin)} receied:{rl_pred}"
    except Exception as e: 
        c+=f"Error running function ranking_loss  {str(e)}"
    return m,c


def eval_q2():
    queries_embed = torch.rand(100,15)
    corpus_embed = torch.rand(200,15)

    m=0
    c=""
    try:
        score_pred = assgn.score(queries_embed, corpus_embed) 
        if torch.allclose( score_pred, score(queries_embed, corpus_embed), atol=1e-3):
            m+=2
            c+=f"score implementation correct. 2 mark."
        else:
            c+= f"score implementation incorrect."
    except Exception as e: 
        c+=f"Error running function score  {str(e)}"

    return m,c


def eval_q3():
    scores = torch.rand(100,63)
    ground_truth = torch.randint(0,2,(100,63))


    m=0
    c=""
    try:
        map_pred = assgn.average_precision(scores, ground_truth) 
        if abs(map_pred -average_precision(scores, ground_truth) )<1e-2:
            m+=5
            c+=f"mean average precision function implementation correct. 5 mark."
        else:
            c+= f"mean average precision function implementation incorrect."
    except Exception as e: 
        c+=f"Error running function mean average precision function {str(e)}"

    return m,c


if __name__ == "__main__":

    roll_number = sys.argv[1]
    name = sys.argv[2]

    marks = [0, 0, 0, 0]
    comments = ["", "", "", ""]
    set_seed()

    isImported = True
    
    try:
        import assignment as assgn
    except Exception as e:
        isImported = False
        q_comments = ['Error importing functions']*4

    # question 1
    if isImported:
        try:
            with open("output.pkl", "rb") as file:
                stud_out = pkl.load(file)
                try:
                  stud_out = stud_out.detach()
                except Exception as e:
                  pass
            with open("test_truth.pkl", "rb") as file:
              our_preds = pkl.load(file)
              map = average_precision(scores=stud_out.detach(), ground_truth=our_preds)
              marks[3] = map
              comments[3] = f"Mean average precision for trained model: {map}"
        except Exception as e:
            comments[3] += f"except Exception as occured: {str(e)}"
            
        try:
            set_seed()
            m, c = eval_q1()
            marks[0] = m
            comments[0] += c
        except Exception as e:
            comments[0] += f"except Exception as eion occured: {str(e)}"
    
        try:
            m, c = eval_q2()
            marks[1] = m
            comments[1] += c
        except Exception as e:
            comments[1] += f"except Exception as eion occured: {str(e)}"

        try:
            m, c = eval_q3()
            marks[2] = m
            comments[2] += c
        except Exception as e:
            comments[2] += f"except Exception as eion occured: {str(e)}"
    

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(comments)
        f.write(f"{roll_number},{name},{marks_string}, {q_comments_string}"+"\n")
