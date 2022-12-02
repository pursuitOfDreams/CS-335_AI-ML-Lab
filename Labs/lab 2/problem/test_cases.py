import json
import numpy as np
import sys
import torch
import time

def test_lu_decomposition(A, L, U):
  comment, marks = "", 3
  lflag, rflag = False, False
  if np.allclose(A - np.matmul(L, U), np.zeros(A.shape)):
    for i, j in zip(np.arange(len(A)), np.arange(len(A))):
      if i < j and L[i, j] != 0:
        comment += f"L matrix is not proper"
        lflag = True
      if i > j and U[i, j] != 0:
        comment += f"R matrix is not proper"
        rflag = True
    marks -= (lflag + rflag)*0.5
  else:
    marks = 0
  comment += f"Total {marks} out of 3."
  return marks, comment  


def test_ques2_1(A):
    #PART 1
    if A is None:
        return 0, "A is none. 0 out of 0.5"
    if A.shape==torch.Size([50,40,5]) and torch.min(A)>=0 and torch.max(A)<=1:
        return 0.5, "0.5 out of 0.5"
    else:
        return 0, "shape is incorrect. 0 out of 0.5"

def test_ques2_2(B):
    if ~B.is_floating_point():
        return 0.5, "0.5 out of 0.5"
    else:
        return 0, "datatype is incorrect. 0 out of 0.5"

def test_ques2_3(C, D):
    #PART 3
    if C is None or D is None or C.shape!=torch.Size([3,100]):
        return 0, "Incorrect. 0 out of 1"
    perm = torch.LongTensor([2,0,1])
    D2 = C[perm,:]

    if torch.equal(D2,D):
        return 1, "1 out of 1."
    else:
        return 0, "Incorrect. 0 out of 1."

def test_ques2_4(E, F):
    #PART 4
    if E is None or F is None:
        return 0, "E or F is none. 0 out of 1."
    if torch.equal(F,torch.sum(E,1)) and E.shape==torch.Size([20,10]):
        return 1, "1 out of 1."
    else:
        return 0, "incorrect. 0 out of 1."

def test_ques2_5(H):
    #PART 5
    if H is None:
        return 0, "H is none. 0 out of 1"
    G1 = torch.zeros(10,10)
    G2 = torch.ones(10,10)
    G3 = torch.zeros(10,10)
    if torch.equal(H,torch.stack((G1,G2,G3),dim=2)):
        return 1, "1 out of 1."
    else:
        return 0, "Stacked tensor is incorrect. 0 out of 1."

def test_ques_3():
    total_marks = 0
    comment = ""
    for input_size in [100, 1000]:
        a = np.random.rand(input_size)
        b = np.random.rand(input_size)

        a_tensor = torch.FloatTensor(a)
        b_tensor = torch.FloatTensor(b)
        try: 
            s1 = time.time() 
            try:
                op1 = assignment.pairwise_ranking_loss_vec(a,b)
            except:
                op1 = assignment.pairwise_ranking_loss_vec(a_tensor, b_tensor)
                try:
                    op1 = op1.numpy()
                except:
                    pass
                
            t1 = time.time()-s1
            s2 = time.time() 
            try:
                op2 = assignment.pairwise_ranking_loss_looped(a,b)
            except:
                op2 = assignment.pairwise_ranking_loss_looped(a_tensor, b_tensor)
                try:
                    op2 = op2.numpy()
                except:
                    pass
            t2 = time.time()-s2
            if np.isscalar(op1): 
                total_marks += 0.34
                comment+= f"pairwise_ranking_loss_vec returns scalar output for input size : {input_size}. +0.34 Mark. "
            else:
                total_marks += 0
                comment+= f"pairwise_ranking_loss_vec does not returns scalar output for input size : {input_size}. +0 Mark. "
            if np.isscalar(op2): 
                total_marks += 0.33
                comment+= f"pairwise_ranking_loss_looped returns scalar output for input size : {input_size}. +0.33 Mark. "
            else:
                total_marks += 0
                comment+= f"pairwise_ranking_loss_looped does not returns scalar output for input size : {input_size}. +0 Mark. "
            if  np.isclose(op1,op2): 
                total_marks += 0.33
                comment+= f"pairwise_ranking_loss_looped and pairwise_ranking_loss_vec return same output for input size : {input_size}. +0.33 Mark. "
            else:
                total_marks += 0
                comment+= f"pairwise_ranking_loss_looped and pairwise_ranking_loss_vec do not return same output for input size : {input_size}. +0 Mark."
        except Exception as e:
            total_marks = 0
            comment += f"Following exception occured: {e}"
            return total_marks, comment

        #Now that assertions have passed, we do time checks
        total_t1 = 0
        total_t2 = 0
        for i in range(10):
            a = np.random.rand(input_size)
            b = np.random.rand(input_size)
            s1 = time.time() 
            op1 = assignment.pairwise_ranking_loss_vec(a,b)
            total_t1 += time.time()-s1
            s2 = time.time() 
            op2 = assignment.pairwise_ranking_loss_looped(a,b)
            total_t2 += time.time()-s2
        if (total_t2/total_t1 > input_size/10):
            total_marks += 0.5
            comment+= f"Speedup (Time) check passed for output for input size : {input_size}. +0.5 Mark. "
        else: 
            total_marks += 0
            comment+= f"Speedup (Time) check not passed for output for input size : {input_size}. +0 Mark. "
    comment += f"Total - {total_marks} out of 3"
    return total_marks, comment



if __name__ == "__main__":

    roll_number = sys.argv[1]
    name = sys.argv[2]

    marks = [0]*7
    q_comments = ['']*7
    isImported = True
    
    try:
        import assignment
    except:
        isImported = False
        q_comments = ['Error importing functions']*4

    # question 1
    if isImported:
        try:
            q_comments[0] += "\""
            A = np.random.rand(10, 10)
            L, U = assignment.LU_decomposition(A)
            m, c = test_lu_decomposition(A, L, U)
            q_comments[0] += c
            marks[0] = m
            q_comments[0] += "\""
        except Exception as e:
            marks[0] = 0
            q_comments[0] = f'\"Following exception occurred:{e}\"'
        

        try:
            q_comments[1] += "\""
            A = assignment.ques2_1()
            m, c = test_ques2_1(A)
            q_comments[1] += c
            marks[1] = m
            q_comments[1] += "\""
        except Exception as e:
            marks[1] = 0
            q_comments[1] = f'\"Following exception occurred:{e}\"'

        
        try:
            q_comments[2] += "\""
            B = assignment.ques2_2()
            m, c = test_ques2_2(B)
            q_comments[2] += c
            marks[2] = m
            q_comments[2] += "\""
        except Exception as e:
            marks[2] = 0
            q_comments[2] = f'\"Following exception occurred:{e}\"'
        
        try:
            q_comments[3] += "\""
            C,D = assignment.ques2_3()
            m, c = test_ques2_3(C, D)
            q_comments[3] += c
            marks[3] = m
            q_comments[3] += "\""
        except Exception as e:
            marks[3] = 0
            q_comments[3] = f'\"Following exception occurred:{e}\"'


        try:
            q_comments[4] += "\""
            E,F = assignment.ques2_4()
            m, c = test_ques2_4(E, F)
            q_comments[4] += c
            marks[4] = m
            q_comments[4] += "\""
        except Exception as e:
            marks[4] = 0
            q_comments[4] = f'\"Following exception occurred:{e}\"'


        try:
            q_comments[5] += "\""
            H = assignment.ques2_5()
            m, c = test_ques2_5(H)
            q_comments[5] += c
            marks[5] = m
            q_comments[5] += "\""
        except Exception as e:
            marks[5] = 0
            q_comments[5] = f'\"Following exception occurred:{e}\"'

        try:
            q_comments[6] += "\""
            m, c = test_ques_3()
            q_comments[6] += c
            marks[6] = m
            q_comments[6] += "\""
        except Exception as e:
            marks[6] = 0
            q_comments[6] = f'\"Following exception occurred:{e}\"'
        
        

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(q_comments)
        f.write(f"{roll_number},{name},{marks_string},{sum(marks)},{q_comments_string}"+"\n")
