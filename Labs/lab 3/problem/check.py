import numpy as np
import time
import assignment
import sys
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

TIME_LIMIT = 3
TLE_MULTIPLIER = [50, 10, 5]

def brute_dct(X: np.ndarray) -> np.ndarray:
    
    m, n = X.shape
    img_dct = np.empty((m,n))
    for k1 in range(m):
        for k2 in range(n):
            val = 0
            for n1 in range(m):
                for n2 in range(n):
                    val += 4*X[n1,n2]*np.cos((np.pi/m)*(n1+0.5)*k1)*np.cos((np.pi/n)*(n2+0.5)*k2)
            img_dct[k1, k2] = val

    return img_dct

def vectorized_dct(X: np.ndarray) -> np.ndarray:
    '''
    @params
        X : np.float64 array of size(m,n)
    return np.float64 array of size(m,n)
    '''
    # TODO
    m, n = X.shape
    
    C_left = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            C_left[i,j] = 2*np.cos((np.pi/m)*(i+0.5)*j)
    
    C_right = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C_right[i,j] = 2*np.cos((np.pi/n)*(i+0.5)*j)
    img_dct = np.matmul(np.matmul(C_left.T, X), C_right)
    return img_dct
    # END TODO

def get_document_ranking(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    @params
        D : n x w x k numpy float64 array 
            where each n x w slice represents a document
            with n vectors of length w 
        Q : m x w numpy float64 array
            which represents a query with m vectors of length w

    return np.ndarray of docIDs sorted in descending order according to relevance score
    '''
    # TODO
    D_rearranged = np.transpose(D,axes=[2,0,1]) 
    scores = np.sum(np.max(np.dot(D_rearranged, Q.T), 1), 1)
    
    return np.argsort(scores)[::-1]

    # END TODO

def compute_state_probability(M: np.ndarray,S_T: int, T: int, S_a: int) -> np.float64:
    '''
    @params
        M   : nxn numpy float64 array
        S_T : int
        T   : int
        S_a : int

    return np.float64
    '''
    # TODO
    N = M.shape[0]
    DP = np.zeros((N,T+1))
    DP[S_a][0] = 1.0

    #Fill up the DP table in a bottom up fashion starting from time t=0 tp time t=T
    for i in range(T):
        for j in range(N):
            for k in range(N):
                DP[j][i+1] += M[k][j] * DP[k][i]
    return DP[S_T][T]
    # END TODO


def compute_state_probability_vec(M: np.ndarray,S_T: int, T: int, S_a: int) -> np.float64:
    '''
    @params
        M   : nxn numpy float64 array
        S_T : int
        T   : int
        S_a : int

    return np.float64
    '''
    # TODO
    N = M.shape[0]
    DP = np.zeros((N,T+1))
    DP[S_a][0] = 1.0

    #Fill up the DP table in a bottom up fashion starting from time t=0 tp time t=T
    for i in range(T):
        X = DP[:,i]
        DP[:,i+1] = M.T@X
    
    # return np.dot(np.linalg.matrix_power(M.T,T), DP[:,0])[S_T]
    return DP[S_T][T]
    # END TODO

if __name__=="__main__":

    num_q = 3

    # Question 1
    roll_number = sys.argv[1]
    name = sys.argv[2]

    marks = [0]*num_q
    q_comments = ['']*num_q
    isImported = True
    
    try:
        import assignment
    except Exception as e:
        isImported = False
        q_comments = [f'\"Error importing function: \n\n {e}\"']*num_q
    # question 1
    if isImported:
        try:
            q_comments[0] += "\""
            for shape in zip([64, 128, 256],[64, 129, 255]):
                X = np.random.randn(*shape)
                tic = time.time()
                correct_vectorized_dct_X = vectorized_dct(X)
                correct_vectorized_dct_time = time.time() - tic

                tic = time.time()
                try:
                    with time_limit(TIME_LIMIT):
                        vectorized_dct_X = assignment.vectorized_dct(X)
                except TimeoutException:
                    vectorized_dct_X = X
                vectorized_dct_time = time.time() - tic
                if np.all(np.abs(vectorized_dct_X - correct_vectorized_dct_X) <= 5e-4):
                    marks[0] += 1/3
                    if vectorized_dct_time/correct_vectorized_dct_time < TLE_MULTIPLIER[0]:
                        marks[0] += 2/3
                    else:
                        q_comments[0] += f'Testcase with shape {shape} resulted in TLE with {vectorized_dct_time/correct_vectorized_dct_time/TLE_MULTIPLIER[0]} times the limit\n\n'
                else:
                    q_comments[0] += f'Testcase with shape {shape} failed\n\n'
            q_comments[0] += "\""
        except Exception as e:
            marks[0] = 0
            q_comments[0] = f'\"Following exception occurred:\n{e}\"'
        
        # question 2
        try:
            q_comments[1] += "\""
            for a,b,c,d in [(30,6,100,5),(40,20, 2000, 10)]:
                D = np.random.rand(a,b,c)
                Q = np.random.rand(d,b)

                tic = time.time()
                correct_doc_ranking = get_document_ranking(D,Q)
                vectorized_ranking_time = time.time() - tic

                tic = time.time()
                try:
                    with time_limit(TIME_LIMIT):
                        student_doc_ranking = assignment.get_document_ranking(D,Q)
                except TimeoutException:
                    student_doc_ranking = 0
                student_ranking_time = time.time() - tic
                if np.all(np.abs(np.array(student_doc_ranking) - np.array(correct_doc_ranking)) <= 5e-4):
                    marks[1] += 1
                    if (student_ranking_time/vectorized_ranking_time) < TLE_MULTIPLIER[1]:
                        marks[1] += 1
                    else:
                        q_comments[1] += f"Testcase with shapes {(a,b,c)}, {(d,b)} resulted in TLE with {student_ranking_time/vectorized_ranking_time/TLE_MULTIPLIER[1]} times the limit\n\n"
                else:
                    q_comments[1] += f"Testcase with shapes {(a,b,c)}, {(d,b)} failed\n\n"
            q_comments[1] += "\""
        except Exception as e:
            marks[1] = 0
            q_comments[1] = f'\"Following exception occurred:\n{e}\"'
        
        # question 3
        try:
            q_comments[2] += "\""

            N = 6
            S_T = 5
            S_a = 1
            M = np.zeros(shape=(N,N),dtype=np.float64)
            M[1][0] = 0.09
            M[0][1] = 0.23
            M[5][1] = 0.62
            M[1][2] = 0.06
            M[0][3] = 0.77
            M[2][3] = 0.63
            M[3][4] = 0.65
            M[5][4] = 0.38
            M[1][5] = 0.85
            M[2][5] = 0.37
            M[3][5] = 0.35
            M[4][5] = 1.0

            for T in [5000, 50000]:
                tic = time.time()
                correct_prob_vec = compute_state_probability_vec(M,S_T,T,S_a)
                correct_prob_vec_time = time.time() - tic

                tic = time.time()
                try:
                    with time_limit(TIME_LIMIT):
                        prob_non_vec = assignment.compute_state_probability(M,S_T,T,S_a)
                except TimeoutException:
                        prob_non_vec = 2
                prob_non_vec_time = time.time()-tic

                tic = time.time()
                try:
                    with time_limit(TIME_LIMIT):
                        prob_vec = assignment.compute_state_probability_vec(M,S_T,T,S_a)
                except TimeoutException:
                        prob_vec = 2
                prob_vec_time = time.time() - tic
                
                if np.abs(prob_non_vec - correct_prob_vec) < 5e-4:
                    marks[2] += 1/2
                else:
                    q_comments[2] += f"Testcase with T = {T} failed on compute_state_probability\n\n"
                
                if np.abs(prob_vec - correct_prob_vec) < 5e-4:
                    marks[2] += 1/3
                    if prob_vec_time/correct_prob_vec_time < TLE_MULTIPLIER[2]:
                        marks[2] += 2/3
                    else:
                        q_comments[2] += f"Testcase with T = {T} resulted in TLE on compute_state_probability_vec with {prob_vec_time/correct_prob_vec_time/TLE_MULTIPLIER[2]} times the limit\n\n"
                else:
                    q_comments[2] += f"Testcase with T = {T} failed on compute_state_probability_vec\n\n"
            q_comments[2] += "\""
        except Exception as e:
            marks[2] = 0
            q_comments[2] = f'\"Following exception occurred:\n{e}\"'
        

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(q_comments)
        f.write(f"{roll_number},{name},{marks_string},{sum(marks)},{q_comments_string}"+"\n")
