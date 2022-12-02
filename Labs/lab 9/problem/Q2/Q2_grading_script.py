import numpy as np
import assignment
import sys
import matplotlib.pyplot as plt
import traceback


def same_dir(x, y):
    for i in range(2):
        if np.all(np.abs(x[:,i] - y[:,i]) < 5e-4) or np.all(np.abs(x[:,i] + y[:,i]) < 5e-4):
           continue
        return False
    return True 

def kernel_pca(X: np.ndarray, kernel: str) -> np.ndarray:
    '''
    Returns projections of the the points along the top two PCA vectors in the high dimensional space.

        Parameters:
                X      : Dataset array of size (n,2)
                kernel : Kernel type. Can take values from ('poly', 'rbf', 'radial')

        Returns:
                X_pca : Projections of the the points along the top two PCA vectors in the high dimensional space of size (n,2)
    '''
    
    if kernel == "radial":
        def cart2pol(x):
            rho = np.linalg.norm(x, axis=1, keepdims=True)
            phi = np.arctan2(x[:,1:2], x[:,0:1])

            return np.hstack([rho, phi])

        X_transformed = cart2pol(X)
        K = X_transformed @ X_transformed.T
        print(K)
    elif kernel == "rbf":
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)**2
        K = np.exp(-15*(norm_X + norm_X.T - 2*np.matmul(X,X.T)))
    elif kernel == "poly":
        K = (1 + X @ X.T) ** 5

    K = K - np.mean(K,axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K)
    # print("K value true",K)
    W, V = np.linalg.eigh(K)
    V = V[:,-1:-3:-1]
    # print(np.linalg.norm(K@V[:,0])/np.linalg.norm(V[:,0]) > np.linalg.norm(K@V[:,1])/np.linalg.norm(V[:,1]))
    return K @ V

if __name__=="__main__":

    num_q = 1

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
            from sklearn.datasets import make_moons, make_circles
            from sklearn.linear_model import LogisticRegression
  
            X_c, y_c = make_circles(n_samples = 3, noise = 0.02, random_state = 7)
            X_m, y_m = make_moons(n_samples = 3, noise = 0.02, random_state = 7)
            for X, kernel in [(X_c,'radial'), (X_m , 'rbf'), (X_c, 'poly')]:
                correct_answer = kernel_pca(X, kernel)
                student_answer = assignment.kernel_pca(X, kernel)
                print(correct_answer/np.linalg.norm(correct_answer, axis=0, keepdims=True) , student_answer/np.linalg.norm(student_answer, axis=0, keepdims=True), sep='\n')
                if same_dir(correct_answer/np.linalg.norm(correct_answer, axis=0, keepdims=True) , student_answer/np.linalg.norm(student_answer, axis=0, keepdims=True)):
                    marks[0] += 3
                else:
                    q_comments[0] += f'Testcase with kernel {kernel} failed\n\n'
            q_comments[0] += "\""
        except Exception as e:
            marks[0] = 0
            traceback.print_exc()
            q_comments[0] = f'\"Following exception occurred:\n{e}\"'
        
    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(q_comments)
        print(f"{roll_number},{name},{marks_string},{sum(marks)},{q_comments_string}"+"\n")
