import matplotlib.pyplot as plt
import numpy as np

def kernel_pca(X: np.ndarray, kernel: str) -> np.ndarray:
    '''
    Returns projections of the the points along the top two PCA vectors in the high dimensional space.

        Parameters:
                X      : Dataset array of size (n,2)
                kernel : Kernel type. Can take values from ('poly', 'rbf', 'radial')

        Returns:
                X_pca : Projections of the the points along the top two PCA vectors in the high dimensional space of size (n,2)
    '''
    X_pca = None
    n, _ = X.shape
    d =5
    a = X
    b = X
    gamma = 15
    a1 = np.repeat(np.expand_dims(a,1),b.shape[0],axis=1)
    b1 = np.repeat(np.expand_dims(b,0),a.shape[0],axis=0)
    I = np.identity(n)
    In = np.ones((n,n))/float(n)
    K = None

    if kernel=='poly':
        print("here0")
        K = (1+X@X.T)**d
    elif kernel=='rbf':
        print("here1")
        K = np.exp(-gamma*(np.sum(((a1-b1)**2),axis=2)))
    elif kernel=='radial':
        print("here2")
        R = np.sqrt(np.sum(X**2, axis =1))
        theta = np.arctan2(X[:,1],X[:,0])
        R= np.reshape(R,(R.shape[0],1))
        theta = np.reshape(theta,(theta.shape[0],1))
        K = R@R.T+theta@theta.T
        # print(K)

    print("here")
    K_centered = np.dot(np.dot((I-In), K), (I-In))
    W, V = np.linalg.eigh(K_centered)
    X_pca = np.column_stack((V[:,-i] for i in range(1,3)))
    return X_pca

if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_circles
    from sklearn.linear_model import LogisticRegression
  
    X_c, y_c = make_circles(n_samples = 500, noise = 0.02, random_state = 517)
    X_m, y_m = make_moons(n_samples = 500, noise = 0.02, random_state = 517)

    X_c_pca = kernel_pca(X_c, 'radial')
    X_m_pca = kernel_pca(X_m, 'rbf')
    
    plt.figure()
    plt.title("Data")
    plt.subplot(1,2,1)
    plt.scatter(X_c[:, 0], X_c[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m[:, 0], X_m[:, 1], c = y_m)
    plt.savefig("1.png")

    plt.figure()
    plt.title("Kernel PCA")
    plt.subplot(1,2,1)
    plt.scatter(X_c_pca[:, 0], X_c_pca[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m_pca[:, 0], X_m_pca[:, 1], c = y_m)
    plt.savefig("2.png")
