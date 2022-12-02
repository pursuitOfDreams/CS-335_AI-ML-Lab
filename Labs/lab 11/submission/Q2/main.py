import numpy as np
import torch
from utils import visualize, store_labels
import pandas as pd

def get_clusters(X, n_clusters=10):
    '''
    Inputs:
        X: coordinates of the points
        n_clusters: (optional) number of clusters required
    Output:
        labels: The cluster index assigned to each point in X, same length as len(X)
    '''
    #### TODO: ####
    centroids = X[:n_clusters,:]
    X1= np.expand_dims(X,1)
    X1 = np.repeat(X1, n_clusters, axis = 1)
    while True:
        distances = np.sqrt(np.sum((X1 - centroids[None,:])**2, axis = 2))
        closest = np.argmin(distances,axis = 1)
        prev_centroids = centroids.copy()
        l = list(set(closest))
        centroids = np.array([np.average(X[np.where(closest==c)], axis=0) for c in l])
        if (np.linalg.norm(centroids-prev_centroids)) < 1e-9:
            break
    
    distances = np.sqrt(np.sum((X1 - centroids)**2, axis = 2))
    labels = np.argmin(distances,axis = 1)
    return labels


    ###############    

    # return labels


if __name__ == "__main__":
    data = pd.read_csv("mnist_samples.csv").values
    # labels = get_clusters(data)
    # store_labels(labels)
    # visualize(data, labels, alpha=0.2)
