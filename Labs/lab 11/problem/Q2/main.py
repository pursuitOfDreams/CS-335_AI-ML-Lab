import numpy
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
    

    ###############    

    # return labels


if __name__ == "__main__":
    data = pd.read_csv("shared_samples.csv").values
    # labels = get_clusters(data)
    # store_labels(labels)
    # visualize(data, labels, alpha=0.2)
