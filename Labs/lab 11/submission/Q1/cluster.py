from utils import store_labels_to_json
from utils import load_points_from_json
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

X = load_points_from_json("points_2D.json")
kmeans = KMeans(n_clusters = 5).fit((X[:,0]**2+X[:,1]**2).reshape(-1,1))
store_labels_to_json(5, kmeans.labels_, "labels.json")

# colors = ["red", "blue", "green", "yellow", "black"]

# plt.scatter(X[:,0],X[:,1], c = list(map(lambda x: colors[x], list(kmeans.labels_))))
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('cluster_2D.jpeg')
