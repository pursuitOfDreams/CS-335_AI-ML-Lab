from utils import store_points_to_json
from utils import load_points_from_json
import numpy as np
from sklearn.decomposition import PCA

X = load_points_from_json('points_4D.json')
pca = PCA(n_components=2)
y = pca.fit_transform(X)
store_points_to_json(y, "points_2D.json")
