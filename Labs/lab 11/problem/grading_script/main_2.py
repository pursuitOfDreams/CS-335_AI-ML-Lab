import numpy
from utils import store_labels
import pandas as pd
from main import get_clusters

data = pd.read_csv("mnist_samples.csv").values
labels = get_clusters(data)
store_labels(labels)