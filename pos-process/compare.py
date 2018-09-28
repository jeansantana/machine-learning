# from __future__ import division
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import glob
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
import sys
from sklearn.neural_network import MLPClassifier

dataset_path = "student-reduced-53.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1]
Y = dataset[:,-1]

knn = KNeighborsClassifier(n_neighbors=8)
cross = cross_val_score(knn, X, Y, cv=10)
accuracy_knn = cross.mean()
print "k-NN:"
print accuracy_knn
print np.std(cross)

_mlp = MLPClassifier(hidden_layer_sizes=(33), random_state = 25, momentum=0.8, max_iter=100, learning_rate_init=0.008)
_cross = cross_val_score(_mlp, X, Y, cv=10)
accuracy_mlp = _cross.mean()
print accuracy_mlp
print np.std(_cross)
