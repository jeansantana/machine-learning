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
from matplotlib.backends.backend_pdf import PdfPages

def print_dataset(dataset):
    for line in dataset:
        s = ""
        for e in line:
            s+= str(e) + ", "
        print s

dataset_path = "student-reduced-53.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1]
Y = dataset[:,-1]

_K_ = 8 # the best k for k-NN
# with distance weight
knn = KNeighborsClassifier(n_neighbors=_K_, weights='distance')
accuracy_knn = cross_val_score(knn, X, Y, cv=10).mean()
print "k-nn accuracy with distance weight (1/distance): %.2f" % (accuracy_knn)
# no distance weight ('weights = uniform' is udes by default)
knn = KNeighborsClassifier(n_neighbors=_K_)
accuracy_knn = cross_val_score(knn, X, Y, cv=10).mean()
print "k-nn accuracy with no distance weight (uniform by default): %.2f" % (accuracy_knn)
