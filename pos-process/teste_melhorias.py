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

dataset_path = "student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1]
Y = dataset[:,-1]

print Y

# to 4 classes
# 0 - 4 (ruim - 0)
# 5 - 9 (regular - 1)
# 10 - 14 (bom - 2)
#  > 15 (ótimo - 3)
_Y = []
for i in Y:
    if :

knn = KNeighborsClassifier(n_neighbors=1)
accuracy_knn = cross_val_score(knn, X, Y, cv=10).mean()
print accuracy_knn
