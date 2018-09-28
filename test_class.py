
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import glob
import matplotlib.pyplot as plt
from sklearn import tree
import sys
from matplotlib.backends.backend_pdf import PdfPages

def print_array(l):
    s = ""
    for e in l:
        s+= str(e) + " "
    print s

dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1] # only attributes here
Y = dataset[:,-1] # class data

# Modify the class data from 21 classes to 4

# _Y = []
# for v in Y:
#     # class: 0
#     if v >= 0 and v <= 4:
#         _Y.append(0)
#     # class: 1
#     elif v >= 5 and v <= 9:
#         _Y.append(1)
#     # class: 2
#     elif v >= 10 and v <= 14:
#         _Y.append(2)
#     # class: 3
#     else:
#         _Y.append(3)
# Y = np.array(_Y)

knn = KNeighborsClassifier(n_neighbors=1)
ad = tree.DecisionTreeClassifier()
accuracy_knn = cross_val_score(knn, X, Y, cv=10).mean()
accuracy_ad = cross_val_score(ad, X, Y, cv=10).mean()

print "AD: ", accuracy_ad
print "k-NN", accuracy_knn
