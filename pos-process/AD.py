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

ad = tree.DecisionTreeClassifier()
accuracy_ad = cross_val_score(ad, X, Y, cv=10).mean()

print "accuracy_ad: %.2f " % (accuracy_ad)

# from sklearn import grid_search
# parameters = {'max_depth':range(3,20)}
# clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
# clf.fit(X=X, y=Y)
# tree_model = clf.best_estimator_
# print (clf.best_score_, clf.best_params_)
