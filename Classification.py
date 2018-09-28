# from __future__ import division
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

# Use v = list(range(N)) and shuffle(v) to do this
dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

# load reduced datasets
datasets = []
ls = glob.glob("base_student/reduced_bases/by_instances/*.csv")
print ls
for f in ls:
    datasets.append(np.loadtxt( open(f, "rb"), delimiter="," ))

# KNN Clasifier
# base size x accuracy
results = []
for ds in datasets:
    # cross-validation 10 folds
    knn = KNeighborsClassifier(n_neighbors=1)
    ad = tree.DecisionTreeClassifier()
    X = ds[:,:-1]
    Y = ds[:,-1]
    accuracy_knn = cross_val_score(knn, X, Y, cv=10).mean()
    accuracy_ad = cross_val_score(ad, X, Y, cv=10).mean()
    results.append( [ds.shape[0], accuracy_knn, accuracy_ad] )
# sort results by base size
results.sort(key=lambda x: x[0])
# turn python list in numpy array
results = np.array(results)
# print results
# print "Best ", results[results.argmax(axis=0)[1]]
# plot
with PdfPages(sys.argv[0] + '.pdf') as pdf:
    fig = plt.figure()
    x = results[:, 0]
    y = results[:, 1]
    z = results[:, 2]
    plt.style.use('seaborn-whitegrid')
    plt.xlabel("Base Size")
    plt.ylabel("Accuracy")
    # plot for KNN data
    plt.plot(x, y, marker="o", linestyle='--', color='b', label='KNN')
    # plot for AD data
    plt.plot(x, z, marker="o", linestyle='--', color='r', label='AD')
    plt.legend()
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()
    plt.show()
