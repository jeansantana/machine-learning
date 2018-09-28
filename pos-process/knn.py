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

def normalize(dataset):
    working = dataset.T
    ds = []
    for line in working:
        ln = []
        _max = np.amax(line)
        _min = np.amin(line)
        for v in line:
            value = (v - _min ) / (_max - _min)
            ln.append( value );
        ds.append(ln)

    ds = np.array(ds)
    return ds.T

dataset_path = "student-reduced-53.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1]
Y = dataset[:,-1]

if len(sys.argv) > 1 and sys.argv[1] == "1" :
    # dataset = preprocessing.normalize(dataset)
    X = normalize(X)

M_RG = X.shape[0]/2 # Max range, 50% of the insts number

results = []
for k in range(1, M_RG + 1):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    accuracy_knn = cross_val_score(knn, X, Y, cv=10).mean()
    print "k = %d, k-nn accuracy: %.2f" % (k, accuracy_knn)
    results.append([k, accuracy_knn]);

results = np.array(results)

import matplotlib.ticker as ticker
with PdfPages(sys.argv[0] + '.pdf') as pdf:
    fig = plt.figure()
    # axes = fig.add_subplot(1,1,1)
    x = results[:, 0]
    y = results[:, 1]
    # z = results[:, 2]
    plt.style.use('seaborn-whitegrid')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    # print np.arange(.0, 0.7, 0.05)
    # plt.yticks(np.arange(0.1, 0.7, 0.1))
    # plt.yticks(np.arange(0, 5, 0.05))
    # plot for KNN data

    ay = plt.axes()
    # ax.yaxis.set_major_locator(ticker.MultipleLocator())
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ay.set_ylim(0, 0.6)
    plt.plot(x, y, marker="o", linestyle='--', color='b')
    # plot for AD data
    # plt.plot(x, z, marker="o", linestyle='--', color='r', label='AD')
    # plt.legend()
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()
    plt.show()
