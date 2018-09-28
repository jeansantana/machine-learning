import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import glob
import matplotlib.pyplot as plt
from sklearn import tree
import sys
from matplotlib.backends.backend_pdf import PdfPages

dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1] # only attributes here
Y = dataset[:,-1] # class data

attrs = range(0, X.shape[1])
results = []
for i in range(5, 101, 5):

    n_attrs = int((i/100.00) * X.shape[1])

    X_r = X
    # print n_attrs, " < ", X.shape[1]
    if (n_attrs < X.shape[1]):
        attrs = range(0, X.shape[1])
        attrs_chosen = random.sample(attrs, n_attrs)
        _X_r = []
        for n in attrs_chosen:
            _X_r.append(X[:,n])
        X_r = np.array(_X_r)
        X_r = np.transpose(_X_r)
    # print X_r.shape[0], "x", X_r.shape[1]
    # print X_r

    knn = KNeighborsClassifier(n_neighbors=1)
    ad = tree.DecisionTreeClassifier()

    accuracy_knn = cross_val_score(knn, X_r, Y, cv=10).mean()
    accuracy_ad = cross_val_score(ad, X_r, Y, cv=10).mean()
    print X_r.shape[1], " / ", X.shape[1], " = ", (1.0*X_r.shape[1])/X.shape[1], "-> ", i/100.0
    results.append( [(i/100.0), accuracy_knn, accuracy_ad] )

for res in results:
    print res
results = np.array(results)

# plot
with PdfPages(sys.argv[0] + '.pdf') as pdf:
    fig = plt.figure()
    x = results[:, 0]
    y = results[:, 1]
    z = results[:, 2]
    plt.style.use('seaborn-whitegrid')
    plt.xlabel("Variance Covered")
    plt.ylabel("Accuracy")
    # plot for KNN data
    plt.plot(x, y, marker="o", linestyle='--', color='b', label='KNN')
    # plot for AD data
    plt.plot(x, z, marker="o", linestyle='--', color='r', label='AD')
    # plt.legend(loc=3)
    plt.legend()
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()
    plt.show()
