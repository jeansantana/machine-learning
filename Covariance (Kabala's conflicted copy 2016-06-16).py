import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import tree
import sys
from matplotlib.backends.backend_pdf import PdfPages

dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1] # only attributes here
Y = dataset[:,-1] # class data

# correlation = np.corrcoef(X.T)
# covariance = np.cov(X.T)
# changing cov por corr
covariance = np.corrcoef(X.T)
# print covariance
# save matrix covariance in file
np.savetxt("covariance_matrix.csv", covariance, delimiter=",")

cov_average = {}
for i in range(0, covariance.shape[1]):
    cov_average[i] = 0

for i in range(1, covariance.shape[1]):
    for j in range(0, i):
        cov_average[i]+= abs(covariance[i][j]) / (covariance.shape[1] - 1)
        cov_average[j]+= abs(covariance[i][j]) / (covariance.shape[1] - 1)

# print cov_average
cov_average_sort = sorted(cov_average.items(), key=lambda x:x[1])
# # print cov_average_sort
# for tp in cov_average_sort:
#     print tp[0], " -- ", tp[1]
results = []
for i in range(5, 101, 5):

    n_attrs = int((i/100.00) * X.shape[1])

    X_r = [] # reduced basis
    for n in range(0, n_attrs):
        X_r.append(X[ :, cov_average_sort[n][0] ])

    X_r = np.array(X_r).T
    # print X_r
    # print X_r.shape[0], " X ", X_r.shape[1]
    knn = KNeighborsClassifier(n_neighbors=1)
    ad = tree.DecisionTreeClassifier()

    accuracy_knn = cross_val_score(knn, X_r, Y, cv=10).mean()
    accuracy_ad = cross_val_score(ad, X_r, Y, cv=10).mean()
    results.append( [i/100.0, accuracy_knn, accuracy_ad] )

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
