import pandas as pd
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import glob
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
from matplotlib.backends.backend_pdf import PdfPages

dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1] # only attributes here
Y = dataset[:,-1] # class data

# variance covered x accuracy
results = []
print X.shape[1]
for N in range(5, 101, 5):
    X_r = X
    # Define the reduction method
    var_cov = N/100.0
    n_attrs = int (var_cov * X.shape[1])
    if (len(sys.argv) < 2 ):
        sys.exit("Please, choose a reduction method: pca or lda option in execution")
    elif (sys.argv[1] == "pca"):
        reduction_method = PCA(n_components=n_attrs)
    elif (sys.argv[1] == "lda"):
        reduction_method = LinearDiscriminantAnalysis(n_components=n_attrs)
    # Reduce basis by a choose reduction method
    X_r = reduction_method.fit(X, Y).transform(X)
    # X_r = reduction_method.fit_transform(X, Y)
    print "Basis size: ", X_r.shape[1], "---", X.shape[1]
    # print "N, n_attrs: ", N, "--", X_r.shape[1], " VC = ", var_cov

    # cross-validation 10 folds
    knn = KNeighborsClassifier(n_neighbors=1)
    ad = tree.DecisionTreeClassifier()

    accuracy_knn = cross_val_score(knn, X_r, Y, cv=10).mean()
    accuracy_ad = cross_val_score(ad, X_r, Y, cv=10).mean()
    # chnage to:
    # results.append( [N/100.0, accuracy_knn, accuracy_ad] )
    results.append( [(1.0 * X_r.shape[1])/X.shape[1], accuracy_knn, accuracy_ad] )
# sort results by base size
# results.sort(key=lambda x: x[0])
# turn python list in numpy array
# print results

for res in results:
    print res

results = np.array(results)
# print results
# print "Best ", results[results.argmax(axis=0)[1]]
# plot
with PdfPages(sys.argv[0] + sys.argv[1] + '.pdf') as pdf:
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
