import pandas as pd
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
from sklearn.cluster import KMeans
from daviesBouldinIndex import DaviesBouldinIndex
from matplotlib.backends.backend_pdf import PdfPages

def mean_db(runs, centroid, label, X):
    results = []
    for i in range(runs):
        db = DaviesBouldinIndex(centroid, label, X)
        results.append(db.getDBindex())
    return np.average(results)

dataset_path = "student-reduced-53.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1]
Y = dataset[:,-1]

# To teste
X = dataset

RUNS = 30
# results = []
x = []
y = []
# results.append(["k", "DB index"])
best = 100000000.0
best_label = []
cluster16 = []
for i in range(2, 41):
    k_means = KMeans(n_clusters=i, max_iter=500, random_state=10)
    k_means.fit(X)
    # if i == 2: print k_means.cluster_centers_
    m_db = mean_db(RUNS, k_means.cluster_centers_, k_means.labels_, X)
    # results.append( [i, db.getDBindex()] )
    if i == 16:
        cluster16 = k_means.labels_
    if m_db < best:
        best = m_db
        best_label = k_means.labels_
    x.append(i)
    y.append(m_db)

# print results
print "DB: ", best
s="["
for e in best_label:
    s+=str(e) + ", "
s+="]"
print s
print len(set(best_label))

print len(set(cluster16))
s="["
for e in cluster16:
    s+=str(e) + ", "
s+="]"
print s

with PdfPages("k_means" + '.pdf') as pdf:
    fig = plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.xlabel("k")
    plt.ylabel("DB index")
    ay = plt.axes()
    ay.set_ylim(0, 1.6)
    plt.plot(x, y, marker="o", linestyle='--', color='b', label="k-means")
    plt.legend()
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()
