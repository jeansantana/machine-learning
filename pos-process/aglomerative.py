import pandas as pd
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
from sklearn.cluster import AgglomerativeClustering
from daviesBouldinIndex import DaviesBouldinIndex
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage
from scipy import cluster

def get_cluster_centroid(X, cluster):
    n_attrs = X.shape[1]
    _centroid = n_attrs * [0]
    for i in cluster:
        for j in range(0, n_attrs):
            _centroid[j]+= X[i][j]/(len(cluster) * 1.0)

    return _centroid

def get_cluster(id, labels):
    cluster = []
    for i in range( len(labels) ):
        if labels[i] == id:
            cluster.append(i)
    return cluster

def get_cluster_centroids(X, labels):
    _centroid = []
    clusters = set(labels)
    clusters = list(clusters)

    for i in range(len(clusters)):
        _cluster = get_cluster(clusters[i], labels)
        _centroid.append( get_cluster_centroid(X, _cluster) )

    return np.array(_centroid)

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
best = 1000000.0
best_label = []
cluster16  = []
# results.append(["k", "DB index"])
for i in range(2, 41):
    # k_means = KMeans(n_clusters=i, max_iter=500, random_state=10)
    # k_means.fit(X)
    agr = AgglomerativeClustering(n_clusters=i)
    agr.fit(X)
    centroid = get_cluster_centroids(X, agr.labels_)
    # print centroid
    m_db = mean_db(1, centroid, agr.labels_, X)
    if i == 16:
        cluster16 = agr.labels_
    if m_db < best:
        best = m_db
        best_label = agr.labels_
    # results.append( [i, db.getDBindex()] )
    x.append(i)
    y.append(m_db)

print "DB: ", best
s="["
for e in best_label:
    s+=str(e) + ", "
s+="]"
print s
print len(set(best_label))
print "best 16"
print (len(set(cluster16)))
s="["
for e in cluster16:
    s+=str(e) + ", "
s+="]"
print s
# print results

with PdfPages("AgglomerativeClustering" + '.pdf') as pdf:
    fig = plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.xlabel("k")
    plt.ylabel("DB index")
    ay = plt.axes()
    ay.set_ylim(0, 1.6)
    plt.plot(x, y, marker="o", linestyle='--', color='b', label="AgglomerativeClustering")
    plt.legend()
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()
