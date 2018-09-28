import pandas as pd
import numpy as np
from Constants import *

dataset_path = "base_student/student-all-numeric-data.csv"

def getCorrCoef(X, attra, attrb):
    return np.corrcoef(X[:,attra], X[:,attrb])[0, 1]

def getCorrMatrix(X):
    M = []
    for i in range(0, X.shape[1]):
        aux=[]
        for j in range(0, X.shape[1]):
            aux.append(getCorrCoef(X, i, j))
        M.append(aux)
    return M
dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )
# print dataset
X = dataset[:,:-1]

CM = getCorrMatrix(X)
# CM = np.corrcoef(X.T)
np.savetxt("correlation_matrix.csv", CM, delimiter=",")

# The average correlation between each attribute
dic = {}
meanAttrsCorr = len(CM)*[0]
for i in range(1, len(CM)):
    for j in range(0, i):
        meanAttrsCorr[i]+= abs(CM[i][j]) / (len(CM) - 1)
        meanAttrsCorr[j]+= abs(CM[i][j]) / (len(CM) - 1)
        dic[Constants.names_numeric_header[i]] = meanAttrsCorr[i]
        dic[Constants.names_numeric_header[j]] = meanAttrsCorr[j]

data = pd.Series(dic)
data.sort_values(axis=0, ascending=False, inplace=True)
data.to_csv("average_correlation.csv", sep=",")

# data.sort_values(axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
# print meanAttrsCorr
# meanAttrsCorr.sort( reverse = True )

# np.savetxt("average_correlation.csv", meanAttrsCorr, delimiter="\n")

# for i in range(0, X.shape[1]):
#     s = ""
#     for j in range(0, X.shape[1]):
#         s+= str(CM[i][j]) + ", "
#     print s

# print "SIZE: ", X.shape[1]
# print X[:,0]
# print corrcoef
