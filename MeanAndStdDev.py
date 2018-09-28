import pandas as pd
import numpy as np
from Constants import *

dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )
# print dataset
X = dataset[:,:-1]
mean = np.mean(X, axis=0)
var = np.var(X, axis=0)
std = np.std(X, axis=0)


N_COLS = 2
mycsv = np.zeros( shape=( len(Constants.names_numeric_header), N_COLS ) )
# mycsv[0] = Constants.names_numeric_header
print "Atributo, Media, Variancia, Desvio Padrao"
for i in range( 0, len(Constants.names_numeric_header) ):
    print Constants.names_numeric_header[i] + ", " + str(mean[i]) + ", " + str(var[i]) + ", " + str(std[i]);
print ",,,"
print "Menor Variabilidade, , Maior Variabilidade, "
print "Atributo, Valor, Atributo, Valor"
idxMin = np.argmin(std)
idxMax = np.argmax(std)
print Constants.names_numeric_header[idxMin] + ", " + str(std[idxMin]) + ", " + Constants.names_numeric_header[idxMax] + ", " + str(std[idxMax]) 
# np.savetxt("mean_and_stddev.csv", mycsv, delimiter=',')

# for i in range(0, len(dataset_path)):
#     dataset_path[i]


# dataset = pd.read_csv(dataset_path, header=None, names=names_header)
# dataset['school'] = dataset['school'] == "TESTE"
# dataset.to_csv("/home/jeansilva/teste.csv", sep=",")
# print dataset
# print dataset['school']
# data = []
# for i in range(0, len(dataset.columns)):
#     s = ""
#     attr = str(dataset.columns[i])
#     for j in range(0, len(dataset[attr])):
#         # print dataset.columns[i]
#         s+= str(dataset[attr][j]) + ","
#     data.append(s)
# print data
