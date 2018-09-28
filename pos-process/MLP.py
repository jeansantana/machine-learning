import pandas as pd
import numpy as np
from random import shuffle
from sklearn.cross_validation import cross_val_score
import glob
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neural_network import MLPClassifier

def print_dataset(dataset):
    for line in dataset:
        s = ""
        for e in line:
            s+= str(e) + ", "
        print s

def fill_lr():
    values = []
    values.append(0.1)
    for i in range(1, 9):
        values.append(values[i - 1] + .1)

    _values = []
    _values.append(0.01)
    for i in range(1, 9):
        _values.append(_values[i - 1] + .01)

    __values = []
    __values.append(0.001)
    for i in range(1, 9):
        __values.append(__values[i - 1] + .001)

    return values + _values + __values

def mlp(runs, _cv, l_rate, n_neurons, it):
    values = []
    for i in range(runs):
        _mlp = MLPClassifier(hidden_layer_sizes=(n_neurons), random_state = i, momentum=0.8, max_iter=it, learning_rate_init=l_rate)
        accuracy_mlp = cross_val_score(_mlp, X, Y, cv=_cv).mean()
        values.append(accuracy_mlp)
    return values

def best_res_mlp(runs, _cv, l_rate, n_neurons, it):
    values = mlp(runs, _cv, l_rate, n_neurons, it)
    best = values[0]
    for i in range(1, len(values)):
        if values[i] > best:
            best = values[i]
    return best

dataset_path = "student-reduced-53.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

X = dataset[:,:-1]
Y = dataset[:,-1]

learn_rate = [0.9, 0.05, 0.008]
# fill lean rate
# learn_rate = fill_lr()
# fill num neurons
neurons_num = [33, 50, 100]
# fill iterations
iterations = [100, 1000, 10000]

results = ""
best_conf = {}
best_acc = -1;
for l_rate in learn_rate:
    for n_neurons in  neurons_num:
        for it in iterations:
            # print "processing: ", l_rate, ", ", n_neurons, ", " , it
            results+= str(l_rate) + ", " + str(n_neurons) + ", " + str(it)
            # hidden_layer_sizes is a list where the ith element represents the number of neurons in the ith hidden layer.
            # mlp = MLPClassifier(hidden_layer_sizes=(n_neurons), random_state=1, momentum=0.8, max_iter=it, learning_rate_init=l_rate)
            accuracy_mlp = best_res_mlp(30, 2, l_rate, n_neurons, it)
            if accuracy_mlp > best_acc:
                best_conf['l_rate'] = l_rate
                best_conf['n_neurons'] = n_neurons
                best_conf['it'] = it
                best_acc = accuracy_mlp
            results+= ", " + str(accuracy_mlp) + "\n";

print results
print "\nmelhor configuracao de params\nl_rate, n_neurons, it, best_acc"

print best_conf['l_rate'], ",", best_conf['n_neurons'], ",", best_conf['it'], ",", best_acc
# 30 runs, cv = 10
values = mlp(30, 10, best_conf['l_rate'], best_conf['n_neurons'], best_conf['it'])
for i in range(len(values)):
    print str(i) + ", " + str(values[i])
print "melhor resultado:", np.amax(values)
print "Media,", np.average(values)
print "Desvio Padrao,", np.std(values)
