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

dataset_path = "student-reduced-53.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

def plot_graph(filename, y_label, x_label, x, y, label="MLP"):
    with PdfPages(filename + '.pdf') as pdf:
        fig = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.plot(x, y, marker="o", linestyle='--', color='b', label=label)
        plt.legend()
        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
        plt.close()

def fill_lr():

    _values = []
    _values.append(0.01)
    for i in range(1, 9):
        _values.append(_values[i - 1] + .01)

    __values = []
    __values.append(0.001)
    for i in range(1, 9):
        __values.append(__values[i - 1] + .001)

    return __values + _values

X = dataset[:,:-1]
Y = dataset[:,-1]

# x = []
# y = []
#
# lrs = fill_lr()
# # Variando a taxa de aprendizado
# for l_rate in lrs:
#     _mlp = MLPClassifier(hidden_layer_sizes=(33), random_state = 25, momentum=0.8, max_iter=100, learning_rate_init=l_rate)
#     accuracy_mlp = cross_val_score(_mlp, X, Y, cv=10).mean()
#     x.append(l_rate)
#     y.append(accuracy_mlp)
#
# plot_graph("taxa_aprendizado", "accuracy", "learn rate", x, y, "learn rate")
#
# Variando o numero de neuronios
# _x = []
# _y = []
# for n in range(2, 101):
#     _mlp = MLPClassifier(hidden_layer_sizes=(n), random_state = 25, momentum=0.8, max_iter=100, learning_rate_init=0.008)
#     accuracy_mlp = cross_val_score(_mlp, X, Y, cv=10).mean()
#     _x.append(n)
#     _y.append(accuracy_mlp)
#
# plot_graph("n_neurons", "accuracy", "# neurons", _x, _y, "# neurons")

# Variando o numero de iteracoes
__x = []
__y = []
for i in xrange(100, 10001, 100):
    print i
    _mlp = MLPClassifier(hidden_layer_sizes=(33), random_state = 25, momentum=0.8, max_iter=i, learning_rate_init=0.008)
    accuracy_mlp = cross_val_score(_mlp, X, Y, cv=10).mean()
    __x.append(i)
    __y.append(accuracy_mlp)

plot_graph("iteracoes", "accuracy", "# iterations", __x, __y, "# iterations")
