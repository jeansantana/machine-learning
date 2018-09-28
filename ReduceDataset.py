import pandas as pd
import numpy as np
from random import shuffle

dataset_path = "base_student/student-all-numeric-data.csv"

dataset = np.loadtxt( open(dataset_path, "rb"), delimiter="," )

set_sizes = []
for i in range(5, 100, 5):
    set_sizes.append( dataset.shape[0] - int( dataset.shape[0] * ( i / 100.0 ) ) )
datasets = []
indexes = list(range(dataset.shape[0]))

for i in set_sizes:
    shuffle(indexes)
    new_dataset = []
    for j in range(0, i):
        new_dataset.append( dataset[ indexes[j] ] )
    datasets.append(np.array(new_dataset))
datasets = [dataset] + datasets

for ds in datasets:
    name_file = "student-reduced-" + str(ds.shape[0]) + ".csv"
    print name_file, "was created sucessfully"
    np.savetxt("base_student/reduced_bases/by_instances/" + name_file, ds, delimiter=",")
