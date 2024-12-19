import numpy as np

def average_accuracy(no_tasks, accuracy_table):
    ACC = np.sum([accuracy_table[t][no_tasks-1] for t in range(0, no_tasks)]) / no_tasks

    return ACC

def backward_transfer(no_tasks, accuracy_table):
    BWT =np.sum([accuracy_table[t][no_tasks-1] - accuracy_table[t][t] for t in range(0, no_tasks - 1)]) / (no_tasks - 1)

    return BWT