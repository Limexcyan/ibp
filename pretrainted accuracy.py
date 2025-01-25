from main import calculate_accuracy, evaluate_previous_tasks
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from hypnettorch.mnets import MLP
from hypnettorch.mnets.resnet import ResNet
from scipy.special.cython_special import eval_sh_legendre

from epsMLP import epsMLP
import attacks
import metrics

from ZenkeNet64 import ZenkeNet
from hypnettorch.hnets import HMLP
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
import hypnettorch.utils.hnet_regularizer as hreg
from torch import nn
from datetime import datetime
from itertools import product
#from torchpercentile import Percentile
from copy import deepcopy
from retry import retry

from datasets import (
    set_hyperparameters,
    prepare_split_cifar100_tasks,
    prepare_split_cifar100_tasks_aka_FeCAM,
    prepare_permuted_mnist_tasks,
    prepare_split_mnist_tasks,
    prepare_split_mnist_tasks,
    prepare_tinyimagenet_tasks,
)


dataframe = pd.DataFrame(
    columns=["after_learning_of_task", "tested_task", "accuracy"]
)

criterion = nn.CrossEntropyLoss()
path_to_datasets = "./Data"

dataset = "PermutedMNIST"

target_network = torch.load("Results/grid_search/permuted_mnist/0/target_network_after_9_task.pt")
hypernetwork = torch.load("Results/grid_search/permuted_mnist/0/hypernetwork_after_9_task.pt")
target_network.eval()
hypernetwork.eval()

accuracy = calculate_accuracy(
    data=data,
    target_network=target_network,
    weights=weights,
    parameters=parameters,
    evaluation_dataset="test"
)

print(f"Accuracy: {accuracy:.2f}%")


