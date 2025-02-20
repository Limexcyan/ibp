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

parameters = pd.dataframe('parameters_mask_sparsity_0.csv')

dataset_tasks_list = prepare_permuted_mnist_tasks(
    path_to_datasets,
    parameters["input_shape"],
    parameters["number_of_tasks"],
    parameters["padding"],
    parameters["no_of_validation_samples"],
)

dataset = "PermutedMNIST"