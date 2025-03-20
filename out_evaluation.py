import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from hypnettorch.mnets import MLP
from epsMLP import epsMLP
from hypnettorch.hnets import HMLP
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from torch import nn
from datetime import datetime
from itertools import product
from copy import deepcopy
from retry import retry
from datasets import set_hyperparameters, prepare_permuted_mnist_tasks, prepare_split_mnist_tasks
from torch.hub import load_state_dict_from_url
from torchattacks.attacks.fgsm import FGSM
from torchattacks import PGD, AutoAttack

init_epsilon = 0.01

def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle_file(filename):
    return torch.load(filename, map_location=torch.device(parameters["device"]))

def calculate_out_accuracy(data, target_network, weights, parameters, sample=10, epsilon=0.01):
    target_network = deepcopy(target_network)
    target_network.eval()

    input_data = data.get_test_inputs()
    input_data = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")

    batch_size, input_dim = input_data.shape
    input_expanded = input_data.unsqueeze(1).repeat(1, sample, 1)
    noise = (torch.rand_like(input_expanded) * 2 - 1) * epsilon
    X_noisy = input_expanded + noise

    X_noisy_flat = X_noisy.view(batch_size * sample, input_dim)

    with torch.no_grad():
        centers, epsilons = target_network.forward(X_noisy_flat, weights=weights)

    epsilons = epsilons.view_as(centers)

    lower_bounds = centers - epsilons
    upper_bounds = centers + epsilons

    is_inside = (X_noisy_flat >= lower_bounds) & (X_noisy_flat <= upper_bounds)
    is_inside = is_inside.all(dim=1)

    accuracy = is_inside.float().mean().item()

    outside_points = X_noisy_flat[~is_inside]
    outside_centers = centers[~is_inside]
    avg_distance = (outside_points - outside_centers).abs().mean().item() if outside_points.numel() > 0 else 0

    return accuracy, avg_distance


def main_running_experiments(path_to_datasets, parameters, hypernetwork_model, sample=10, use_chunks=False):
    if parameters["dataset"] == "PermutedMNIST":
        dataset_list_of_tasks = prepare_permuted_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"],
        )
    elif parameters["dataset"] == "SplitMNIST":
        dataset_list_of_tasks = prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
            number_of_tasks=parameters["number_of_tasks"],
        )

    target_network = epsMLP(
        n_in=parameters["input_shape"],
        n_out=list(dataset_list_of_tasks[0].get_train_outputs())[0].shape[0],
        hidden_layers=parameters["target_hidden_layers"],
        use_bias=parameters["use_bias"],
        no_weights=True,
        epsilon=init_epsilon,
    ).to(parameters["device"])
    if not use_chunks:
        hypernetwork = HMLP(
            target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=parameters["embedding_sizes"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hnet_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"],
        ).to(parameters["device"])
    else:
        hypernetwork = ChunkedHMLP(
            target_shapes=target_network.param_shapes,
            chunk_size=parameters["chunk_size"],
            chunk_emb_size=parameters["chunk_emb_size"],
            cond_in_size=parameters["embedding_sizes"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hnet_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"],
        ).to(parameters["device"])

    hnet_weights = load_pickle_file(hypernetwork_model)
    dataframe = pd.DataFrame(columns=["tested_task", "accuracy"])

    for task in range(parameters["number_of_tasks"]):

        currently_tested_task = dataset_list_of_tasks[task]

        accuracy, distance = calculate_out_accuracy(
            currently_tested_task,
            target_network,
            hypernetwork.forward(cond_id=task, weights=hnet_weights),
            parameters=parameters,
            sample=sample
        )
        result = {
            "tested_task": task,
            "accuracy": accuracy.cpu().item(),
            "distance": distance.cpu().item(),
        }
        print(f"Accuracy for task {task}: {accuracy}%.")

        dataframe = dataframe.append(result, ignore_index=True)

    dataframe = dataframe.astype({"tested_task": "int"})
    dataframe.to_csv(f'{parameters["saving_folder"]}/results_attacks.csv', sep=";")
    return dataframe

if __name__ == "__main__":
    path_to_datasets = "./Data"
    dataset = "SplitMNIST" # 'PermutedMNIST', 'SplitMNIST'
    summary_results_filename = "adversarial_results"
    hyperparameters = set_hyperparameters(dataset)

    parameters = {
        "embedding_sizes": hyperparameters["embedding_sizes"][0],
        "activation_function": hyperparameters["activation_function"],
        "hnet_hidden_layers": hyperparameters["hypernetworks_hidden_layers"][0],
        "input_shape": hyperparameters["shape"],
        "augmentation": hyperparameters["augmentation"],
        "number_of_tasks": hyperparameters["number_of_tasks"],
        "dataset": dataset,
        "seed": hyperparameters["seed"][0],
        "target_network": hyperparameters["target_network"],
        "target_hidden_layers": hyperparameters["target_hidden_layers"],
        "no_of_validation_samples_per_class": hyperparameters["no_of_validation_samples_per_class"],
        "padding": hyperparameters["padding"],
        "use_bias": hyperparameters["use_bias"],
        "device": hyperparameters["device"],
        "saving_folder": f'{hyperparameters["saving_folder"]}0/',
    }
    if "no_of_validation_samples" in hyperparameters:
        parameters["no_of_validation_samples"] = hyperparameters["no_of_validation_samples"]

    os.makedirs(parameters["saving_folder"], exist_ok=True)
    if parameters["seed"] is not None:
        set_seed(parameters["seed"])

    hnet001_path = 'Results/split_mnist_test/1903 001b001/hnet100.0.pt'

    main_running_experiments(path_to_datasets, parameters, hnet001_path, sample=10)