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

def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle_file(filename):
    return torch.load(filename, map_location=torch.device(parameters["device"]))


def calculate_accuracy(data, target_network, weights, parameters, attack=None):
    target_network = deepcopy(target_network)
    target_network.eval()
    test_input = data.input_to_torch_tensor(data.get_test_inputs(), parameters["device"], mode="inference")
    test_output = data.output_to_torch_tensor(data.get_test_outputs(), parameters["device"], mode="inference")
    test_input.requires_grad = True
    gt_classes = test_output.max(dim=1)[1]

    if parameters["use_batch_norm_memory"]:
        logits, _ = target_network.forward(test_input, weights=weights, condition=parameters["number_of_task"])
    else:
        logits, _ = target_network.forward(test_input, weights=weights)

    predictions = logits.max(dim=1)

    target_network.mode = 'test'
    attack_instance = attack
    if attack is None:
        perturbed_pred = predictions
    else:
        #if attack == 'PGD':
        #    attack_instance = PGD(target_network, eps=0.0099, alpha=1 / 255, steps=1, random_start=False)
        if 1==1: #attack == 'FGSM':
            attack_instance = FGSM(target_network, eps=0.00999)
        #elif attack == 'AutoAttack':
        #    attack_instance = AutoAttack(target_network, norm='Linf', eps=8 / 255, version='standard', seed=None,
        #                                 verbose=False)

        adv_images = attack_instance(test_input, gt_classes)
        adv_logits = target_network.forward(adv_images, weights=weights)
        perturbed_pred = adv_logits.argmax(dim=1)

    perturbed_acc = 100 * (torch.sum(gt_classes == perturbed_pred).float() / gt_classes.numel())
    target_network.mode = None

    return perturbed_acc


def main_running_experiments(path_to_datasets, parameters, hypernetwork_model, epsilon=0.00, attack=None, use_chunks=False):
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
        epsilon=epsilon,
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

    hypernetwork.eval()
    target_network.eval()
    for task in range(parameters["number_of_tasks"]):
        hypernetwork = deepcopy(hypernetwork)
        currently_tested_task = dataset_list_of_tasks[task]
        hypernetwork_weights = hypernetwork.forward(cond_id=task, weights=hnet_weights)
        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            hypernetwork_weights,
            parameters={
                "device": parameters["device"],
                "use_batch_norm_memory": False,
                "number_of_task": parameters["number_of_tasks"],
            },
            attack=attack,
        )
        result = {
            "tested_task": task,
            "accuracy": accuracy.cpu().item(),
        }
        print(f"Accuracy for task {task}: {accuracy}%.")

        dataframe = dataframe.append(result, ignore_index=True)

    dataframe = dataframe.astype({"tested_task": "int"})
    dataframe.to_csv(f'{parameters["saving_folder"]}/results_attacks.csv', sep=";")
    return dataframe

def one_attack_multiinterval_plot(hnet_eps):
    test_string = f'{parameters["saving_folder"]}/results_attacks_{hnet_eps}.csv'

if __name__ == "__main__":
    path_to_datasets = "./Data"
    dataset = "SplitMNIST" # 'PermutedMNIST', 'SplitMNIST'
    summary_results_filename = "adversarial_results"
    hyperparameters = set_hyperparameters(dataset, grid_search=False, part=1)

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

    hnet000_path = 'Results/split_mnist_test/2502 000/hnet0.pt'
    # hnet001_path = 'Results/split_mnist_test/2502 001/hnet100.0.pt'
    # hnet005_path = 'Results/split_mnist_test/2202 005/hnet500.0.pt'
    main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0, attack=FGSM)
    main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.005, attack=FGSM)
    main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.009, attack=FGSM)
    main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.0099, attack=FGSM)
    main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.00999, attack=FGSM)
    main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.02, attack=FGSM)


    # main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.0499, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet000_path, epsilon=0.1, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet001_path, epsilon=0.0499, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet001_path, epsilon=0.1, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet005_path, epsilon=0.005, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet005_path, epsilon=0.00999, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet005_path, epsilon=0.02, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet005_path, epsilon=0.0499, attack=FGSM)
    # main_running_experiments(path_to_datasets, parameters, hnet005_path, epsilon=0.1, attack=FGSM)

