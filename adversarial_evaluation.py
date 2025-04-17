import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
from hypnettorch.hnets import HMLP
from torchattacks.attacks.fgsm import FGSM
from torchattacks import PGD, AutoAttack

from IntervalNets.IntervalMLP import IntervalMLP
from datasets import set_hyperparameters, prepare_permuted_mnist_tasks, prepare_split_mnist_tasks

def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle_file(filename):
    return torch.load(filename, map_location=torch.device(parameters["device"]))

def calculate_accuracy(
        data, 
        target_network, 
        weights, 
        parameters, 
        evaluation_dataset,  
        epsilon_attack,
        perturbation_epsilon,
        attack=None):
    attack = str(attack)
    target_network = deepcopy(target_network)
    target_network.eval()
    # with torch.no_grad():
    if evaluation_dataset == "validation":
        input_data = data.get_val_inputs()
        output_data = data.get_val_outputs()
    elif evaluation_dataset == "test":
        input_data = data.get_test_inputs()
        output_data = data.get_test_outputs()
    elif evaluation_dataset == "train":
        input_data = data.get_train_inputs()
        output_data = data.get_train_outputs()

    data_input = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")
    data_output = data.output_to_torch_tensor(output_data, parameters["device"], mode="inference")

    data_input.requires_grad = True
    gt_classes = data_output.max(dim=1)[1]

    if parameters["use_batch_norm_memory"]:
        logits, _ = target_network.forward(
            data_input, 
            epsilon=perturbation_epsilon,
            weights=weights,
            condition=parameters["number_of_task"])
    else:
        logits, _ = target_network.forward(
            data_input, 
            epsilon=perturbation_epsilon,
            weights=weights)
    predictions = logits.max(dim=1)[1]
    attack_instance = 0
    if attack == 'None':
        perturbed_acc = 100 * (torch.sum(gt_classes == predictions).float() / gt_classes.numel())
        return perturbed_acc
    else:
        if attack == 'PGD':
            attack_instance = PGD(target_network, eps=epsilon_attack, alpha=1/255, steps=1, random_start=False)
        if attack == 'FGSM':
            attack_instance = FGSM(target_network, eps=epsilon_attack)
        elif attack == 'AutoAttack':
            attack_instance = AutoAttack(target_network, norm='Linf', eps=epsilon_attack, version='standard', seed=None, verbose=False)

    print('evaluation dataset: ', evaluation_dataset)
    print('attack: ', attack)
    print('atack epsilon', epsilon_attack)

    adv_images = attack_instance(data_input, gt_classes)

    if parameters["use_batch_norm_memory"]:
        adv_logits, _ = target_network.forward(adv_images, 
                                            epsilon=0.0,
                                            weights=weights,
                                            condition=parameters["number_of_task"])
    else:
        adv_logits, _ = target_network.forward(adv_images, 
                                            epsilon=0.0,
                                            weights=weights)

    perturbed_pred = adv_logits.argmax(dim=1)

    perturbed_acc = 100 * (torch.sum(gt_classes == perturbed_pred).float() / gt_classes.numel())
    return perturbed_acc


def main_running_experiments(
        path_to_datasets, 
        parameters, 
        hypernetwork_model, 
        epsilon_attack, 
        attack=None):
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
    
    # TODO Change this code to include new architectures, not only IntervalMLP
    target_network = IntervalMLP(
        n_in=parameters["input_shape"],
        n_out=list(dataset_list_of_tasks[0].get_train_outputs())[0].shape[0],
        hidden_layers=parameters["target_hidden_layers"],
        use_bias=parameters["use_bias"],
        no_weights=True,
        epsilon=parameters["perturbation_epsilon"],
    ).to(parameters["device"])

    hypernetwork = HMLP(
        target_network.param_shapes,
        uncond_in_size=0,
        cond_in_size=parameters["embedding_sizes"],
        activation_fn=parameters["activation_function"],
        layers=parameters["hnet_hidden_layers"],
        num_cond_embs=parameters["number_of_tasks"],
    ).to(parameters["device"])

    hnet_weights = load_pickle_file(hypernetwork_model)
    dataframe = pd.DataFrame(columns=["tested_task", "accuracy"])

    for task in range(parameters["number_of_tasks"]):

        currently_tested_task = dataset_list_of_tasks[task]

        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            hypernetwork.forward(cond_id=task, weights=hnet_weights),
            parameters=parameters,
            evaluation_dataset='test',
            attack=attack,
            epsilon=epsilon_attack
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
    
    # Please check attack strength
    epsilon_attack = 0.1

    main_running_experiments(
        path_to_datasets=path_to_datasets,
        parameters=parameters,
        hypernetwork_model=hnet001_path, 
        epsilon_attack=epsilon_attack,
        attack='FGSM')