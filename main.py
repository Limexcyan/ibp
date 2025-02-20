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
from torchattacks import PGD, FGSM, AutoAttack

def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_row_to_file(filename, elements):
    if not filename.endswith(".csv"):
        filename += ".csv"
    filename = filename.replace(".pt", "")
    with open(filename, "a+") as stream:
        np.savetxt(stream, np.array(elements)[np.newaxis], delimiter=";", fmt="%s")


def write_pickle_file(filename, object_to_save):
    torch.save(object_to_save, f"{filename}.pt")


@retry((OSError, IOError))
def load_pickle_file(filename):
    return torch.load(filename)


def get_shapes_of_network(model):
    shapes_of_model = []
    for layer in model.weights:
        shapes_of_model.append(list(layer.shape))
    return shapes_of_model


def calculate_number_of_iterations(number_of_samples, batch_size, number_of_epochs):
    no_of_iterations_per_epoch = int(np.ceil(number_of_samples / batch_size))
    total_no_of_iterations = int(no_of_iterations_per_epoch * number_of_epochs)
    return no_of_iterations_per_epoch, total_no_of_iterations


def calculate_accuracy(data, target_network, weights, parameters, evaluation_dataset):
    target_network = deepcopy(target_network)
    target_network.eval()
    # with torch.no_grad():
    if 1:
        if evaluation_dataset == "validation":
            input_data = data.get_val_inputs()
            output_data = data.get_val_outputs()
        elif evaluation_dataset == "test":
            input_data = data.get_test_inputs()
            output_data = data.get_test_outputs()

        # TODO assert na nieujemnosc danych (input_data)

        test_input = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")
        test_output = data.output_to_torch_tensor(output_data, parameters["device"], mode="inference")
        test_input.requires_grad = True
        gt_classes = test_output.max(dim=1)[1]
        if parameters["use_batch_norm_memory"]:
            logits = target_network.forward(test_input, weights=weights, condition=parameters["number_of_task"])
        else:
            logits = target_network.forward(test_input, weights=weights)
        if len(logits) == 2:
            logits, _ = logits
        predictions = logits.max(dim=1)[1]
        attack = None
        if evaluation_dataset == "test" and attack is not None:
            target_network.mode = 'test'
            if attack == 'PGD':
                attack = PGD(target_network, eps=1/255, alpha=1/255, steps=10, random_start=False)
            elif attack == 'FGSM':
                attack = FGSM(target_network, eps=8/255)
            elif attack == 'AutoAttack':
                attack = AutoAttack(target_network, norm='Linf', eps=8/255, version='standard', seed=None, verbose=False)
            adv_images = attack(test_input, gt_classes)
            adv_logits = target_network.forward(adv_images, weights=weights)
            perturbed_pred = adv_logits.max(dim=1)[1]
            perturbed_acc = 100 * (torch.sum(gt_classes == perturbed_pred).float() / gt_classes.numel())
            target_network.mode = None
            return perturbed_acc
        accuracy = torch.sum(gt_classes == predictions).float() / gt_classes.numel() * 100.0
        return accuracy


def evaluate_previous_tasks(hypernetwork, target_network, dataframe_results, list_of_permutations, parameters):
    hypernetwork.eval()
    target_network.eval()
    for task in range(parameters["number_of_task"] + 1):
        currently_tested_task = list_of_permutations[task]
        hypernetwork_weights = hypernetwork.forward(cond_id=task)
        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            hypernetwork_weights,
            parameters=parameters,
            evaluation_dataset="test",
        )
        result = {
            "after_learning_of_task": parameters["number_of_task"],
            "tested_task": task,
            "accuracy": accuracy.cpu().item(),
        }
        print(f"Accuracy for task {task}: {accuracy}%.")
        dataframe_results = dataframe_results.append(result, ignore_index=True)
    return dataframe_results


def save_parameters(saving_folder, parameters, name=None):
    if name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"parameters_{current_time}.csv"
    with open(f"{saving_folder}/{name}", "w") as file:
        for key in parameters.keys():
            file.write(f"{key};{parameters[key]}\n")


def plot_heatmap(load_path):
    dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
    dataframe = dataframe.astype({"after_learning_of_task": "int32", "tested_task": "int32"})
    table = dataframe.pivot("after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.tight_layout()
    plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
    plt.close()


def train_single_task(hypernetwork, target_network, criterion, parameters, dataset_list_of_tasks, current_no_of_task):
    if parameters["optimizer"] == "adam":
        optimizer = torch.optim.Adam([*hypernetwork.parameters()],lr=parameters["learning_rate"])
    elif parameters["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop([*hypernetwork.parameters()], lr=parameters["learning_rate"])
    if parameters["best_model_selection_method"] == "val_loss":
        best_hypernetwork = deepcopy(hypernetwork)
        best_target_network = deepcopy(target_network)
        best_val_accuracy = 0.0
    hypernetwork.train()
    target_network.train()
    print(f"task: {current_no_of_task}")
    use_batch_norm_memory = False
    current_dataset_instance = dataset_list_of_tasks[current_no_of_task]
    if parameters["number_of_epochs"] is not None:
        (
            no_of_iterations_per_epoch,
            parameters["number_of_iterations"],
        ) = calculate_number_of_iterations(
            current_dataset_instance.num_train_samples,
            parameters["batch_size"],
            parameters["number_of_epochs"],
        )
        if parameters["lr_scheduler"]:
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "max",
                factor=np.sqrt(0.1),
                patience=5,
                min_lr=0.5e-6,
                cooldown=0,
                verbose=True,
            )
    for iteration in range(parameters["number_of_iterations"]):
        current_batch = current_dataset_instance.next_train_batch(
            parameters["batch_size"]
        )
        tensor_input = current_dataset_instance.input_to_torch_tensor(
            current_batch[0], parameters["device"], mode="train"
        )
        tensor_output = current_dataset_instance.output_to_torch_tensor(
            current_batch[1], parameters["device"], mode="train"
        )
        gt_output = tensor_output.max(dim=1)[1]

        optimizer.zero_grad()
        hnet_weights = hypernetwork.forward(cond_id=current_no_of_task)

        if parameters["target_network"] == "epsMLP":
            default_eps = target_network.epsilon
            total_iterations = parameters["number_of_iterations"]
            inv_total_iterations = 1 / total_iterations
            target_network.epsilon = iteration * inv_total_iterations * default_eps
            # FIXME hnet czy target weights?
            prediction, eps_prediction = target_network.forward(tensor_input, weights=hnet_weights)

            z_lower = prediction - eps_prediction.T
            z_upper = prediction + eps_prediction.T
            z = torch.where((nn.functional.one_hot(gt_output, prediction.size(-1))).bool(), z_lower, z_upper)

            loss_spec = criterion(z, gt_output)
            loss_fit = criterion(prediction, gt_output)
            kappa = 1 - (iteration * inv_total_iterations * 0.5)

            loss_current_task = kappa * loss_fit + (1 - kappa) * loss_spec
            target_network.epsilon = default_eps
        else:
            # FIXME hnet czy target weights?
            prediction = target_network.forward(tensor_input, weights=hnet_weights)
            loss_current_task = criterion(prediction, gt_output)
            print(f' loss: {loss_current_task.item()}')
        loss_current_task.backward()
        optimizer.step()
        if parameters["number_of_epochs"] is None:
            condition = (iteration % 100 == 0) or (
                iteration == (parameters["number_of_iterations"] - 1)
            )
        else:
            condition = (
                (iteration % 100 == 0)
                or (iteration == (parameters["number_of_iterations"] - 1))
                or (((iteration + 1) % no_of_iterations_per_epoch) == 0)
            )

        if condition:
            if parameters["number_of_epochs"] is not None:
                current_epoch = (iteration + 1) // no_of_iterations_per_epoch
                print(f"Current epoch: {current_epoch}")
            accuracy = calculate_accuracy(
                current_dataset_instance,
                target_network,
                hnet_weights,
                parameters={
                    "device": parameters["device"],
                    "use_batch_norm_memory": use_batch_norm_memory,
                    "number_of_task": current_no_of_task,
                },
                evaluation_dataset="validation",
            )
            print(
                f"Task {current_no_of_task}, iteration: {iteration + 1},"
                f" loss: {loss_current_task.item()}, validation accuracy: {accuracy}"
            )
            if parameters["best_model_selection_method"] == "val_loss":
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    best_hypernetwork = deepcopy(hypernetwork)
                    best_target_network = deepcopy(target_network)
            if (
                parameters["number_of_epochs"] is not None
                and parameters["lr_scheduler"]
                and (((iteration + 1) % no_of_iterations_per_epoch) == 0)
            ):
                print("Finishing the current epoch")
                plateau_scheduler.step(accuracy)

    if parameters["best_model_selection_method"] == "val_loss":
        return best_hypernetwork, best_target_network
    else:
        return hypernetwork, target_network


def build_multiple_task_experiment(dataset_list_of_tasks, parameters, use_chunks=False):
    output_shape = list(dataset_list_of_tasks[0].get_train_outputs())[0].shape[0]
    if parameters["target_network"] == "MLP":
        target_network = MLP(
            n_in=parameters["input_shape"],
            n_out=output_shape,
            hidden_layers=parameters["target_hidden_layers"],
            use_bias=parameters["use_bias"],
            no_weights=True,
        ).to(parameters["device"])
    elif parameters["target_network"] == "epsMLP":
        target_network = epsMLP(
            n_in=parameters["input_shape"],
            n_out=output_shape,
            hidden_layers=parameters["target_hidden_layers"],
            use_bias=parameters["use_bias"],
            no_weights=True,
            epsilon=0.05,
        ).to(parameters["device"])
    if not use_chunks:
        hypernetwork = HMLP(
            target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=parameters["embedding_size"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"],
        ).to(parameters["device"])
    else:
        hypernetwork = ChunkedHMLP(
            target_shapes=target_network.param_shapes,
            chunk_size=parameters["chunk_size"],
            chunk_emb_size=parameters["chunk_emb_size"],
            cond_in_size=parameters["embedding_size"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"],
        ).to(parameters["device"])

    criterion = nn.CrossEntropyLoss()
    dataframe = pd.DataFrame(columns=["after_learning_of_task", "tested_task", "accuracy"])

    use_batch_norm_memory = False
    hypernetwork.train()
    for no_of_task in range(parameters["number_of_tasks"]):
        hypernetwork, target_network = train_single_task(
            hypernetwork,
            target_network,
            criterion,
            parameters,
            dataset_list_of_tasks,
            no_of_task,
        )
        if no_of_task <= (parameters["number_of_tasks"] - 1):
            write_pickle_file(
                f'{parameters["saving_folder"]}/hypernetwork_after_{no_of_task}_task', hypernetwork.weights
            )
        dataframe = evaluate_previous_tasks(
            hypernetwork,
            target_network,
            dataframe,
            dataset_list_of_tasks,
            parameters={
                "device": parameters["device"],
                "use_batch_norm_memory": use_batch_norm_memory,
                "number_of_task": no_of_task,
            },
        )
        dataframe = dataframe.astype({"after_learning_of_task": "int", "tested_task": "int"})
        dataframe.to_csv(f'{parameters["saving_folder"]}/results_{parameters["name_suffix"]}.csv', sep=";")
    return hypernetwork, target_network, dataframe


def main_running_experiments(path_to_datasets, parameters):
    if parameters["dataset"] == "PermutedMNIST":
        dataset_tasks_list = prepare_permuted_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"],
        )
    elif parameters["dataset"] == "SplitMNIST":
        dataset_tasks_list = prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
            number_of_tasks=parameters["number_of_tasks"],
        )
    hypernetwork, target_network, dataframe = build_multiple_task_experiment(
        dataset_tasks_list, parameters, use_chunks=parameters["use_chunks"]
    )
    no_of_last_task = parameters["number_of_tasks"] - 1
    accuracies = dataframe.loc[dataframe["after_learning_of_task"] == no_of_last_task]["accuracy"].values
    row_with_results = (
        f"{dataset_tasks_list[0].get_identifier()};"
        f'{parameters["augmentation"]};'
        f'{parameters["embedding_size"]};'
        f'{parameters["seed"]};'
        f'{str(parameters["hypernetwork_hidden_layers"]).replace(" ", "")};'
        f'{parameters["use_chunks"]};{parameters["chunk_emb_size"]};'
        f'{parameters["target_network"]};'
        f'{str(parameters["target_hidden_layers"]).replace(" ", "")};'
        f'{parameters["resnet_number_of_layer_groups"]};'
        f'{parameters["resnet_widening_factor"]};'
        f'{parameters["norm_regularizer_masking"]};'
        f'{parameters["best_model_selection_method"]};'
        f'{parameters["optimizer"]};'
        f'{parameters["activation_function"]};'
        f'{parameters["learning_rate"]};{parameters["batch_size"]};'
        f'{parameters["beta"]};{parameters["sparsity_parameter"]};'
        f'{parameters["norm"]};{parameters["lambda"]};'
        f"{np.mean(accuracies)};{np.std(accuracies)}"
    )
    append_row_to_file(
        f'{parameters["grid_search_folder"]}'
        f'{parameters["summary_results_filename"]}.csv',
        row_with_results,
    )

    load_path = f'{parameters["saving_folder"]}/results_{parameters["name_suffix"]}.csv'
    plot_heatmap(load_path)

    return hypernetwork, target_network, dataframe

if __name__ == "__main__":
    path_to_datasets = "./Data"
    dataset = "SplitMNIST"
    # 'PermutedMNIST', 'CIFAR100', 'SplitMNIST', 'TinyImageNet', 'CIFAR100_FeCAM_setup'
    part = 1
    create_grid_search = True
    if create_grid_search:
        summary_results_filename = "grid_search_results"
    else:
        summary_results_filename = "summary_results"
    hyperparameters = set_hyperparameters(dataset, grid_search=create_grid_search, part=part)
    header = (
        "dataset_name;augmentation;embedding_size;seed;hypernetwork_hidden_layers;"
        "use_chunks;chunk_emb_size;target_network;target_hidden_layers;"
        "layer_groups;widening;norm_regularizer_masking;final_model;optimizer;"
        "hypernet_activation_function;learning_rate;batch_size;beta;"
        "sparsity;norm;lambda;mean_accuracy;std_accuracy"
    )
    append_row_to_file(f'{hyperparameters["saving_folder"]}{summary_results_filename}.csv', header)

    for no, elements in enumerate(
        product(
            hyperparameters["embedding_sizes"],
            hyperparameters["learning_rates"],
            hyperparameters["betas"],
            hyperparameters["hypernetworks_hidden_layers"],
            hyperparameters["sparsity_parameters"],
            hyperparameters["lambdas"],
            hyperparameters["batch_sizes"],
            hyperparameters["norm_regularizer_masking_opts"],
            hyperparameters["seed"],
        )
    ):
        embedding_size = elements[0]
        learning_rate = elements[1]
        beta = elements[2]
        hypernetwork_hidden_layers = elements[3]
        sparsity_parameter = elements[4]
        lambda_par = elements[5]
        batch_size = elements[6]
        norm_regularizer_masking = elements[7]
        seed = elements[8]

        parameters = {
            "input_shape": hyperparameters["shape"],
            "augmentation": hyperparameters["augmentation"],
            "number_of_tasks": hyperparameters["number_of_tasks"],
            "dataset": dataset,
            "seed": seed,
            "hypernetwork_hidden_layers": hypernetwork_hidden_layers,
            "activation_function": hyperparameters["activation_function"],
            "norm_regularizer_masking": norm_regularizer_masking,
            "use_chunks": hyperparameters["use_chunks"],
            "chunk_size": hyperparameters["chunk_size"],
            "chunk_emb_size": hyperparameters["chunk_emb_size"],
            "target_network": hyperparameters["target_network"],
            "target_hidden_layers": hyperparameters["target_hidden_layers"],
            "resnet_number_of_layer_groups": hyperparameters["resnet_number_of_layer_groups"],
            "resnet_widening_factor": hyperparameters["resnet_widening_factor"],
            "adaptive_sparsity": hyperparameters["adaptive_sparsity"],
            "sparsity_parameter": sparsity_parameter,
            "learning_rate": learning_rate,
            "best_model_selection_method": hyperparameters["best_model_selection_method"],
            "lr_scheduler": hyperparameters["lr_scheduler"],
            "batch_size": batch_size,
            "number_of_epochs": hyperparameters["number_of_epochs"],
            "number_of_iterations": hyperparameters["number_of_iterations"],
            "no_of_validation_samples_per_class": hyperparameters["no_of_validation_samples_per_class"],
            "embedding_size": embedding_size,
            "norm": hyperparameters["norm"],
            "lambda": lambda_par,
            "optimizer": hyperparameters["optimizer"],
            "beta": beta,
            "padding": hyperparameters["padding"],
            "use_bias": hyperparameters["use_bias"],
            "use_batch_norm": hyperparameters["use_batch_norm"],
            "device": hyperparameters["device"],
            "saving_folder": f'{hyperparameters["saving_folder"]}{no}/',
            "grid_search_folder": hyperparameters["saving_folder"],
            "name_suffix": f"mask_sparsity_{sparsity_parameter}",
            "summary_results_filename": summary_results_filename,
        }
        if "no_of_validation_samples" in hyperparameters:
            parameters["no_of_validation_samples"] = hyperparameters["no_of_validation_samples"]

        os.makedirs(parameters["saving_folder"], exist_ok=True)
        # save_parameters(parameters["saving_folder"], parameters, name=f'parameters_{parameters["name_suffix"]}.csv')
        if seed is not None:
            set_seed(seed)

        hypernetwork, target_network, dataframe = main_running_experiments(path_to_datasets, parameters)