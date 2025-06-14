import os
import random
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from hypnettorch.hnets import HMLP

from Attacks.fgsm import FGSM
from Attacks.pgd import PGD
from Attacks.auto_attack import AutoAttack

from IntervalNets.IntervalAlexNet import IntervalAlexNet
# from IntervalNets.IntervalAlexNet_reduced import IntervalAlexNet
from IntervalNets.IntervalResNet18 import IntervalResNet18
from IntervalNets.IntervalMLP import IntervalMLP
from datasets import (
    set_hyperparameters,
    prepare_permuted_mnist_tasks,
    prepare_split_mnist_tasks,
    prepare_rotated_mnist_tasks,
    prepare_split_cifar100_tasks,
    prepare_tinyimagenet_tasks
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pickle_file(filepath: str, device: str):
    return torch.load(filepath, map_location=torch.device(device))
       

def get_attack_instance(attack_name: str, model, weights, epsilon: float, device: str, alpha_pgd: float = None):
    if attack_name == 'PGD':
        return PGD(model, weights, eps=epsilon, alpha=alpha_pgd, steps=10, random_start=False, device=device)
    elif attack_name == 'FGSM':
        return FGSM(model, weights, eps=epsilon, device=device)
    elif attack_name == 'AutoAttack':
        return AutoAttack(model, weights, eps=epsilon, n_iter=10, device=device)
    raise ValueError(f"Unsupported attack type: {attack_name}")

def evaluate_model(data, model, weights, parameters, dataset_split, epsilon_attack, alpha_pgd=None, 
                   attack_type=None, task_id=None):
    model = deepcopy(model).eval()

    input_data = getattr(data, f'get_{dataset_split}_inputs')()
    output_data = getattr(data, f'get_{dataset_split}_outputs')()

    x = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")
    y = data.output_to_torch_tensor(output_data, parameters["device"], mode="inference")
    x.requires_grad = True
    labels = y.argmax(dim=1)

    condition = task_id if parameters.get("use_batch_norm") else None
    logits, _ = model(x, epsilon=parameters["perturbation_epsilon"], weights=weights, condition=condition)
    preds = logits.argmax(dim=1)

    if not attack_type or attack_type.lower() == 'none':
        acc = 100 * (preds == labels).float().mean()
        print(f"Task {task_id} | Clean Accuracy: {acc:.5f}%")
        return acc

    attack = get_attack_instance(attack_type, model, weights, epsilon_attack, parameters["device"], alpha_pgd)
    adv_x = attack.forward(x, labels, task_id)

    adv_logits, _ = model(adv_x, epsilon=parameters["perturbation_epsilon"], weights=weights, condition=condition)
    adv_preds = adv_logits.argmax(dim=1)

    acc = 100 * (adv_preds == labels).float().mean()
    print(f"Task {task_id} | Adversarial Accuracy ({attack_type}): {acc:.5f}%")
    return acc

def prepare_dataset(parameters, path_to_datasets):
    print(f"Preparing dataset: {parameters['dataset']}")
    if parameters["dataset"] == "PermutedMNIST":
        return prepare_permuted_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"]
        )
    elif parameters["dataset"] == "RotatedMNIST":
        return prepare_rotated_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"]
        )
    elif parameters["dataset"] == "SplitMNIST":
        return prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
            number_of_tasks=parameters["number_of_tasks"]
        )
    elif parameters["dataset"] == "CIFAR100":
        return prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
        )
    elif parameters["dataset"] == "TinyImageNet":
        return prepare_tinyimagenet_tasks(
            path_to_datasets,
            seed=parameters["seed"],
        )
    else:
        raise ValueError(f"Unknown dataset type: {parameters['dataset']}")

def create_target_network(parameters, output_size):
    print(f"Creating target network: {parameters['target_network']}")
    if parameters["target_network"] == "IntervalMLP":
        target_network = IntervalMLP(
            n_in=parameters["input_shape"],
            n_out=output_size,
            hidden_layers=parameters["target_hidden_layers"],
            use_bias=parameters["use_bias"],
            no_weights=True,
        ).to(parameters["device"])
    elif parameters["target_network"] == "AlexNet":
        target_network = IntervalAlexNet(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            num_classes=output_size,
            no_weights=True,
            use_batch_norm=parameters["use_batch_norm"],
            bn_track_stats=False,
            distill_bn_stats=False
        ).to(parameters["device"])
    elif parameters["target_network"] == "ResNet":
        target_network = IntervalResNet18(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            use_bias=True,
            use_fc_bias=parameters["use_bias"],
            bottleneck_blocks=False,
            num_classes=output_size,
            num_feature_maps=[16, 32, 64, 128],
            blocks_per_group=[2, 2, 2, 2],
            no_weights=True,
            use_batch_norm=parameters["use_batch_norm"],
            projection_shortcut=True,
            bn_track_stats=False,
            cutout_mod=False,
            mode="default"
        ).to(parameters["device"])
    else:
        raise ValueError(f"Unknown target network type: {parameters['target_network']}")
    return target_network

def create_hypernetwork(parameters, param_shapes):
    print(f"Creating hypernetwork with {parameters['number_of_tasks']} conditional embeddings")
    return HMLP(
        param_shapes,
        uncond_in_size=0,
        cond_in_size=parameters["embedding_sizes"],
        activation_fn=parameters["activation_function"],
        layers=parameters["hnet_hidden_layers"],
        num_cond_embs=parameters["number_of_tasks"]
    ).to(parameters["device"])

def run_experiments(path_to_datasets, parameters, hypernet_path, epsilon_attack, alpha_pgd=None, attack_type=None):
    print(f"Running experiments for {parameters['dataset']} with attack type: {attack_type}")
    datasets = prepare_dataset(parameters, path_to_datasets)
    output_size = datasets[0].get_train_outputs()[0].shape[0]
    target_net = create_target_network(parameters, output_size)
    hypernet = create_hypernetwork(parameters, target_net.param_shapes)
    hnet_weights = load_pickle_file(hypernet_path, parameters["device"])

    updated_weights = {
        hypernet_key: hnet_weights[loaded_key]
        for (hypernet_key, _), loaded_key in zip(hypernet.named_parameters(), hnet_weights.keys())
    }

    results = []
    hypernet.load_state_dict(updated_weights, strict=False)

    for task_id, task_data in enumerate(datasets):
        print(f"Evaluating task {task_id}...")
        task_weights = hypernet.forward(cond_id=task_id)
        acc = evaluate_model(
            task_data, 
            target_net, 
            task_weights, 
            parameters,
            dataset_split='test',
            epsilon_attack=epsilon_attack,
            alpha_pgd=alpha_pgd,
            attack_type=attack_type,
            task_id=task_id
        )
        print(f"Task {task_id} | Accuracy: {acc:.5f}%")
        results.append({"tested_task": task_id, "accuracy": acc.item()})

    df = pd.DataFrame(results).astype({"tested_task": int})

    results_path = os.path.join(parameters["saving_folder"], f"results_attacks_{attack_type}.csv")
    df.to_csv(results_path, sep=";", index=False)
    print(f"Results saved to {results_path}")
    return df

def run_multiple_seeds(dataset, path_to_datasets, hypernet_model_path_template, epsilon_attack, 
                       alpha_pgd, attack_type="FGSM", seeds=list(range(5))):
    grid_search = False
    hyperparams = set_hyperparameters(dataset, grid_search)
    
    for seed in seeds:
        parameters = {
            "embedding_sizes": hyperparams["embedding_sizes"][0],
            "activation_function": hyperparams["activation_function"],
            "hnet_hidden_layers": hyperparams["hypernetworks_hidden_layers"][0],
            "input_shape": hyperparams["shape"],
            "augmentation": hyperparams["augmentation"],
            "number_of_tasks": hyperparams["number_of_tasks"],
            "dataset": dataset,
            "seed": seed + 1,
            "target_network": hyperparams["target_network"],
            "target_hidden_layers": hyperparams["target_hidden_layers"],
            "no_of_validation_samples_per_class": hyperparams["no_of_validation_samples_per_class"],
            "padding": hyperparams["padding"],
            "use_bias": hyperparams["use_bias"],
            "device": hyperparams["device"],
            "perturbation_epsilon": hyperparams["perturbation_epsilons"][0],
            "use_batch_norm": hyperparams["use_batch_norm"],
            "saving_folder": f"{hypernet_model_path_template}{seed}",
        }
       
        if "no_of_validation_samples" in hyperparams:
            parameters["no_of_validation_samples"] = hyperparams["no_of_validation_samples"]
        os.makedirs(parameters["saving_folder"], exist_ok=True)
        if parameters["seed"] is not None:
            set_seed(parameters["seed"])
        hypernet_path = hypernet_model_path_template.format(seed=seed)
        print(f"\nStarting experiment with seed {seed + 1}")
        run_experiments(
            path_to_datasets=path_to_datasets,
            parameters=parameters,
            hypernet_path=f"{hypernet_path}/{seed}/hnet.pt",
            epsilon_attack=epsilon_attack,
            alpha_pgd=alpha_pgd,
            attack_type=attack_type
        )
    # Aggregate and save results
    all_dfs = []
    for seed in seeds:
        result_path = os.path.join(f"{hypernet_model_path_template}{seed}/", f"results_attacks_{attack_type}.csv")
        if os.path.exists(result_path):
            df = pd.read_csv(result_path, sep=";")
            df["seed"] = seed
            all_dfs.append(df)
        else:
            print(f"Warning: Missing results file for seed {seed}: {result_path}")

    if all_dfs:
        combined_df = pd.concat(all_dfs)
        stats_df = combined_df.groupby("seed")["accuracy"].agg(["mean", "std"]).reset_index()
        stats_df.rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"}, inplace=True)
        aggregated_results_path = os.path.join(hypernet_model_path_template, f"aggregated_results_attacks_{attack_type}.csv")
        stats_df.to_csv(aggregated_results_path, sep=";", index=False)
        print(f"Averaged results saved to {aggregated_results_path}\n")
        print(f'Averaged mean: {stats_df["accuracy_mean"].mean()}, stdev: {stats_df["accuracy_mean"].std()}')
    else:
        print("No results were found to aggregate.")

if __name__ == "__main__":
    run_multiple_seeds(
        dataset="TinyImageNet",
        path_to_datasets="./Data",
        hypernet_model_path_template="./Results/TinyImageNet/eps_0/",
        epsilon_attack=8/255.0,
        alpha_pgd=4/255.0,
        attack_type="FGSM",
        seeds=list(range(1))
    )