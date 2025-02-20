from datasets import (
    set_hyperparameters,
)
from main import (
    apply_mask_to_weights_of_network,
    get_number_of_batch_normalization_layer,
    set_seed,
    prepare_network_sparsity,
)
from copy import deepcopy
from hypnettorch.hnets import HMLP
from evaluation import prepare_target_network
import torch
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from epsMLP import epsMLP

def calculate_distance_between_networks(hyperparameters,
                                        layer,
                                        no_of_last_task,
                                        output_shape,
                                        path_to_models):
    """
    Calculates distance between a selected target network layer
    after i-th and j-th tasks.
    """
    weights = {}
    distances = np.zeros((no_of_last_task + 1, no_of_last_task + 1))
    # key denotes a task
    for cur_task in range(1, no_of_last_task + 1):
        print(cur_task)
        target_network = prepare_target_network(
            hyperparameters, output_shape)
        target_network.load_state_dict(
            torch.load(f'{path_to_models}target_network_after_{cur_task+1}_task'))
        target_network.eval()
        this_layer_weights = target_network._weights[2 * layer].flatten()
        this_layer_bias = target_network._weights[2 * layer + 1].flatten()
        weights[cur_task] = torch.cat((this_layer_weights, this_layer_bias))

    for cur_task in range(no_of_last_task + 1):
        for other_task in range(cur_task + 1, no_of_last_task + 1):
            distances[cur_task, other_task] = torch.sum(
                torch.abs(weights[cur_task] - weights[other_task])).item()
    np.savetxt(
        f'{path_to_models}distances_layer_{layer}.csv',
        distances,
        delimiter=' & ',
        fmt='%.2f'
    )
    return distances


if __name__ == "__main__":
    dataset = "PermutedMNIST"
    path_to_models = (
        '/home/ukasz/ibp/Results/permuted_mnist_best_hyperparams/0/'
        'Results/grid_search/permuted_mnist/0/'
        # '/home/kksiazek/Desktop/Continual_learning/HyperMask/hyperbinarynetwork/'
        # 'Results/permuted_mnist_best_hyperparams/0/'
    )
    part = 0
    grid_search = False


    hyperparameters = set_hyperparameters(
        dataset,
        grid_search=grid_search,
        part=part
    )

    output_shape = 10
    no_of_last_task = 9

    # Weights after following tasks for a given layer
    # layer 0: 1000x1024 and 1000 bias
    # layer 1: 1000x1000 and 1000 bias
    # layer 2: 10x1000 and 10 bias
    total_distances = []
    for layer in range(3):
        total_distances.append(
            calculate_distance_between_networks(
                hyperparameters,
                layer,
                no_of_last_task,
                output_shape,
                path_to_models)
        )


    # Configure  hypernetwork
    no_of_task = 3
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
        target_network
    )
    hypernetwork = HMLP(
        target_network.param_shapes[no_of_batch_norm_layers:],
        uncond_in_size=0,
        cond_in_size=hyperparameters["embedding_sizes"][0],
        activation_fn=hyperparameters["activation_function"],
        layers=hyperparameters["hypernetworks_hidden_layers"][0],
        num_cond_embs=hyperparameters["number_of_tasks"],
    ).to(hyperparameters["device"])
    hypernetwork.load_state_dict(
        torch.load(f'{path_to_models}hypernetwork_after_{no_of_last_task}_task')
    )
    hypernetwork.eval()
    hypernetwork_output = hypernetwork.forward(
        cond_id=no_of_task
    )
    masks = prepare_network_sparsity(
        hypernetwork_output,
        hyperparameters["sparsity_parameters"]
    )
    target_weights = deepcopy(target_network.weights)
    target_masked_weights = apply_mask_to_weights_of_network(
        target_weights,
        masks
    )

    no_of_selected_layer = 4
    # mask_selected_layer = hypernetwork_output[no_of_selected_layer].cpu().detach().numpy()
    # mask_selected_layer = mask_selected_layer.flatten()
    mask_selected_layer = target_masked_weights[4].detach().cpu().numpy()
    fig, ax = plt.subplots()
    pos = ax.matshow(mask_selected_layer, cmap=mpl.colormaps['Blues'])
    fig.colorbar(pos, ax=ax)


    sns.set_theme()
    plt.hist(first_layer, bins=150, density=True)
