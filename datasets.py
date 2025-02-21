import os
import numpy as np
import torch
from hypnettorch.data.special import permuted_mnist
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from TinyImageNet import TinyImageNet
from CIFAR100_FeCAM import SplitCIFAR100Data_FeCAM
import attacks

def generate_random_permutations(
    shape_of_data_instance, number_of_permutations
):
    """
    Prepare a list of random permutations of the selected shape
    for continual learning tasks.

    Arguments:
    ----------
      *shape_of_data_instance*: a number defining shape of the dataset
      *number_of_permutations*: int, a number of permutations that will
                                be prepared; it corresponds to the total
                                number of tasks
      *seed*: int, optional argument, default: None
              if one would get deterministic results
    """
    list_of_permutations = []
    for _ in range(number_of_permutations):
        list_of_permutations.append(
            np.random.permutation(shape_of_data_instance)
        )
    return list_of_permutations


def prepare_split_cifar100_tasks(
    datasets_folder, validation_size, use_augmentation, use_cutout=False
):
    """
    Prepare a list of 10 tasks with 10 classes per each task.
    i-th task, where i in {0, 1, ..., 9} will store samples
    from classes {10*i, 10*i + 1, ..., 10*i + 9}.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    'apply_cutout' option form 'torch_input_transforms'.
    """
    handlers = []
    for i in range(0, 100, 10):
        handlers.append(
            SplitCIFAR100Data(
                datasets_folder,
                use_one_hot=True,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_cutout=use_cutout,
                labels=range(i, i + 10),
            )
        )
    return handlers


def prepare_split_cifar100_tasks_aka_FeCAM(
    datasets_folder,
    number_of_tasks,
    no_of_validation_samples_per_class,
    use_augmentation,
    use_cutout=False,
):
    """
    Prepare a list of 5, 10 or 20 incremental tasks with 20, 10 or 5 classes,
    respectively, per each task. Furthermore, the first task contains
    a higher number of classes, i.e. 50 or 40. Therefore, in these cases,
    the total number of tasks is equal to 6, 11 or 21.
    Also, there is a possibility of 5 tasks with 20 classes per each.
    The order of classes is the same like in FeCAM, also the scenarios
    are constructed in such a way to enable a fair comparison with FeCAM

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *number_of_tasks* (int) Defines how many continual learning tasks
                        will be created. Possible options: 6, 11 or 21
      *no_of_validation_samples_per_class*: (int) The number of validation
                                            samples in a single class
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    'apply_cutout' option form 'torch_input_transforms'.
    """
    # FeCAM considered four scenarios: 5, 10 and 20 incremental tasks
    # and 5 tasks with the equal number of classes
    assert number_of_tasks in [5, 6, 11, 21]
    # The order of image classes in the case of FeCAM was not 0-10, 11-20, etc.,
    # but it was chosen randomly by the authors, and was at follows:
    class_orders = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
        94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
        84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
        69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
        17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
        1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
        38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
        40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
        98, 13, 99, 7, 34, 55, 54, 26, 35, 39
    ]
    # Incremental tasks from Table I, FeCAM
    if number_of_tasks == 6:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([10 for i in range(5)])
    elif number_of_tasks == 11:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([5 for i in range(10)])
    elif number_of_tasks == 21:
        numbers_of_classes_per_tasks = [40]
        numbers_of_classes_per_tasks.extend([3 for i in range(20)])
    # Tasks with the equal number of elements, Table V, FeCAM
    elif number_of_tasks == 5:
        numbers_of_classes_per_tasks = [20 for i in range(5)]

    handlers = []
    for i in range(len(numbers_of_classes_per_tasks)):
        current_number_of_tasks = numbers_of_classes_per_tasks[i]
        validation_size = (
            no_of_validation_samples_per_class * current_number_of_tasks
        )
        handlers.append(
            SplitCIFAR100Data_FeCAM(
                datasets_folder,
                use_one_hot=True,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_cutout=use_cutout,
                labels=class_orders[
                    (i * current_number_of_tasks) : (
                        (i + 1) * current_number_of_tasks
                    )
                ],
            )
        )
    return handlers


def prepare_tinyimagenet_tasks(
    datasets_folder, seed, validation_size=250, number_of_tasks=40
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the TinyImageNet dataset according to the WSN setup.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which TinyImageNet
                         is stored / will be downloaded
      *seed*: (int) Necessary for the preparation of random permutation
              of the order of classes in consecutive tasks.
      *validation_size*: (optional int) defines the number of validation
                         samples in each task, by default it is 250 like
                         in the case of WSN
      *number_of_tasks*: (optional int) defines the number of continual
                         learning tasks (by default: 40)

    Returns a list of TinyImageNet objects.
    """
    # Set randomly the order of classes
    rng = np.random.default_rng(seed)
    class_permutation = rng.permutation(200)
    # 40 classification tasks with 5 classes in each
    handlers = []
    for i in range(0, 5 * number_of_tasks, 5):
        current_labels = class_permutation[i : (i + 5)]
        print(f"Order of classes in the current task: {current_labels}")
        handlers.append(
            TinyImageNet(
                data_path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=True,
                labels=current_labels,
            )
        )
    return handlers


def prepare_permuted_mnist_tasks(
    datasets_folder, input_shape, number_of_tasks, padding, validation_size
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the PermutedMNIST dataset.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *input_shape*: (int) a number defining shape of the dataset
      *validation_size*: (int) The number of validation samples

    Returns a list of PermutedMNIST objects.
    """
    permutations = generate_random_permutations(input_shape, number_of_tasks)
    return permuted_mnist.PermutedMNISTList(
        permutations,
        datasets_folder,
        use_one_hot=True,
        padding=padding,
        validation_size=validation_size,
    )


def prepare_split_mnist_tasks(
    datasets_folder, validation_size, use_augmentation, number_of_tasks=5
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the SplitMNIST dataset. By default, it should be
    5 task containing consecutive pairs of classes:
    [0, 1], [2, 3], [4, 5], [6, 7] and [8, 9].

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (bool) defines whether dataset augmentation
                          will be applied
      *number_of_tasks* (int) a number defining the number of learning
                        tasks, by default 5.

    Returns a list of SplitMNIST objects.
    """
    return get_split_mnist_handlers(
        datasets_folder,
        use_one_hot=True,
        validation_size=validation_size,
        num_classes_per_task=2,
        num_tasks=number_of_tasks,
        use_torch_augmentation=use_augmentation,
    )


def set_hyperparameters(dataset, grid_search=False, part=0):
    """
    Set hyperparameters of the experiments, both in the case of grid search
    optimization and a single network run.

    Arguments:
    ----------
      *dataset*: 'PermutedMNIST', 'SplitMNIST' or 'CIFAR100'
      *grid_search*: (Boolean optional) defines whether a hyperparameter
                     optimization should be performed or hyperparameters
                     for just a single run have to be returned
      *part* (only for SplitMNIST or CIFAR100!) selects a subset
             of hyperparameters for optimization (by default 0)

    Returns a dictionary with necessary hyperparameters.
    """
    if dataset == "PermutedMNIST":
        # us or paper
        param_owner = 'paper'
        if param_owner == 'us':
            if grid_search:
                hyperparams = {
                    "embedding_sizes": [24],
                    "learning_rates": [0.001],
                    "batch_sizes": [128],
                    "norm_regularizer_masking_opts": [True, False],
                    "betas": [0.001, 0.0005, 0.005],
                    "hypernetworks_hidden_layers": [[100, 100]],
                    "sparsity_parameters": [0],
                    "lambdas": [0.001, 0.0005],
                    "best_model_selection_method": "val_loss",
                    "saving_folder": "./Results/grid_search/permuted_mnist/",
                    # not for optimization, just for multiple cases
                    "seed": [1, 2, 3, 4, 5],
                }

            else:
                # Best hyperparameters
                hyperparams = {
                    "seed": [1, 2, 3, 4, 5],
                    "embedding_sizes": [24],
                    "sparsity_parameters": [0],
                    "learning_rates": [0.001],
                    "batch_sizes": [128],
                    "betas": [0.0005],
                    "lambdas": [0.001],
                    "norm_regularizer_masking_opts": [True],
                    "hypernetworks_hidden_layers": [[100, 100]],
                    "best_model_selection_method": "last_model",
                    "saving_folder": "./Results/permuted_mnist_best_hyperparams/",
                }

            # Both in the grid search and individual runs
            hyperparams["lr_scheduler"] = False
            hyperparams["number_of_iterations"] = 5000
            hyperparams["number_of_epochs"] = None
            hyperparams["no_of_validation_samples"] = 5000
            hyperparams["no_of_validation_samples_per_class"] = 500
            hyperparams["target_hidden_layers"] = [1000, 1000]
            hyperparams["target_network"] = "epsMLP"
            hyperparams["resnet_number_of_layer_groups"] = None
            hyperparams["resnet_widening_factor"] = None
            hyperparams["optimizer"] = "adam"
            hyperparams["chunk_size"] = 100
            hyperparams["chunk_emb_size"] = 8
            hyperparams["use_chunks"] = False
            hyperparams["adaptive_sparsity"] = True
            hyperparams["use_batch_norm"] = False
            # Directly related to the MNIST dataset
            hyperparams["padding"] = 2
            hyperparams["shape"] = (28 + 2 * hyperparams["padding"]) ** 2
            hyperparams["number_of_tasks"] = 10
            hyperparams["augmentation"] = False
        # paper
        else:
            if grid_search:
                hyperparams = {
                    "embedding_sizes": [24],
                    "learning_rates": [0.001],
                    "batch_sizes": [32],
                    "norm_regularizer_masking_opts": [True, False],
                    "betas": [0.001, 0.0005, 0.005, 0.0001, 0.01, 0.05],
                    "hypernetworks_hidden_layers": [[100, 100]],
                    "sparsity_parameters": [0],
                    "lambdas": [0.001, 0.0005, 0.0001, 0.005, 0.01, 0.05],
                    "best_model_selection_method": "val_loss",
                    "saving_folder": "./Results/grid_search/permuted_mnist/",
                    # not for optimization, just for multiple cases
                    "seed": [1, 2, 3, 4, 5],
                }

            else:
                # Best hyperparameters
                hyperparams = {
                    "seed": [1, 2, 3, 4, 5],
                    "embedding_sizes": [24],
                    "sparsity_parameters": [0],
                    "learning_rates": [0.001],
                    "batch_sizes": [32],
                    "betas": [0.0005],
                    "lambdas": [0.001],
                    "norm_regularizer_masking_opts": [True],
                    "hypernetworks_hidden_layers": [[100,100]],
                    "best_model_selection_method": "last_model",
                    "saving_folder": "./Results/permuted_mnist_best_hyperparams/",
                }

            # Both in the grid search and individual runs
            hyperparams["lr_scheduler"] = False
            hyperparams["number_of_iterations"] = 10000
            hyperparams["number_of_epochs"] = 10
            hyperparams["no_of_validation_samples"] = 1000
            hyperparams["no_of_validation_samples_per_class"] = 100
            hyperparams["target_hidden_layers"] = [256,256, 10]
            hyperparams["target_network"] = "epsMLP"
            hyperparams["resnet_number_of_layer_groups"] = None
            hyperparams["resnet_widening_factor"] = None
            hyperparams["optimizer"] = "adam"
            hyperparams["chunk_size"] = 100
            hyperparams["chunk_emb_size"] = 8
            hyperparams["use_chunks"] = False
            hyperparams["adaptive_sparsity"] = True
            hyperparams["use_batch_norm"] = False
            # Directly related to the MNIST dataset
            hyperparams["padding"] = 2
            hyperparams["shape"] = (28 + 2 * hyperparams["padding"]) ** 2
            hyperparams["number_of_tasks"] = 10
            hyperparams["augmentation"] = False

    elif dataset == "CIFAR100":
        if grid_search:
            hyperparams = {
                "seed": [5],
                "sparsity_parameters": [0],
                "embedding_sizes": [48],
                "betas": [0.01, 0.05, 0.1, 1],
                "lambdas": [0.01, 0.1, 1],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True,
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
                hyperparams["use_chunks"] = False
                hyperparams["use_batch_norm"] = False
            else:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/grid_search/CIFAR_100_part_{part}/"

        else:
            # Best hyperparameters for ResNet
            hyperparams = {
                "seed": [1, 2, 3, 4, 5],
                "embedding_sizes": [48],
                "sparsity_parameters": [0],
                "betas": [0.01],
                "lambdas": [1],
                "batch_sizes": [32],
                "learning_rates": [0.001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
            }
            if part == 0:
                # ResNet
                pass
            elif part == 1:
                # ZenkeNet
                hyperparams["lambdas"] = [0.01]
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
            else:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_best_hyperparams_part_{part}/"
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 500
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 10
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == "TinyImageNet":
        if grid_search:
            hyperparams = {
                "seed": [5],
                "sparsity_parameters": [0, 30],
                "embedding_sizes": [48],
                "betas": [0.01, 0.1],
                "lambdas": [0.01, 0.1],
                "learning_rates": [0.001],
                "batch_sizes": [16],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[10, 10], [100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True,
                "saving_folder": f"./Results/TinyImageNet_grid_search_part_{part}/",
            }
            if part == 0:
                pass
            elif part in [1, 2, 3]:
                # ZenkeNet
                hyperparams["seed"] = [6]
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
                hyperparams["use_batch_norm"] = False
                hyperparams["hypernetworks_hidden_layers"] = [[100, 100]]
                hyperparams["learning_rates"] = [0.001]
                if part == 1:
                    hyperparams["betas"] = [0.01, 0.1, 1.0]
                    hyperparams["lambdas"] = [0.01, 0.1, 1.0]
                    hyperparams["sparsity_parameters"] = [0, 50, 70]
                    hyperparams["embedding_sizes"] = [96]
                    hyperparams["learning_rates"] = [0.005]
                elif part == 2:
                    hyperparams["embedding_sizes"] = [128]
                    hyperparams["betas"] = [0.01, 0.1, 1.0]
                    hyperparams["lambdas"] = [0.01, 0.1, 1.0]
                    hyperparams["sparsity_parameters"] = [0, 30, 50]
                elif part == 3:
                    hyperparams["embedding_sizes"] = [192]
                    hyperparams["sparsity_parameters"] = [0, 50]
                    hyperparams["betas"] = [0.001, 0.01, 0.1, 1]
                    hyperparams["lambdas"] = [0.001, 0.01, 0.1, 1]
            else:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/Tiny_Zenke_grid_search_part_{part}/"
        else:
            # ResNet
            hyperparams = {
                "seed": [5, 6, 7, 8, 9],
                "embedding_sizes": [96],
                "sparsity_parameters": [0],
                "betas": [1],
                "lambdas": [0.1],
                "batch_sizes": [16],
                "learning_rates": [0.0001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100, 100]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 10,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
                "saving_folder": "./Results/TinyImageNet/ResNet_best_hyperparams/",
            }
            if part == 0:
                pass
            # ZenkeNet
            elif part == 1:
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
                hyperparams["use_batch_norm"] = False
                hyperparams["sparsity_parameters"] = [50]
                hyperparams["betas"] = [0.01]
                hyperparams["lambdas"] = [1.0]
                hyperparams["learning_rates"] = [0.001]
                hyperparams[
                    "saving_folder"
                ] = f"./Results/TinyImageNet/ZenkeNet_best_hyperparams/"
            else:
                raise ValueError(f"Wrong argument: {part}!")
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 250
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 64
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 12288
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 40
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == "SplitMNIST":
        if grid_search:
            hyperparams = {
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "norm_regularizer_masking_opts": [False],
                "betas": [0.001],
                "hypernetworks_hidden_layers": [[25, 25]],
                "sparsity_parameters": [0],
                "lambdas": [0.001],
                # Seed is not for optimization but for ensuring multiple results
                "seed": [1, 2, 3, 4, 5],
                "best_model_selection_method": "last_model",
                "embedding_sizes": [128],
                "augmentation": True,
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams["embedding_sizes"] = [96]
                hyperparams["hypernetworks_hidden_layers"] = [[50, 50]]
                hyperparams["betas"] = [0.01]
                hyperparams["sparsity_parameters"] = [30]
                hyperparams["lambdas"] = [0.0001]
            elif part == 2:
                hyperparams["sparsity_parameters"] = [30]
                hyperparams["norm_regularizer_masking_opts"] = [True]
            else:
                raise ValueError("Not implemented subset of hyperparameters!")

            hyperparams["saving_folder"] = "./Results/grid_search/split_mnist/"

        else:
            # Best hyperparameters
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [72],
                "sparsity_parameters": [30],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.001],
                "lambdas": [0.001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[75, 75]],
                "augmentation": True,
                "best_model_selection_method": "last_model",
                "saving_folder": "./Results/split_mnist_test/",
            }
        hyperparams["lr_scheduler"] = False
        hyperparams["target_network"] = "epsMLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["number_of_iterations"] = 2000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["no_of_validation_samples_per_class"] = 200
        hyperparams["target_hidden_layers"] = [400, 400]
        hyperparams["shape"] = 28**2
        hyperparams["number_of_tasks"] = 5
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 72
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None

    elif dataset == "CIFAR100_FeCAM_setup":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "sparsity_parameters": [0],
                "betas": [0.1],
                "lambdas": [1],
                "batch_sizes": [32],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams["embedding_sizes"] = [24, 48, 96]
                hyperparams["learning_rates"] = [0.0001, 0.001, 0.01]
                hyperparams["hypernetworks_hidden_layers"] = [[100], [200]]
                hyperparams["number_of_tasks"] = 5
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_FeCAM_setup_part_{part}/"
        else:
            # Best hyperparameters for ResNet
            hyperparams = {
                "seed": [1, 2, 3, 4, 5],
                "embedding_sizes": [48],
                "sparsity_parameters": [0],
                "betas": [0.01],
                "lambdas": [1],
                "batch_sizes": [32],
                "learning_rates": [0.0001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[200]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
            }
            # FeCAM considered three incremental scenarios: with 6, 11 and 21 tasks
            # ResNet - parts 0, 1 and 2
            # ZenkeNet - parts 3, 4 and 5
            # Also, one scenario with equal number of classes: ResNet - part 6
            if part in [0, 3]:
                hyperparams["number_of_tasks"] = 6
            elif part in [1, 4]:
                hyperparams["number_of_tasks"] = 11
            elif part in [2, 5]:
                hyperparams["number_of_tasks"] = 21
            elif part in [6, 7]:
                hyperparams["number_of_tasks"] = 5
            if part in [3, 4, 5, 7]:
                hyperparams["lambdas"] = [0.01]
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
            if part not in [0, 1, 2, 3, 4, 5, 6, 7]:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_FeCAM_part_{part}/"
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "ResNetF", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    else:
        raise ValueError("This dataset is not implemented!")

    # General hyperparameters
    hyperparams["activation_function"] = torch.nn.ELU()
    hyperparams["norm"] = 1  # L1 norm
    hyperparams["use_bias"] = True
    hyperparams["save_consecutive_masks"] = False
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["dataset"] = dataset
    os.makedirs(hyperparams["saving_folder"], exist_ok=True)
    return hyperparams


if __name__ == "__main__":
    datasets_folder = "./Data"
    os.makedirs(datasets_folder, exist_ok=True)
    validation_size = 500
    use_data_augmentation = False
    use_cutout = False

    split_cifar100_list = prepare_split_cifar100_tasks(
        datasets_folder, validation_size, use_data_augmentation, use_cutout
    )


#
# def calculate_accuracy(data, target_network, weights, parameters, evaluation_dataset):
#     target_network.eval()
#     #with torch.no_grad():
#     if 1==1:
#         if evaluation_dataset == "validation":
#             input_data = data.get_val_inputs()
#             output_data = data.get_val_outputs()
#         elif evaluation_dataset == "test":
#             input_data = data.get_test_inputs()
#             output_data = data.get_test_outputs()
#
#         test_input = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")
#         test_output = data.output_to_torch_tensor(output_data, parameters["device"], mode="inference")
#         test_input.requires_grad = True
#         gt_classes = test_output.max(dim=1)[1]
#         if parameters["use_batch_norm_memory"]:
#             logits = target_network.forward(test_input, weights=weights, condition=parameters["number_of_task"])
#         else:
#             logits = target_network.forward(test_input, weights=weights)
#         if len(logits) == 2:
#             logits, _ = logits
#         predictions = logits.max(dim=1)[1]
#         attack = "FGSM"
#         if evaluation_dataset == "test" and attack is not None:
#             target_network.mode = 'test'
#             if attack == 'PGD':
#                 attack = PGD(target_network, eps=1/255, alpha=1/255, steps=10, random_start=False)
#             elif attack == 'FGSM':
#                 attack = FGSM(target_network, eps=8/255)
#             elif attack == 'AutoAttack':
#                 attack = AutoAttack(target_network, norm='Linf', eps=8/255, version='standard', seed=None, verbose=False)
#             adv_images = attack(test_input, gt_classes)
#             adv_logits = target_network.forward(adv_images, weights=weights)
#             perturbed_pred = adv_logits.max(dim=1)[1]
#             perturbed_acc = 100 * (torch.sum(gt_classes == perturbed_pred).float() / gt_classes.numel())
#             target_network.mode = None
#             return perturbed_acc
#     accuracy = torch.sum(gt_classes == predictions).float() / gt_classes.numel() * 100.0
#     return accuracy



#
# import os
# import random
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch.optim as optim
# from hypnettorch.mnets import MLP
# from epsMLP import epsMLP
# from hypnettorch.hnets import HMLP
# from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
# from torch import nn
# from datetime import datetime
# from itertools import product
# from copy import deepcopy
# from retry import retry
# from datasets import set_hyperparameters, prepare_permuted_mnist_tasks, prepare_split_mnist_tasks
# from torchattacks import PGD, FGSM, AutoAttack
#
# def set_seed(value):
#     random.seed(value)
#     np.random.seed(value)
#     torch.manual_seed(value)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# def append_row_to_file(filename, elements):
#     if not filename.endswith(".csv"):
#         filename += ".csv"
#     filename = filename.replace(".pt", "")
#     with open(filename, "a+") as stream:
#         np.savetxt(stream, np.array(elements)[np.newaxis], delimiter=";", fmt="%s")
#
#
# def write_pickle_file(filename, object_to_save):
#     torch.save(object_to_save, f"{filename}.pt")
#
#
# @retry((OSError, IOError))
# def load_pickle_file(filename):
#     return torch.load(filename)
#
#
# def get_shapes_of_network(model):
#     shapes_of_model = []
#     for layer in model.weights:
#         shapes_of_model.append(list(layer.shape))
#     return shapes_of_model
#
#
# def calculate_number_of_iterations(number_of_samples, batch_size, number_of_epochs):
#     no_of_iterations_per_epoch = int(np.ceil(number_of_samples / batch_size))
#     total_no_of_iterations = int(no_of_iterations_per_epoch * number_of_epochs)
#     return no_of_iterations_per_epoch, total_no_of_iterations
#
#
# def calculate_accuracy(data, target_network, weights, parameters, evaluation_dataset):
#     target_network.eval()
#     #with torch.no_grad():
#     if 1==1:
#         if evaluation_dataset == "validation":
#             input_data = data.get_val_inputs()
#             output_data = data.get_val_outputs()
#         elif evaluation_dataset == "test":
#             input_data = data.get_test_inputs()
#             output_data = data.get_test_outputs()
#
#         test_input = data.input_to_torch_tensor(input_data, parameters["device"], mode="inference")
#         test_output = data.output_to_torch_tensor(output_data, parameters["device"], mode="inference")
#         test_input.requires_grad = True
#         gt_classes = test_output.max(dim=1)[1]
#         if parameters["use_batch_norm_memory"]:
#             logits = target_network.forward(test_input, weights=weights, condition=parameters["number_of_task"])
#         else:
#             logits = target_network.forward(test_input, weights=weights)
#         if len(logits) == 2:
#             logits, _ = logits
#         predictions = logits.max(dim=1)[1]
#         attack = "FGSM"
#         if evaluation_dataset == "test" and attack is not None:
#             target_network.mode = 'test'
#             if attack == 'PGD':
#                 attack = PGD(target_network, eps=1/255, alpha=1/255, steps=10, random_start=False)
#             elif attack == 'FGSM':
#                 attack = FGSM(target_network, eps=8/255)
#             elif attack == 'AutoAttack':
#                 attack = AutoAttack(target_network, norm='Linf', eps=8/255, version='standard', seed=None, verbose=False)
#             adv_images = attack(test_input, gt_classes)
#             adv_logits = target_network.forward(adv_images, weights=weights)
#             perturbed_pred = adv_logits.max(dim=1)[1]
#             perturbed_acc = 100 * (torch.sum(gt_classes == perturbed_pred).float() / gt_classes.numel())
#             target_network.mode = None
#             return perturbed_acc
#     accuracy = torch.sum(gt_classes == predictions).float() / gt_classes.numel() * 100.0
#     return accuracy
#
#
# def evaluate_previous_tasks(hypernetwork, target_network, dataframe_results, list_of_permutations, parameters):
#     hypernetwork.eval()
#     target_network.eval()
#     for task in range(parameters["number_of_task"] + 1):
#         currently_tested_task = list_of_permutations[task]
#         target_weights = target_network.weights
#         accuracy = calculate_accuracy(
#             currently_tested_task,
#             target_network,
#             target_weights,
#             parameters=parameters,
#             evaluation_dataset="test",
#         )
#         result = {
#             "after_learning_of_task": parameters["number_of_task"],
#             "tested_task": task,
#             "accuracy": accuracy.cpu().item(),
#         }
#         print(f"Accuracy for task {task}: {accuracy}%.")
#         dataframe_results = dataframe_results.append(result, ignore_index=True)
#     return dataframe_results
#
#
# def save_parameters(saving_folder, parameters, name=None):
#     if name is None:
#         current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         name = f"parameters_{current_time}.csv"
#     with open(f"{saving_folder}/{name}", "w") as file:
#         for key in parameters.keys():
#             file.write(f"{key};{parameters[key]}\n")
#
#
# def plot_heatmap(load_path):
#     dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
#     dataframe = dataframe.astype({"after_learning_of_task": "int32", "tested_task": "int32"})
#     table = dataframe.pivot("after_learning_of_task", "tested_task", "accuracy")
#     sns.heatmap(table, annot=True, fmt=".1f")
#     plt.tight_layout()
#     plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
#     plt.close()
#
#
# def train_single_task(hypernetwork, target_network, criterion, parameters, dataset_list_of_tasks, current_no_of_task):
#     if parameters["optimizer"] == "adam":
#         optimizer = torch.optim.Adam(
#             [*hypernetwork.parameters()], lr=parameters["learning_rate"])
#     elif parameters["optimizer"] == "rmsprop":
#         optimizer = torch.optim.RMSprop([*hypernetwork.parameters()], lr=parameters["learning_rate"])
#     if parameters["best_model_selection_method"] == "val_loss":
#         best_hypernetwork = deepcopy(hypernetwork)
#         best_target_network = deepcopy(target_network)
#         best_val_accuracy = 0.0
#     hypernetwork.train()
#     print(f"task: {current_no_of_task}")
#     use_batch_norm_memory = False
#     current_dataset_instance = dataset_list_of_tasks[current_no_of_task]
#     if parameters["number_of_epochs"] is not None:
#         (
#             no_of_iterations_per_epoch,
#             parameters["number_of_iterations"],
#         ) = calculate_number_of_iterations(
#             current_dataset_instance.num_train_samples,
#             parameters["batch_size"],
#             parameters["number_of_epochs"],
#         )
#         if parameters["lr_scheduler"]:
#             plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer,
#                 "max",
#                 factor=np.sqrt(0.1),
#                 patience=5,
#                 min_lr=0.5e-6,
#                 cooldown=0,
#                 verbose=True,
#             )
#     for iteration in range(parameters["number_of_iterations"]):
#         current_batch = current_dataset_instance.next_train_batch(parameters["batch_size"])
#         tensor_input = current_dataset_instance.input_to_torch_tensor(
#             current_batch[0], parameters["device"], mode="train"
#         )
#         tensor_output = current_dataset_instance.output_to_torch_tensor(
#             current_batch[1], parameters["device"], mode="train"
#         )
#         gt_output = tensor_output.max(dim=1)[1]
#         optimizer.zero_grad()
#         target_weights = target_network.weights
#         if parameters["target_network"] == "epsMLP":
#             default_eps = target_network.epsilon
#             total_iterations = parameters["number_of_iterations"]
#             inv_total_iterations = 1 / total_iterations
#             target_network.epsilon = iteration * inv_total_iterations * default_eps
#
#             prediction, eps_prediction = target_network.forward(tensor_input, weights=target_weights)
#
#             z_lower = prediction - eps_prediction.T
#             z_upper = prediction + eps_prediction.T
#             z = torch.where((nn.functional.one_hot(gt_output, prediction.size(-1))).bool(), z_lower, z_upper)
#
#             loss_spec = criterion(z, gt_output)
#             loss_fit = criterion(prediction, gt_output)
#             kappa = 1 - (iteration * inv_total_iterations * 0.5)
#
#             loss = kappa * loss_fit + (1 - kappa) * loss_spec
#             target_network.epsilon = default_eps
#         else:
#             prediction = target_network.forward(tensor_input, weights=target_weights)
#             loss = criterion(prediction, gt_output)
#             print(f' loss: {loss.item()}')
#
#         loss.backward()
#         optimizer.step()
#         if parameters["number_of_epochs"] is None:
#             condition = (iteration % 100 == 0) or (iteration == (parameters["number_of_iterations"] - 1))
#         else:
#             condition = (
#                 (iteration % 100 == 0)
#                 or (iteration == (parameters["number_of_iterations"] - 1))
#                 or (((iteration + 1) % no_of_iterations_per_epoch) == 0)
#             )
#
#         if condition:
#             if parameters["number_of_epochs"] is not None:
#                 current_epoch = (iteration + 1) // no_of_iterations_per_epoch
#                 print(f"Current epoch: {current_epoch}")
#             accuracy = calculate_accuracy(
#                 current_dataset_instance,
#                 target_network,
#                 target_weights,
#                 parameters={
#                     "device": parameters["device"],
#                     "use_batch_norm_memory": use_batch_norm_memory,
#                     "number_of_task": current_no_of_task,
#                 },
#                 evaluation_dataset="validation",
#             )
#             print(f"Task {current_no_of_task}, iteration: {iteration + 1},"
#                 f" loss: {loss.item()}, validation accuracy: {accuracy}"
#             )
#             if parameters["best_model_selection_method"] == "val_loss":
#                 if accuracy > best_val_accuracy:
#                     best_val_accuracy = accuracy
#                     best_hypernetwork = deepcopy(hypernetwork)
#                     best_target_network = deepcopy(target_network)
#             if (
#                 parameters["number_of_epochs"] is not None
#                 and parameters["lr_scheduler"]
#                 and (((iteration + 1) % no_of_iterations_per_epoch) == 0)
#             ):
#                 print("Finishing the current epoch")
#                 plateau_scheduler.step(accuracy)
#
#     if parameters["best_model_selection_method"] == "val_loss":
#         return best_hypernetwork, best_target_network
#     else:
#         return hypernetwork, target_network
#
# def build_multiple_task_experiment(dataset_list_of_tasks, parameters, use_chunks=False):
#     output_shape = list(dataset_list_of_tasks[0].get_train_outputs())[0].shape[0]
#     if parameters["target_network"] == "MLP":
#         target_network = MLP(
#             n_in=parameters["input_shape"],
#             n_out=output_shape,
#             hidden_layers=parameters["target_hidden_layers"],
#             use_bias=parameters["use_bias"],
#             no_weights=True,
#         ).to(parameters["device"])
#     elif parameters["target_network"] == "epsMLP":
#         target_network = epsMLP(
#             n_in=parameters["input_shape"],
#             n_out=output_shape,
#             hidden_layers=parameters["target_hidden_layers"],
#             use_bias=parameters["use_bias"],
#             no_weights=True,
#             epsilon=0.01,
#         ).to(parameters["device"])
#     if not use_chunks:
#         hypernetwork = HMLP(
#             target_network.param_shapes,
#             uncond_in_size=0,
#             cond_in_size=parameters["embedding_size"],
#             activation_fn=parameters["activation_function"],
#             layers=parameters["hypernetwork_hidden_layers"],
#             num_cond_embs=parameters["number_of_tasks"],
#         ).to(parameters["device"])
#     else:
#         hypernetwork = ChunkedHMLP(
#             target_shapes=target_network.param_shapes,
#             chunk_size=parameters["chunk_size"],
#             chunk_emb_size=parameters["chunk_emb_size"],
#             cond_in_size=parameters["embedding_size"],
#             activation_fn=parameters["activation_function"],
#             layers=parameters["hypernetwork_hidden_layers"],
#             num_cond_embs=parameters["number_of_tasks"],
#         ).to(parameters["device"])
#
#     criterion = nn.CrossEntropyLoss()
#     dataframe = pd.DataFrame(columns=["after_learning_of_task", "tested_task", "accuracy"])
#
#     use_batch_norm_memory = False
#     hypernetwork.train()
#     for no_of_task in range(parameters["number_of_tasks"]):
#         hypernetwork, target_network = train_single_task(
#             hypernetwork,
#             target_network,
#             criterion,
#             parameters,
#             dataset_list_of_tasks,
#             no_of_task,
#         )
#         # TODO czy wczytac jakas siec?
#         if no_of_task <= (parameters["number_of_tasks"] - 1):
#             write_pickle_file(
#                 f'{parameters["saving_folder"]}/hypernetwork_after_{no_of_task}_task', hypernetwork.weights
#             )
#             # write_pickle_file(
#             #     f'{parameters["saving_folder"]}/target_network_after_{no_of_task}_task',target_network.weights
#             # )
#         dataframe = evaluate_previous_tasks(
#             hypernetwork,
#             target_network,
#             dataframe,
#             dataset_list_of_tasks,
#             parameters={
#                 "device": parameters["device"],
#                 "use_batch_norm_memory": use_batch_norm_memory,
#                 "number_of_task": no_of_task,
#             },
#         )
#         dataframe = dataframe.astype({"after_learning_of_task": "int", "tested_task": "int"})
#         dataframe.to_csv(f'{parameters["saving_folder"]}/results_{parameters["name_suffix"]}.csv', sep=";")
#     return hypernetwork, target_network, dataframe
#
#
# def main_running_experiments(path_to_datasets, parameters):
#     if parameters["dataset"] == "PermutedMNIST":
#         dataset_tasks_list = prepare_permuted_mnist_tasks(
#             path_to_datasets,
#             parameters["input_shape"],
#             parameters["number_of_tasks"],
#             parameters["padding"],
#             parameters["no_of_validation_samples"],
#         )
#     elif parameters["dataset"] == "SplitMNIST":
#         dataset_tasks_list = prepare_split_mnist_tasks(
#             path_to_datasets,
#             validation_size=parameters["no_of_validation_samples"],
#             use_augmentation=parameters["augmentation"],
#             number_of_tasks=parameters["number_of_tasks"],
#         )
#     hypernetwork, target_network, dataframe = build_multiple_task_experiment(
#         dataset_tasks_list, parameters, use_chunks=parameters["use_chunks"]
#     )
#     no_of_last_task = parameters["number_of_tasks"] - 1
#     accuracies = dataframe.loc[dataframe["after_learning_of_task"] == no_of_last_task]["accuracy"].values
#     row_with_results = (
#         f"{dataset_tasks_list[0].get_identifier()};"
#         f'{parameters["augmentation"]};'
#         f'{parameters["embedding_size"]};'
#         f'{parameters["seed"]};'
#         f'{str(parameters["hypernetwork_hidden_layers"]).replace(" ", "")};'
#         f'{parameters["use_chunks"]};{parameters["chunk_emb_size"]};'
#         f'{parameters["target_network"]};'
#         f'{str(parameters["target_hidden_layers"]).replace(" ", "")};'
#         f'{parameters["resnet_number_of_layer_groups"]};'
#         f'{parameters["resnet_widening_factor"]};'
#         f'{parameters["norm_regularizer_masking"]};'
#         f'{parameters["best_model_selection_method"]};'
#         f'{parameters["optimizer"]};'
#         f'{parameters["activation_function"]};'
#         f'{parameters["learning_rate"]};{parameters["batch_size"]};'
#         f'{parameters["beta"]};{parameters["sparsity_parameter"]};'
#         f'{parameters["norm"]};{parameters["lambda"]};'
#         f"{np.mean(accuracies)};{np.std(accuracies)}"
#     )
#     append_row_to_file(
#         f'{parameters["grid_search_folder"]}'
#         f'{parameters["summary_results_filename"]}.csv',
#         row_with_results,
#     )
#
#     load_path = f'{parameters["saving_folder"]}/results_{parameters["name_suffix"]}.csv'
#     plot_heatmap(load_path)
#
#     return hypernetwork, target_network, dataframe
#
#
# if __name__ == "__main__":
#     path_to_datasets = "./Data"
#     dataset = "SplitMNIST"
#     # 'PermutedMNIST', 'CIFAR100', 'SplitMNIST', 'TinyImageNet', 'CIFAR100_FeCAM_setup'
#     part = 1
#     create_grid_search = True
#     if create_grid_search:
#         summary_results_filename = "grid_search_results"
#     else:
#         summary_results_filename = "summary_results"
#     hyperparameters = set_hyperparameters(dataset, grid_search=create_grid_search, part=part)
#     header = (
#         "dataset_name;augmentation;embedding_size;seed;hypernetwork_hidden_layers;"
#         "use_chunks;chunk_emb_size;target_network;target_hidden_layers;"
#         "layer_groups;widening;norm_regularizer_masking;final_model;optimizer;"
#         "hypernet_activation_function;learning_rate;batch_size;beta;"
#         "sparsity;norm;lambda;mean_accuracy;std_accuracy"
#     )
#     append_row_to_file(f'{hyperparameters["saving_folder"]}{summary_results_filename}.csv', header)
#
#     for no, elements in enumerate(
#         product(
#             hyperparameters["embedding_sizes"],
#             hyperparameters["learning_rates"],
#             hyperparameters["betas"],
#             hyperparameters["hypernetworks_hidden_layers"],
#             hyperparameters["sparsity_parameters"],
#             hyperparameters["lambdas"],
#             hyperparameters["batch_sizes"],
#             hyperparameters["norm_regularizer_masking_opts"],
#             hyperparameters["seed"],
#         )
#     ):
#         embedding_size = elements[0]
#         learning_rate = elements[1]
#         beta = elements[2]
#         hypernetwork_hidden_layers = elements[3]
#         sparsity_parameter = elements[4]
#         lambda_par = elements[5]
#         batch_size = elements[6]
#         norm_regularizer_masking = elements[7]
#         seed = elements[8]
#
#         parameters = {
#             "input_shape": hyperparameters["shape"],
#             "augmentation": hyperparameters["augmentation"],
#             "number_of_tasks": hyperparameters["number_of_tasks"],
#             "dataset": dataset,
#             "seed": seed,
#             "hypernetwork_hidden_layers": hypernetwork_hidden_layers,
#             "activation_function": hyperparameters["activation_function"],
#             "norm_regularizer_masking": norm_regularizer_masking,
#             "use_chunks": hyperparameters["use_chunks"],
#             "chunk_size": hyperparameters["chunk_size"],
#             "chunk_emb_size": hyperparameters["chunk_emb_size"],
#             "target_network": hyperparameters["target_network"],
#             "target_hidden_layers": hyperparameters["target_hidden_layers"],
#             "resnet_number_of_layer_groups": hyperparameters["resnet_number_of_layer_groups"],
#             "resnet_widening_factor": hyperparameters["resnet_widening_factor"],
#             "adaptive_sparsity": hyperparameters["adaptive_sparsity"],
#             "sparsity_parameter": sparsity_parameter,
#             "learning_rate": learning_rate,
#             "best_model_selection_method": hyperparameters["best_model_selection_method"],
#             "lr_scheduler": hyperparameters["lr_scheduler"],
#             "batch_size": batch_size,
#             "number_of_epochs": hyperparameters["number_of_epochs"],
#             "number_of_iterations": hyperparameters["number_of_iterations"],
#             "no_of_validation_samples_per_class": hyperparameters["no_of_validation_samples_per_class"],
#             "embedding_size": embedding_size,
#             "norm": hyperparameters["norm"],
#             "lambda": lambda_par,
#             "optimizer": hyperparameters["optimizer"],
#             "beta": beta,
#             "padding": hyperparameters["padding"],
#             "use_bias": hyperparameters["use_bias"],
#             "use_batch_norm": hyperparameters["use_batch_norm"],
#             "device": hyperparameters["device"],
#             "saving_folder": f'{hyperparameters["saving_folder"]}{no}/',
#             "grid_search_folder": hyperparameters["saving_folder"],
#             "name_suffix": f"mask_sparsity_{sparsity_parameter}",
#             "summary_results_filename": summary_results_filename,
#         }
#         if "no_of_validation_samples" in hyperparameters:
#             parameters["no_of_validation_samples"] = hyperparameters["no_of_validation_samples"]
#
#         os.makedirs(parameters["saving_folder"], exist_ok=True)
#         # save_parameters(parameters["saving_folder"], parameters, name=f'parameters_{parameters["name_suffix"]}.csv')
#         if seed is not None:
#             set_seed(seed)
#
#         hypernetwork, target_network, dataframe = main_running_experiments(path_to_datasets, parameters)
