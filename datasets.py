import os
import numpy as np

import torch
from hypnettorch.data.special import permuted_mnist
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers

from DatasetHandlers.TinyImageNet import TinyImageNet
from DatasetHandlers.RotatedMNIST import RotatedMNISTlist
from DatasetHandlers.ImageNetSubset import SubsetImageNet


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

def generate_random_rotations(
    number_of_rotations, seed=None
):
    """
    Prepare a list of random rotations of the selected shape
    for continual learning tasks.

    Arguments:
    ----------
      *number_of_rotations*: int, a number of rotations that will
                                be prepared; it corresponds to the total
                                number of tasks
      *seed*: int, optional argument, default: None
              if one would get deterministic results
    """
    list_of_rotations = []

    if seed is not None:
        np.random.seed(seed)

    for _ in range(number_of_rotations):
        list_of_rotations.append(
            np.random.uniform(0, 360)
        )
    return list_of_rotations


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

def prepare_imagenet_subset_tasks(
    datasets_folder, validation_size, use_augmentation, input_shape,
):
    """
    Prepare a list of five tasks related
    to the ImageNet-Subset dataset according to the FeCAM setup.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which the main folder
                         of the ImageNet-Subset is stored
      *seed*: (int) Necessary for the preparation of random permutation
              of the samples order in batches.
      *validation_size*: (int) defines the total number of validation
                         samples in each task, 0 if the validation set
                         should not be used
      *use_augmentation*: (bool) defines whether dataset augmentation
                          will be applied

    Returns a list of ImageNet-Subset objects.
    """
    handlers = []
    # 5 tasks, 20 classes per each, like in FeCAM MSCIL setting
    for task_no in range(5):
        handlers.append(
            SubsetImageNet(
                data_path=datasets_folder,
                number_of_task=task_no,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_one_hot=True,
                input_shape=input_shape,
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


def prepare_rotated_mnist_tasks(
    datasets_folder, input_shape, number_of_tasks, padding, validation_size
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the RotatedMNIST dataset.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *input_shape*: (int) a number defining shape of the dataset
      *validation_size*: (int) The number of validation samples

    Returns a list of RotatedMNIST objects.
    """
    rotations = generate_random_rotations(input_shape, number_of_tasks)
    return RotatedMNISTlist(
        rotations,
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


def set_hyperparameters(dataset, grid_search=False):
    """
    Set hyperparameters of the experiments, both in the case of grid search
    optimization and a single network run.

    Arguments:
    ----------
      *dataset*: 'PermutedMNIST', 'SplitMNIST', 'CIFAR100', 'TinyImageNet', 'RotatedMNIST'
      *grid_search*: (Boolean optional) defines whether a hyperparameter
                     optimization should be performed or hyperparameters
                     for just a single run have to be returned

    Returns a dictionary with necessary hyperparameters.
    """
    if dataset == "PermutedMNIST":
        if grid_search:
            # hyperparams = {
            #     "embedding_sizes": [24, 48, 96],
            #     "learning_rates": [0.001],
            #     "batch_sizes": [128,64],
            #     "betas": [0.001, 0.0005, 0.005],
            #     "mixup_alphas": [0.1, 0.2, 0.3, None],
            #     "perturbation_epsilons": [2/255.0, 20/255.0, 25/250.0],
            #     "hypernetworks_hidden_layers": [[100, 50]],
            #     "best_model_selection_method": "val_loss",
            #     "saving_folder": "./Results/grid_search/permuted_mnist/",
            #     # not for optimization, just for multiple cases
            #     "seed": [1],
            # }
            hyperparams = {
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.005],
                "perturbation_epsilons": [2/255],
                "hypernetworks_hidden_layers": [[100, 100]],
                "mixup_alphas": [0.1, 0.2, 0.3, None],
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/grid_search/permuted_mnist/",
                # not for optimization, just for multiple cases
                "seed": [1],
            }

        else:
            # Best hyperparameters without Mixup
            hyperparams = {
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.005],
                "perturbation_epsilons": [0.01],
                "hypernetworks_hidden_layers": [[100, 100]],
                "mixup_alphas": [None],
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/grid_search/permuted_mnist/",
                # not for optimization, just for multiple cases
                "seed": [1],
            }

        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = 20
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 5000
        hyperparams["no_of_validation_samples_per_class"] = 500
        hyperparams["target_hidden_layers"] = [256, 256]
        hyperparams["target_network"] = "IntervalMLP"
        hyperparams["optimizer"] = "adam"
        hyperparams["use_batch_norm"] = False
        # Directly related to the MNIST dataset
        hyperparams["padding"] = 2
        hyperparams["shape"] = (28 + 2 * hyperparams["padding"]) ** 2
        hyperparams["number_of_tasks"] = 10
        hyperparams["augmentation"] = False
        hyperparams["use_batch_norm_memory"] = False

    elif dataset == "CIFAR100":
        if grid_search:
            hyperparams = {
                "seed": [5],
                "embedding_sizes": [48],
                "betas": [0.01, 0.05, 0.1, 1],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "hypernetworks_hidden_layers": [[100]],
                "mixup_alphas": [0.1, 0.2, 0.3, None],
                "perturbation_epsilons": [0.01],
                "optimizer": "adam",
                "use_batch_norm": False,
                "target_network": "AlexNet",
                "number_of_epochs": 200,
                "augmentation": False,
                "saving_folder": f"./Results/grid_search/CIFAR_100/"}
        else:
            hyperparams = {
                "seed": [42],
                "embedding_sizes": [512],
                "betas": [0.01],
                "batch_sizes": [32],
                "learning_rates": [0.001],
                "hypernetworks_hidden_layers": [[100,50]],
                "mixup_alphas": [None],
                "perturbation_epsilons": [0.005],
                "use_batch_norm": True,
                "number_of_epochs": 200,
                "target_network": "AlexNet",
                "optimizer": "adam",
                "augmentation": False,
                "saving_folder": f"./Results/CIFAR_100_best_hyperparams/"
                # Best for reduced architecture
                # "seed": [42],
                # "embedding_sizes": [512],
                # "betas": [0.01],
                # "batch_sizes": [32],
                # "learning_rates": [0.001],
                # "hypernetworks_hidden_layers": [[200, 50]],
                # "perturbation_epsilons": [4/255],
                # "use_batch_norm": True,
                # "number_of_epochs": 200,
                # "target_network": "AlexNet",
                # "optimizer": "adam",
                # "augmentation": False,
                # "saving_folder": f"./Results/CIFAR_100_best_hyperparams/"
                }

        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 500
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "AlexNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 10
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == "TinyImageNet":
        if grid_search:
            hyperparams = {
                "seed": [42],
                "embedding_sizes": [48, 96, 128],
                "betas": [0.01, 0.05, 0.1],
                "learning_rates": [0.001],
                "batch_sizes": [16,32],
                "hypernetworks_hidden_layers": [[100],[200]],
                "perturbation_epsilons": [0.01, 0.05, 0.1, 0.5],
                "mixup_alphas": [None],
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "number_of_epochs": 10,
                "augmentation": True,
                "saving_folder": f"./Results/TinyImageNet/",
            }
        else:
            # ResNet
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [48],
                "betas": [1],
                "batch_sizes": [16],
                "learning_rates": [0.001],
                "hypernetworks_hidden_layers": [[100]],
                "mixup_alphas": [None],
                 "perturbation_epsilons": [0.0],
                "use_batch_norm": True,
                "number_of_epochs": 10,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": False,
                "saving_folder": "./Results/TinyImageNet/ResNet_best_hyperparams/",
            }
           
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
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == "SplitMNIST":
        if grid_search:
            hyperparams = {
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.001],
                "hypernetworks_hidden_layers": [[25, 25]],
                "perturbation_epsilons": [0.01],
                "mixup_alphas": [0.1, 0.2, 0.3, None],
                "seed": [1, 2, 3, 4, 5],
                "best_model_selection_method": "val_loss",
                "embedding_sizes": [64],
                "augmentation": True,
                "use_batch_norm_memory": False,
                "saving_folder": "./Results/grid_search/split_mnist/"}
        else:
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [72],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01],
                "perturbation_epsilons": [0.01],
                "mixup_alphas": [0.1],
                "hypernetworks_hidden_layers": [[75, 75]],
                "augmentation": True,
                "use_batch_norm_memory": False,
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/split_mnist_test/",
            }

        hyperparams["lr_scheduler"] = False
        hyperparams["target_network"] = "IntervalMLP"
        hyperparams["optimizer"] = "adam"
        hyperparams["number_of_iterations"] = 2000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["no_of_validation_samples_per_class"] = 200
        hyperparams["target_hidden_layers"] = [400, 400]
        hyperparams["shape"] = 28**2
        hyperparams["number_of_tasks"] = 5
        hyperparams["use_batch_norm"] = False
        hyperparams["padding"] = None

    elif dataset == "RotatedMNIST":
        if grid_search:
            hyperparams = {
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.001, 0.0005, 0.005],
                "mixup_alphas": [0.1, 0.2, 0.3, None],
                "perturbation_epsilons": [0.01],
                "hypernetworks_hidden_layers": [[100, 100]],
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/grid_search/rotated_mnist/",
                "seed": [1, 2, 3, 4, 5],
            }

        else:
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [96],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.0005],
                "perturbation_epsilons": [0.01],
                "mixup_alphas": [0.1],
                "hypernetworks_hidden_layers": [[100, 100]],
                "best_model_selection_method": "last_model",
                "saving_folder": "./Results/rotated_mnist_best_hyperparams/",
            }

        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = 5000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 5000
        hyperparams["no_of_validation_samples_per_class"] = 500
        hyperparams["target_hidden_layers"] = [256, 256]
        hyperparams["target_network"] = "IntervalMLP"
        hyperparams["optimizer"] = "adam"
        hyperparams["use_batch_norm"] = False
        hyperparams["padding"] = 2
        hyperparams["shape"] = (28 + 2 * hyperparams["padding"]) ** 2
        hyperparams["number_of_tasks"] = 10
        hyperparams["augmentation"] = False
        hyperparams["use_batch_norm"] = False
        
    elif dataset == "ImageNetSubset":
        if grid_search:
            hyperparams = {
                "seed": [42],
                "embedding_sizes": [128,256,512],
                "betas": [0.01, 0.05, 0.1],
                "batch_sizes": [32],
                "learning_rates": [0.001],
                "hypernetworks_hidden_layers": [[100, 50], [200, 50]],
                "mixup_alphas": [0.1, 0.2, 0.3, None],
                "use_batch_norm": True,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
                "shape": 64,
                "target_hidden_layers": None,
                "saving_folder": f"./Results/ImageNetSubset/",
                "perturbation_epsilons": [2/255.0, 1/255.0],
            }
            
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["no_of_validation_samples_per_class"] = 200
        hyperparams["number_of_tasks"] = 5
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"
    else:
        raise ValueError("This dataset is not implemented!")

    # General hyperparameters
    hyperparams["activation_function"] = torch.nn.ReLU()
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
