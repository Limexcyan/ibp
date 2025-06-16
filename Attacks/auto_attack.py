import torch.nn as nn

from autoattack import AutoAttack

class AttackModelWrapper(nn.Module):
    """
    A wrapper class for neural network models to facilitate adversarial attacks.

    This class wraps a given model and its associated weights, providing a forward method
    that computes the model's logits with specific parameters suitable for attack scenarios.

    Args:
        model (nn.Module): The neural network model to be wrapped.
        weights (Any): The weights or parameters to be used by the model during the forward pass.
        device (torch.device or str): The device on which computations will be performed.

    Methods:
        forward(x):
            Performs a forward pass through the wrapped model with epsilon set to 0.0,
            using the provided weights and no additional condition. Returns the logits
            output by the model.

    Example:
        wrapper = AttackModelWrapper(model, weights, device)
        logits = wrapper(input_tensor)
    """
    def __init__(self, model, weights, device):
        super(AttackModelWrapper, self).__init__()
        self.model = model
        self.weights = weights
        self.device = device

    def forward(self, x):
        logits, _ = self.model(x, epsilon=0.0, weights=self.weights, condition=None)
        return logits

    
class AutoAttackWrapper(AutoAttack):
    """
    A wrapper class for the AutoAttack adversarial attack suite, providing a unified interface
    for evaluating model robustness across different datasets and input shapes.
    Args:
        model (torch.nn.Module): The neural network model to attack.
        weights (str or dict): Path to the model weights or a state dict.
        eps (float): Maximum perturbation allowed for the attack.
        dataset (str): Name of the dataset (e.g., "PermutedMNIST", "CIFAR100").
        device (torch.device or str): Device to run the attack on.
        norm (str, optional): Norm to use for the attack ('Linf', 'L2', etc.). Default is 'Linf'.
        version (str, optional): Version of AutoAttack to use. Default is 'custom'.
    Attributes:
        model_wrapper (AttackModelWrapper): Wrapped model for attack compatibility.
        input_shape (tuple): Expected input shape for the dataset.
    Methods:
        forward(images, labels, task_id):
            Runs the standard AutoAttack evaluation on the provided images and labels.
            Args:
                images (torch.Tensor): Input images to attack.
                labels (torch.Tensor): True labels for the images.
                task_id (int): Task identifier (not used in attack).
            Returns:
                dict: Results of the adversarial evaluation.
    """
    def __init__(self, model, weights, eps, dataset, device, norm='Linf', version='custom'):
        self.model_wrapper = AttackModelWrapper(model, weights, device)

        super(AutoAttackWrapper, self).__init__(
            model=self.model_wrapper,
            norm=norm,
            eps=eps,
            version=version,
            verbose=True,
            device=device,
            attacks_to_run=["apgd-ce", "apgd-t", "fab", "square"]
        )
        self.model_wrapper.to(device)

        if dataset in ["PermutedMNIST", "RotatedMNIST"]:
            self.input_shape = (1, 32, 32) # With padding
        elif dataset == "CIFAR100":
            self.input_shape = (3, 32, 32)
        elif dataset in ["TinyImageNet", "ImageNetSubset"]:
            self.input_shape = (3, 64, 64)
        

    def forward(self, images, labels, task_id):
        images = images.view(images.shape[0], *self.input_shape)
        return self.run_standard_evaluation(images, labels)
