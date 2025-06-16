import torch.nn as nn

from autoattack import AutoAttack

class AttackModelWrapper(nn.Module):
    def __init__(self, model, weights, device):
        super(AttackModelWrapper, self).__init__()
        self.model = model
        self.weights = weights
        self.device = device

    def forward(self, x):
        logits, _ = self.model(x, epsilon=0.0, weights=self.weights, condition=None)
        return logits

    
class AutoAttackWrapper(AutoAttack):
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
