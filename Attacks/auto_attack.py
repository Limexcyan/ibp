import torch
import torch.nn as nn

class AutoAttack:
    r"""
    AutoAttack implements an ensemble of strong adversarial attacks:
    - APGD with CrossEntropy loss (APGD-CE)
    - APGD with DLR loss (APGD-DLR)
    - FAB Attack (simplified)
    - Square Attack (simplified)

    Distance Measure : Linf or L2

    Arguments:
        model (nn.Module): Model to attack.
        target_weights (list): Weights generated by a hypernetwork.
        eps (float): Maximum perturbation. (Default: 8/255)
        norm (str): Norm to constrain perturbations ('Linf' or 'L2'). (Default: 'Linf')
        n_iter (int): Number of iterations per sub-attack. (Default: 100)
        device (str): Device to use. (Default: 'cuda')

    Shape:
        - images: (N, C, H, W) or (N, D)
        - labels: (N)

    Examples::
        >>> attack = AutoAttack(model, target_weights, eps=8/255, device="cuda")
        >>> adv_images = attack(images, labels, task_id)
    """

    def __init__(self, model, target_weights, eps=8/255, norm='Linf', n_iter=100, device="cuda"):
        self.model = model
        self.target_weights = target_weights
        self.eps = eps
        self.norm = norm
        self.n_iter = n_iter
        self.device = device

    def forward(self, images, labels, task_id=None):
        """
        Run the full AutoAttack ensemble on input images.

        Args:
            images (torch.Tensor): Clean input images.
            labels (torch.Tensor): Ground truth labels.
            task_id (int, optional): Task identifier if applicable.

        Returns:
            torch.Tensor: Adversarially perturbed images.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Run each sub-attack
        adv_apgd_ce = self.apgd(images, labels, task_id, loss_fn="ce")
        adv_apgd_dlr = self.apgd(images, labels, task_id, loss_fn="dlr")
        adv_fab = self.fab(images, labels, task_id)
        adv_square = self.square(images, labels, task_id)

        # Choose best adversarial examples
        logits_orig, _ = self.model(images, epsilon=0.0, weights=self.target_weights, condition=task_id)
        logits_ce, _ = self.model(adv_apgd_ce, epsilon=0.0, weights=self.target_weights, condition=task_id)
        logits_dlr, _ = self.model(adv_apgd_dlr, epsilon=0.0, weights=self.target_weights, condition=task_id)
        logits_fab, _ = self.model(adv_fab, epsilon=0.0, weights=self.target_weights, condition=task_id)
        logits_square, _ = self.model(adv_square, epsilon=0.0, weights=self.target_weights, condition=task_id)

        pred_orig = logits_orig.argmax(1)
        best_adv = adv_apgd_ce.clone()

        for i in range(images.size(0)):
            # Pick the attack that succeeded
            if pred_orig[i] == logits_ce.argmax(1)[i] and pred_orig[i] != logits_dlr.argmax(1)[i]:
                best_adv[i] = adv_apgd_dlr[i]
            if pred_orig[i] == logits_ce.argmax(1)[i] and pred_orig[i] != logits_fab.argmax(1)[i]:
                best_adv[i] = adv_fab[i]
            if pred_orig[i] == logits_ce.argmax(1)[i] and pred_orig[i] != logits_square.argmax(1)[i]:
                best_adv[i] = adv_square[i]

        return best_adv

    def apgd(self, images, labels, task_id=None, loss_fn="ce"):
        """
        Perform Auto-PGD attack with specified loss.

        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): True labels.
            task_id (int, optional): Task ID.
            loss_fn (str): "ce" for CrossEntropy or "dlr" for DLR loss.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        images = images.clone().detach()
        adv_images = images.clone().detach()
        adv_images.requires_grad = True

        loss = nn.CrossEntropyLoss() if loss_fn == "ce" else self.dlr_loss

        for _ in range(self.n_iter):
            outputs, _ = self.model(adv_images, epsilon=0.0, weights=self.target_weights, condition=task_id)
            loss_value = loss(outputs, labels)

            grad = torch.autograd.grad(loss_value, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            if self.norm == 'Linf':
                adv_images = adv_images + self.eps * grad.sign()
                adv_images = torch.max(torch.min(adv_images, images + self.eps), images - self.eps)
                adv_images = torch.clamp(adv_images, 0, 1)
            elif self.norm == 'L2':
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1)
                grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                normalized_grad = grad / (grad_norm + 1e-10)
                adv_images = adv_images + self.eps * normalized_grad
                delta = adv_images - images
                delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1)
                delta = delta * (self.eps / (delta_norm + 1e-10)).view(-1, *([1] * (delta.dim() - 1)))
                adv_images = torch.clamp(images + delta, 0, 1)

            adv_images = adv_images.clone().detach()
            adv_images.requires_grad = True

        return adv_images.detach()

    def fab(self, images, labels, task_id=None):
        """
        Perform a simplified FAB attack.

        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): True labels.
            task_id (int, optional): Task ID.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        images = images.clone().detach()
        adv_images = images.clone().detach()
        adv_images.requires_grad = True

        for _ in range(int(self.n_iter // 5)):
            outputs, _ = self.model(adv_images, epsilon=0.0, weights=self.target_weights, condition=task_id)
            preds = outputs.argmax(1)

            loss = (outputs.gather(1, preds.view(-1,1)).squeeze() -
                    outputs.gather(1, labels.view(-1,1)).squeeze()).mean()

            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images - 0.5 * grad.sign()
            adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach()

    def square(self, images, labels, task_id=None):
        """
        Perform a simplified Square attack (random perturbation).

        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): True labels.
            task_id (int, optional): Task ID.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        images = images.clone().detach()
        adv_images = images.clone()

        for _ in range(int(self.n_iter // 10)):
            delta = torch.rand_like(adv_images, device=self.device) * 2 * self.eps - self.eps
            adv_images = adv_images + delta
            adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach()

    def dlr_loss(self, outputs, labels):
        """
        Compute the DLR (Difference of Logits Ratio) loss.

        Args:
            outputs (torch.Tensor): Logits from the model.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Loss value.
        """
        sorted_logits, _ = outputs.sort(dim=1, descending=True)
        correct_logit = outputs.gather(1, labels.unsqueeze(1)).squeeze()
        second_best_logit = sorted_logits[:, 1]
        best_logit = sorted_logits[:, 0]

        if outputs.size(1) >= 3:
            third_best_logit = sorted_logits[:, 2]
        else:
            third_best_logit = sorted_logits[:, 1]

        loss = -(correct_logit - second_best_logit) / (best_logit - third_best_logit + 1e-12)
        return loss.mean()


    def __call__(self, images, labels, task_id=None):
        """
        Allow the attack to be called like a function.

        Args:
            images (torch.Tensor): Clean input images.
            labels (torch.Tensor): Ground truth labels.
            task_id (int, optional): Task identifier.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        return self.attack(images, labels, task_id)
