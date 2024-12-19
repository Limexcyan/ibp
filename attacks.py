import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy


def fgsm_attack(image, ksi=20/255):
    """
    Fast Gradient Sign Method
    Args:
        image: input image dataset
        ksi: attack strenght

    Returns:
        adv_image: image dataset after fgsm adversarial attack

    """
    adv_image = image + ksi * image.sign()
    adv_image = torch.clamp(adv_image, min=0, max=1)

    return adv_image

def bim_attack(image, label, num_iter, ksi=40/255, alpha=2/255):
    """
    Basic Iterative Method
    Args:
        image: input image dataset
        num_iter: number of iterations
        ksi: attack strenght
        alpha: step size

    Returns:
        adv_image: image dataset after fgsm adversarial attack

    """
    adv_image = image.clone().detach()

    for it in range(num_iter):
        with torch.no_grad():
            adv_image = adv_image + alpha * adv_image.grad.sign()

            perturbation = torch.clamp(adv_image - image, min=-ksi, max=ksi)
            adv_image = torch.clamp(image + perturbation, min=0, max=1)


    return adv_image

def pgd_attack(image, label, num_iter, ksi=40/255, alpha=2/255, random_start=True):
    """

    Args:
        image: input image dataset
        num_iter: number of iterations
        ksi: attack strenght
        alpha: step size
        random_start: allow random start

    Returns:
        adv_image: image dataset after fgsm adversarial attack
    """
    adv_image = image.clone().detach()
    if random_start:
        adv_image = image + (torch.empty_like(image).uniform_(-ksi, ksi))

    for it in range(num_iter):
        adv_image = adv_image + alpha * adv_image.grad.sign()
        criterion = cross_entropy()
        loss = criterion()
        perturbation = torch.clamp(adv_image - image, min=-ksi, max=ksi)
        adv_image = torch.clamp(image + perturbation, min=0, max=1)


    return adv_image

