import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy


# TODO: Jeśli chodzi o ataki, to są one wszystkie zaimplementowane tutaj: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/index.html
# Myślę, że warto skorzystać sobie z tej paczki, żeby nie kodzić ręcznie.

def fgsm_attack(image, data_grad, ksi=20/255):
    """
    Fast Gradient Sign Method
    Args:
        image: input image dataset
        ksi: attack strenght

    Returns:
        adv_image: image dataset after fgsm adversarial attack

    """
    sign_data_grad = data_grad.sign()
    adv_image = image + ksi * sign_data_grad

    # TODO Tutaj wszystko zależy od tego, jak są znormalizowane dane. Jeżeli są z przedziału [0,1], to jest
    # wszystko ok, a jeśli nie, to będzie źle. Gdy dane były dodatkowo standaryzowane, to można skorzystać
    # z implementacji stąd: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html, w szczególności chodzi mi
    # o funkcję denorm. Wtedy trzeba by też podzielić ksi przez odchylenie standardowe, żebyśmy na pewno porównywali
    # te same wielkości.
    adv_image = torch.clamp(adv_image, min=0, max=1)

    return adv_image

def bim_attack(image, num_iter, ksi=40/255, alpha=2/255):
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

            # TODO Znów, to klipowanie tutaj mocno zależy od tego, jakie dane
            # obsługujemy. Jeśli są znormalizowane do [0,1], to w porządku, ale jeśli
            # nie, to ten atak będzie wykonany nieprawidłowo.
            adv_image = torch.clamp(image + perturbation, min=0, max=1)


    return adv_image

def pgd_attack(image, num_iter, ksi=40/255, alpha=2/255, random_start=True):
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
        perturbation = torch.clamp(adv_image - image, min=-ksi, max=ksi)
        adv_image = torch.clamp(image + perturbation, min=0, max=1)

    return adv_image

def AutoAttack():
    return 0

