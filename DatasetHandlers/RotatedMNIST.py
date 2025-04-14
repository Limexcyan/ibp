"""
Implementation of RotatedMNIST for continual learning tasks.
"""

import torch
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image
import random


class RotatedMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, angle=0, download=True):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        self.angle = angle  # could be a single int or list for more flexible cases

    def __getitem__(self, index):
        img, label = self.mnist[index]

        # Convert to PIL if not already
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Rotate image
        rotated_img = F.rotate(img, self.angle)

        return transforms.ToTensor()(rotated_img), label

    def __len__(self):
        return len(self.mnist)
