"""
Implementation of RotatedMNIST for continual learning tasks.
"""

import torch
import copy
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
from hypnettorch.data.mnist_data import MNISTData


class RotatedMNISTlist():
    def __init__(
            self,
            rotations,
            data_path,
            use_not_hot=True,
            validation_size=0,
            padding=0,
            trgt_padding=None,
            show_rot_change_msg=True
        ):

        self._data = RotatedMNIST(
            data_path,
            use_not_hot=use_not_hot,
            validation_size=validation_size,
            rotation=None,
            padding=padding,
            trgt_padding=trgt_padding
        )

        self._rotations = rotations

        self._show_perm_change_msg = show_rot_change_msg

        self._batch_gens_train = [None] * len(permutations)
        self._batch_gens_test = [None] * len(permutations)
        self._batch_gens_val = [None] * len(permutations)

    def __len__(self):
        return len(self._rotations)

    def __getitem__(self, index):
        """Not implemented."""
        raise NotImplementedError('Not yet implemented!')



