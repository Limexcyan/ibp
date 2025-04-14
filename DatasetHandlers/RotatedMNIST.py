"""
Implementation of RotatedMNIST for continual learning tasks.
"""

import torch
from torch import stack
import copy
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
from hypnettorch.data.mnist_data import MNISTData


class RotatedMNISTlist:
    def __init__(
            self,
            rotations,
            data_path,
            use_not_hot=True,
            validation_size=0,
            padding=0
        ):

        self._data = RotatedMNIST(
            data_path,
            use_one_hot=use_one_hot,
            validation_size=validation_size,
            rotation=None,
            padding=padding
        )

        self._rotations = rotations

        self._transform = None

        self._batch_gens_train = [None] * len(rotations)
        self._batch_gens_test = [None] * len(rotations)
        self._batch_gens_val = [None] * len(rotations)

    def __len__(self):
        return len(self._rotations)

    def __getitem__(self, index):
        raise NotImplementedError('Not yet implemented!')

class RotatedMNIST(MNISTData):
    def __init__(
            self,
            data_path,
            use_one_hot=True,
            validation_size=0,
            rotation=None,
            padding=0
    ):

        super().__init__(
            data_path,
            use_one_hot=use_one_hot,
            validation_size=validation_size,
            use_torch_augmentation=False
        )

        self._padding = padding
        self._input_dim = (28 + padding*2)**2
        self._rotation = rotation

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self._transform = RotatedMNIST.torch_input_transforms(padding=self._padding, rotation=value)

    @property
    def torch_in_shape(self):
        return [
            self.in_shape[0] + 2 * self._padding,
            self.in_shape[1] + 2 * self._padding,
            self.in_shape[2]
        ]

    def get_identifier(self):
        return 'RotatedMNIST'

    def input_to_torch_tensor(
            self,
            x,
            device,
            mode="interference",
            force_no_preprocessing=False,
            sample_ids=None
    ):
        if not force_no_preprocessing:
            assert len(x.shape) == 2

            img_size = 28 + 2 * self._padding
            x = (x * 255.0).astype('uint8')
            x = x.reshape(-1, 28, 28, 1)
            x = stack([self._transform(x[i, ...]) for i in range(x.shape[0]).to(device)])
            x = x.permutate(0, 2, 3, 1)
            x = x.contiguous().view(-1, img_size ** 2)

            return x
        else:
            return MNISTData.input_to_torch_tensor(
                self,
                x,
                device,
                mode=mode,
                force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids
            )

    def torch_input_transforms(
            self,
            padding=0,
            rotation=None
    ):
        transform_list = [
            transforms.ToPILImage('L'),
            transforms.Pad(padding),
        ]

        if rotation is not None:
            transform_list.append(transforms.RandomRotation(degrees=(rotation, rotation), fill=0))

        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    if __name__ == '__main__':
        pass
