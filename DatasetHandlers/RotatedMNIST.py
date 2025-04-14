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

class RotatedMNIST(MNISTData):
    def __init__(
            self,
            data_path,
            use_one_hot=True,
            validation_size=0,
            rotation=None,
            padding=0,
            trgt_padding=None):
        super().__init__(data_path,
                         use_one_hot=use_one_hot,
                         validation_size=validation_size,
                         use_torch_augmentation=False)

        self._padding = padding
        self._input_dim = (28 + padding*2)**2
        self._rotation = rotation

        if trgt_padding is not None and trgt_padding > 0:
            self._data['num_classes'] += trgt_padding
            if self.is_one_hot:
                self._data['out_shape'] = [self._data['out_shape'][0] + trgt_padding]
                out_data = self._data['out_shape']
                self._data['out_data'] = np.concatenate((out_data,
                                                         np.zeros((out_data.shape[0], trgt_padding))), axis=1)

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self._transform = RotatedMNIST.torch_input_transforms(padding=self._padding, rotatio=value)

    @property
    def torch_in_shape(self):
        return [
            self.in_shape[0] + 2 * self._padding,
            self.in_shape[1] + 2 * self._padding,
            self.in_shape[2]
        ]

    def get_inentifier(self):
        return 'RotatedMNIST'

    def input_to_torch_tensot(self,
                              x,
                              device,
                              mode="interference",
                              force_no_preprocessing=False,
                              sample_ids=None):
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

    def torch_input_transforms(padding=0, rotatio=None):
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
