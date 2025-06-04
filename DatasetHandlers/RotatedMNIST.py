from torch import stack
from torchvision import transforms
from hypnettorch.data.mnist_data import MNISTData


class RotatedMNISTlist:
    def __init__(
            self,
            rotations,
            data_path,
            use_one_hot=True,
            validation_size=0,
            padding=0
    ):
        self._rotations = rotations
        self._data_path = data_path
        self._use_one_hot = use_one_hot
        self._validation_size = validation_size
        self._padding = padding

    def __len__(self):
        return len(self._rotations)

    def __getitem__(self, index):
        rotation = self._rotations[index]
        return RotatedMNIST(
            data_path=self._data_path,
            use_one_hot=self._use_one_hot,
            validation_size=self._validation_size,
            rotation=rotation,
            padding=self._padding
        )


class RotatedMNIST(MNISTData):
    def __init__(
            self,
            data_path,
            rotation,
            use_one_hot=True,
            validation_size=0,
            padding=0
    ):
        super().__init__(
            data_path,
            use_one_hot=use_one_hot,
            validation_size=validation_size,
            use_torch_augmentation=False
        )
        self._padding = padding
        self._rotation = rotation
        self._input_dim = (28 + padding * 2) ** 2
        self._transform = self.torch_input_transforms(padding=self._padding, rotation=self._rotation)

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self._transform = self.torch_input_transforms(padding=self._padding, rotation=value)

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
            mode="inference",
            force_no_preprocessing=False,
            sample_ids=None
    ):
        if not force_no_preprocessing:
            assert len(x.shape) == 2  # [batch_size, 784]
            img_size = 28 + 2 * self._padding
            x = (x * 255.0).astype('uint8')
            x = x.reshape(-1, 28, 28, 1)  # [B, H, W, C]

            # Apply transforms per image and stack
            transformed = stack([
                self._transform(x[i].squeeze()).to(device)  # shape: [1, H, W]
                for i in range(x.shape[0])
            ])
            transformed = transformed.view(-1, img_size ** 2)
            return transformed
        else:
            return super().input_to_torch_tensor(
                x, device, mode=mode,
                force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids
            )

    @staticmethod
    def torch_input_transforms(padding=0, rotation=None):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Pad(padding),
        ]
        if rotation is not None:
            transform_list.append(transforms.RandomRotation(degrees=(rotation, rotation), fill=0))
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)
