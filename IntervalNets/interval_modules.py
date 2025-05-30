import torch
import torch.nn.functional as F

from typing import Tuple

class IntervalLinear:
    """
    Interval version of a linear layer. Parameters are generated by a hypernetwork,
    so they are not initialized here.

    For description of the attributes please see the docs of
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(self, in_features: int, out_features: int, device=None) -> None:
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

    def forward(self, mu: torch.Tensor, 
                    eps: torch.Tensor,
                    weight: torch.Tensor,
                    bias: torch.Tensor = None,
                    device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies interval version of a linear transformation.

        Parameters:
        ----------
            mu: torch.Tensor
                Midpoint of the interval.

            eps: torch.Tensor
                Radii of the interval.

            weight: torch.Tensor
                Matrix of  a linear transformation.
            
            bias: torch.Tensor
                Bias of a linar tranformation. Can be None.
            
            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after the linear transformation.
            
            new_eps: torch.Tensor
                'eps' after the linear transformation.
        """

        # Send tensors into devices
        mu     = mu.to(device)
        eps    = eps.to(device)
        weight = weight.to(device)
        bias   = bias.to(device) if bias is not None else bias
        
        # Perform linear transformations
        new_mu = F.linear(
            input=mu,
            weight=weight,
            bias=bias
        )

        new_eps = F.linear(
            input=eps,
            weight=weight.abs(),
            bias=None
        )

        return new_mu, new_eps
    
class IntervalReLU:
    """
    Interval version of ReLU activation function

    For description of the attributes please see the docs of
    https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """

    def __init__(self, inplace: bool = False):
       
       self.inplace = inplace

    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor,
                device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies interval version of a ReLU transformation.

        Parameters:
        ----------
            mu: torch.Tensor
                Midpoint of the interval. 

            eps: torch.Tensor
                Radii of the interval.

            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after ReLU transformation.
            
            new_eps: torch.Tensor
                'eps' after ReLU transformation.
        """

        # Send tensors into devices
        mu  = mu.to(device)
        eps = eps.to(device)

        z_l, z_u = mu - eps, mu + eps
        z_l, z_u = F.relu(z_l), F.relu(z_u)

        new_mu, new_eps  = (z_u + z_l) / 2, (z_u - z_l) / 2

        return new_mu, new_eps
    
class IntervalConv2d:
    """
    For description of the attributes please see the docs of
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 dilation = 1, groups = 1, padding_mode = 'zeros', device="cuda") -> None:

        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        
    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies interval version of a convolutional transformation.

        Parameters:
        ----------
            mu: torch.Tensor
                Midpoint of the interval. 

            eps: torch.Tensor
                Radii of the interval.

            weight: torch.Tensor
                Kernels of a convolutional transformation.
            
            bias: torch.Tensor
                Bias of a convolutional tranformation. Can be None.
            
            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after the convolutional transformation.
            
            new_eps: torch.Tensor
                'eps' after the convolutional transformation.
        """

        # Send tensors into devices
        mu     = mu.to(device)
        eps    = eps.to(device)
        weight = weight.to(device)
        bias   = bias.to(device) if bias is not None else bias
        
        # Perform convolutional transformations
        new_mu = F.conv2d(
              input=mu,
              weight=weight,
              bias=bias,
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )

        new_eps = F.conv2d(
              input=eps,
              weight=weight.abs(),
              bias=None,
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
        
        return new_mu, new_eps
    
class IntervalFlatten:

    """
    Interval flattening.

    For the description of arguments please see docs: https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, mu: torch.Tensor, eps: torch.Tensor,
                device: str = "cuda") -> Tuple[torch.Tensor,torch.Tensor]:
        
        """
        Applies interval flattening.

        Parameters:
        ----------
            mu: torch.Tensor
                Midpoint of the interval. 

            eps: torch.Tensor
                Radii of the interval.

            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda". It is used just for
                convenience to simplify a forward method of NNs.
        
        Returns:
        --------
            new_mu: torch.Tensor
                Flattened 'mu'.
            
            new_eps: torch.Tensor
                Flattened 'eps'.

        """
        mu  = mu.flatten(self.start_dim, self.end_dim).to(device)
        eps = eps.flatten(self.start_dim, self.end_dim).to(device)

        return mu, eps


class IntervalBatchNorm:
    def __init__(self) -> None:
        pass

    def forward(self, mu, eps, weight, bias, running_mean, running_var, stats_id, 
                batch_norm_forward, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies interval version of a batch normalization layer. The implmentation needs to
        be compatible with hypernetworks.

        Parameters:
        ----------
            mu: torch.Tensor
                Midpoint of the interval. 

            eps: torch.Tensor
                Radii of the interval.

            weight: torch.Tensor
                Learnable weight of the affine transformation.

            bias: torch.Tensor
                Learnable bias of the affine transformation.

            running_mean: torch.Tensor
                Running means calculated over batches of data.

            running_var: torch.Tensor
                Running variances calculated over batches of data.

            stats_id: int
                Identifier of statistics to be used.

            batch_norm_forward: Callable
                Forward method of 'hypnettorch.utils.batchnorm_layer.BatchNormLayer'.
            
            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after the BatchNorm transformation.
            
            new_eps: torch.Tensor
                'eps' after the BatchNorm transformation.
        """
        mu = mu.to(device)
        eps = eps.to(device)
        weight = weight.to(device)
        bias = bias.to(device)

        new_mu = batch_norm_forward(
            mu,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            stats_id=stats_id
        )

        new_eps = F.batch_norm(
            input=eps,
            running_mean=torch.zeros(eps.shape[1], device=device),
            running_var=running_var,
            weight=weight.abs(),
            bias=torch.zeros_like(bias, device=device),
            training=True
        )

        return new_mu, new_eps
    
class IntervalAvgPool2d:
    """
    Interval version of AvgPool2d.
    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, mu, eps, device="cuda") -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Applies interval average-pooling.

        Parameters:
        -----------
            mu: torch.Tensor
                Midpoint of the interval.
            eps: torch.Tensor
                Radius of the interval.
            device: str
                Device for computation ("cpu" or "cuda").

        Returns:
        --------
            new_mu: torch.Tensor
                Pooled midpoint.
            new_eps: torch.Tensor
                Pooled radius.
        """
        mu = mu.to(device)
        eps = eps.to(device)

        z_lower, z_upper = mu - eps, mu + eps
        z_lower = F.avg_pool2d(
            z_lower,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        z_upper = F.avg_pool2d(
            z_upper,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        new_mu, new_eps = (z_upper+z_lower)/2.0, (z_upper-z_lower)/2.0

        return new_mu, new_eps

    
class IntervalDropout:
    """
    Interval version of Dropout.
    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html
    """
    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def forward(self, mu, eps, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies interval dropout.

        Parameters:
        ----------
            mu: torch.Tensor
                Midpoint of the interval.
            eps: torch.Tensor
                Radius of the interval.
            device: str
                Device for computation ("cpu" or "cuda").

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after dropout.
            new_eps: torch.Tensor
                'eps' after dropout.
        """
        mu = mu.to(device)
        eps = eps.to(device)
        z_lower, z_upper = mu - eps, mu + eps

        mask = (torch.rand_like(mu) > self.p).float()
        z_lower = z_lower * mask / (1 - self.p)
        z_upper = z_upper * mask / (1 - self.p)

        new_mu = (z_lower + z_upper) / 2.0
        new_eps = (z_upper - z_lower) / 2.0
        return new_mu, new_eps
