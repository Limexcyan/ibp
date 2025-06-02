import torch
import torch.nn.functional as F
from typing import Tuple

class CROWNLinear:
    def __init__(self, in_features: int, out_features: int, device=None) -> None:
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

    def forward(
        self,
        z_l: torch.Tensor,
        z_u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear layer bound propagation using interval bound method.

        Args:
            z_l: Lower input bounds [batch, in_features]
            z_u: Upper input bounds [batch, in_features]
            weight: Weight matrix [out_features, in_features]
            bias: Optional bias [out_features]
            device: Target device

        Returns:
            Tuple of (lower bound, upper bound) [batch, out_features]
        """
        z_l = z_l.to(device)
        z_u = z_u.to(device)

        W_pos = torch.clamp(weight, min=0)
        W_neg = torch.clamp(weight, max=0)

        z_l_out = z_l @ W_pos.T + z_u @ W_neg.T
        z_u_out = z_u @ W_pos.T + z_l @ W_neg.T

        if bias is not None:
            z_l_out = z_l_out + bias
            z_u_out = z_u_out + bias

        return z_l_out, z_u_out



class CROWNConv2d:
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(
        self,
        z_l: torch.Tensor,
        z_u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conv2D bound propagation using interval bound method.

        Args:
            z_l: Lower input bounds [batch, C_in, H, W]
            z_u: Upper input bounds [batch, C_in, H, W]
            weight: Conv filter [C_out, C_in, KH, KW]
            bias: Optional bias [C_out]
            device: Target device

        Returns:
            Tuple of (lower bound, upper bound) [batch, C_out, H_out, W_out]
        """
        z_l = z_l.to(device)
        z_u = z_u.to(device)
        
        W_pos = torch.clamp(weight, min=0)
        W_neg = torch.clamp(weight, max=0)

        z_l_out = F.conv2d(z_l, W_pos, bias=bias) + F.conv2d(z_u, W_neg)
        z_u_out = F.conv2d(z_u, W_pos, bias=bias) + F.conv2d(z_l, W_neg)

        return z_l_out, z_u_out


class CROWNReLU:
    def __init__(self):
        pass

    def forward(
        self,
        z_l: torch.Tensor,
        z_u: torch.Tensor,
        alpha: torch.Tensor = None,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ReLU bound propagation using linear relaxation.

        Args:
            z_l: Lower input bounds
            z_u: Upper input bounds
            alpha: Optional optimized slope tensor
            device: Target device

        Returns:
            Tuple of (lower bound, upper bound) after ReLU
        """
        z_l = z_l.to(device)
        z_u = z_u.to(device)
        zero = torch.zeros_like(z_l)
        one = torch.ones_like(z_l)

        assert (z_l <= z_u).all()

        # Default α (if not learned)
        if alpha is None:
            alpha = torch.where(
                z_u <= 0, zero,
                torch.where(z_l >= 0, one, z_u / (z_u - z_l + 1e-12))
            )

        # Lower bound: always ReLU(l)
        z_l_out = torch.where(z_l >= 0, z_l, torch.zeros_like(z_l))

        # Upper bound: piecewise linear upper relaxation
        z_u_out = torch.where(
            z_u <= 0, torch.zeros_like(z_u),
            torch.where(z_l >= 0, z_u, alpha * (z_u - z_l))
        )

        # Ensure numerical correctness
        assert (z_l_out <= z_u_out).all(), f"ReLU bounds violated. Min diff: {(z_u_out - z_l_out).min().item()}"

        return z_l_out, z_u_out


class CROWNAvgPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(
        self,
        z_l: torch.Tensor,
        z_u: torch.Tensor,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Average Pooling bound propagation.

        Args:
            z_l: Lower input bounds [batch, C, H, W]
            z_u: Upper input bounds [batch, C, H, W]
            device: Target device

        Returns:
            Tuple of (lower bound, upper bound) after pooling
        """
        z_l = z_l.to(device)
        z_u = z_u.to(device)

        z_l_out = F.avg_pool2d(z_l, self.kernel_size, self.stride, self.padding)
        z_u_out = F.avg_pool2d(z_u, self.kernel_size, self.stride, self.padding)
        return z_l_out, z_u_out


class CROWNFlatten:
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(
        self,
        z_l: torch.Tensor,
        z_u: torch.Tensor,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten bounds across dimensions.

        Args:
            z_l: Lower bounds
            z_u: Upper bounds
            device: Target device

        Returns:
            Flattened (lower, upper) bounds
        """
        z_l = z_l.to(device)
        z_u = z_u.to(device)

        z_l_out = z_l.flatten(self.start_dim, self.end_dim)
        z_u_out = z_u.flatten(self.start_dim, self.end_dim)
        return z_l_out, z_u_out
    
class CROWNBatchNorm:
    def __init__(self):
        pass

    def forward(self, z_l: torch.Tensor, z_u: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                running_mean: torch.Tensor, running_var: torch.Tensor, stats_id, 
                batch_norm_forward, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BatchNorm with interval bounds using CROWN-style lower and upper bounds internally,
        compatible with IntervalBatchNorm API.
        """
        z_l = z_l.to(device)
        z_u = z_u.to(device)

        mu = (z_u+z_l)/2.0

        # Determine if input is convolutional (4D) or linear (2D)
        if mu.dim() == 4:
            # Conv2d: (N, C, H, W)
            reduce_dims = [0, 2, 3]
            shape = (1, -1, 1, 1)
        elif mu.dim() == 2:
            # Linear: (N, F)
            reduce_dims = [0]
            shape = (1, -1)
        else:
            raise ValueError(f"Unsupported input dimensions for BatchNorm: {mu.shape}")

        # Estimate batch statistics from mu (midpoint)
        mean = mu.mean(dim=reduce_dims, keepdim=True)
        var = ((mu - mean) ** 2).mean(dim=reduce_dims, keepdim=True)
        std = torch.sqrt(var + 1e-5)

        gamma = weight.view(*shape)
        beta = bias.view(*shape)

        norm_l = gamma * (z_l - mean) / std + beta
        norm_u = gamma * (z_u - mean) / std + beta

        # Final bounds
        z_l_bn = torch.min(norm_l, norm_u)
        z_u_bn = torch.max(norm_l, norm_u)

        return z_l_bn, z_u_bn
