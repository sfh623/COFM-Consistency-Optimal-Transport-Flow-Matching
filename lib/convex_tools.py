import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import copy
import gc
from abc import ABC, abstractmethod
from torch import nn, autograd,Tensor
from typing import Optional, Tuple, Dict
from torch.distributions import Categorical, MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily


__all__ = [
    "Positivity", "NoPositivity", "LazyClippedPositivity", "NegExpPositivity", "ExponentialPositivity",
    "ClippedPositivity", "ConvexLinear", "ConvexConv2d", "LinearSkip", "Conv2dSkip",
]


class Positivity(ABC):
    """ Interface for function that makes weights positive. """

    @abstractmethod
    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        """ Transform raw weight to positive weight. """
        ...

    def inverse_transform(self, pos_weight: torch.Tensor) -> torch.Tensor:
        """ Transform positive weight to raw weight before transform. """
        return self.__call__(pos_weight)


class NoPositivity(Positivity):
    """
    Dummy for positivity function.

    This should make it easier to compare ICNNs to regular networks.
    """

    def __call__(self, weight):
        return weight


class LazyClippedPositivity(Positivity):
    """
    Make weights positive by clipping negative weights after each update.

    References
    ----------
    Amos et al. (2017)
        Input-Convex Neural Networks.
    """

    def __call__(self, weight):
        with torch.no_grad():
            weight.clamp_(0)

        return weight
 

class NegExpPositivity(Positivity):
    """
    Make weights positive by applying exponential function on negative values during forward pass.

    References
    ----------
    Sivaprasad et al. (2021)
        The Curious Case of Convex Neural Networks.
    """

    def __call__(self, weight):
        return torch.where(weight < 0, weight.exp(), weight)


class ExponentialPositivity(Positivity):
    """
    Make weights positive by applying exponential function during forward pass.
    """

    def __call__(self, weight):
        return torch.exp(weight)

    def inverse_transform(self, pos_weight):
        return torch.log(pos_weight)


class ClippedPositivity(Positivity):
    """
    Make weights positive by using applying ReLU during forward pass.
    """

    def __call__(self, weight):
        return torch.relu(weight)


class ConvexLinear(nn.Linear):
    """Linear layer with positive weights."""

    def __init__(self, *args, positivity: Positivity = None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given as kwarg for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.positivity(self.weight), self.bias)


class ConvexConv2d(nn.Conv2d):
    """Convolutional layer with positive weights."""

    def __init__(self, *args, positivity=None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x, self.positivity(self.weight), self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class ConvexLayerNorm(nn.LayerNorm):
    """
    LayerNorm with positive weights and tracked statistics.

    Tracking statistics is necessary to make LayerNorm a convex function during inference.
    During training this module is not a convex function.
    """

    def __init__(self, normalized_shape, positivity: Positivity = None,
                 eps=1e-5, affine=True, device=None, dtype=None,
                 momentum: float = 0.1, track_running_stats: bool = True):
        if positivity is None:
            raise TypeError("positivity must be given for convex layer")

        self.track_running_stats = False
        self.momentum = momentum
        super().__init__(normalized_shape, eps, affine, device, dtype)

        self.positivity = positivity
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(normalized_shape))
            self.register_buffer("running_var", torch.ones(normalized_shape))
            self.num_batches_tracked = 0
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.num_batches_tracked = None

        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            nn.init.zeros_(self.running_mean)
            nn.init.ones_(self.running_var)
            self.num_batches_tracked = 0

    def reset_parameters(self):
        self.reset_running_stats()
        raw_val = self.positivity.inverse_transform(torch.ones(1)).item()
        nn.init.constant_(self.weight, raw_val)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        pos_weight = self.positivity(self.weight)
        if not self.track_running_stats:
            return nn.functional.layer_norm(
                x, self.normalized_shape, pos_weight, self.bias, self.eps
            )

        if self.training:
            dims = tuple(range(-len(self.normalized_shape), 0))
            mean = x.mean(dims, keepdim=True)
            var = x.var(dims, unbiased=False, keepdim=True)
            if self.training:
                self.num_batches_tracked += 1
                self.running_mean = (
                        (1 - self.momentum) * self.running_mean
                        + self.momentum * torch.mean(mean)
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * torch.mean(var)
                )
        else:
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / (var + self.eps) ** .5
        return pos_weight * x_norm + self.bias


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.weight), self.bias) * gain


class PosLinear2(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, torch.nn.functional.softmax(self.weight, 1), self.bias)##relu not work

class PositiveDense(nn.Module):
    """
    input x，softplus，
    y = x @ softplus; gain = 1/(in_features)。

    """
    def __init__(self, in_features, out_features, kernel_init=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = nn.Parameter(torch.empty(in_features, out_features))
        if kernel_init is not None:
            kernel_init(self.kernel)
        else:
            #nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.kernel, a=0, mode='fan_in', nonlinearity='relu')
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, in_features]
        kernel_pos = self.softplus(self.kernel) if callable(self.softplus) else self.kernel
        y = torch.matmul(x, kernel_pos)
        gain = 1.0 / (1*self.in_features)
        y = y * gain
        return y

class LinearSkip(nn.Module):
    """
    Fully-connected skip-connection with learnable parameters.

    The learnable parameters of this skip-connection must not be positive
    if they skip to any hidden layer from the input.
    This is the kind of skip-connection that is commonly used in ICNNs.
    """

    def __init__(self, in_features: int, out_features: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Linear(in_features, out_features, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

class Conv2dSkip(nn.Module):
    """
    Convolutional skip-connection with learnable parameters.

    The learnable parameters of this skip-connection must not be positive
    if they skip to any hidden layer from the input.
    This is the kind of skip-connection that is commonly used in ICNNs.
    """

    def __init__(self, in_channels: int, out_channels: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)


class BiConvex(nn.Module):
    """
    Combination of two convex networks for learning more general functions.

    References
    ----------
    Sankaranarayanan and Rengaswamy (2022)
        CDiNN – convex difference neural networks.
    """

    def __init__(self, conv_net: nn.Module):
        super().__init__()
        self.conv_net = conv_net
        self.conc_net = copy.deepcopy(conv_net)

    def forward(self, *args, **kwargs):
        return self.conv_net(*args, **kwargs) - self.conc_net(*args, **kwargs)


class ConvexInitialiser:
    @staticmethod
    @torch.no_grad()
    def init_log_normal_(weight: torch.Tensor, mean_sq: float, var: float) -> torch.Tensor:
        """
        Initialise weights with samples from a log-normal distribution.

        Parameters
        ----------
        weight : torch.Tensor
            The parameter to be initialised.
        mean_sq : float
            The squared mean of the normal distribution underlying the log-normal.
        var : float
            The variance of the normal distribution underlying the log-normal.

        Returns
        -------
        torch.Tensor
            The modified weight tensor.
        """
        log_mom2 = math.log(mean_sq + var)
        log_mean = math.log(mean_sq) - log_mom2 / 2.
        log_var = log_mom2 - math.log(mean_sq)
        return torch.nn.init.normal_(weight, log_mean, math.sqrt(log_var)).exp_()

    def __init__(self, var: float = 1., corr: float = 0.5,
                 bias_noise: float = 0., alpha: float = 0.):
        self.target_var = var
        self.target_corr = corr
        self.bias_noise = bias_noise
        self.relu_scale = 2. / (1. + alpha ** 2)

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        if bias is None:
            raise ValueError("Principled Initialisation for ICNNs requires bias parameter")

        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in)
        weight_mean_sq, weight_var = weight_dist
        self.init_log_normal_(weight, weight_mean_sq, weight_var)

        bias_mean, bias_var = bias_dist
        torch.nn.init.normal_(bias, bias_mean, math.sqrt(bias_var))

    def compute_parameters(self, fan_in: int, no_bias: bool = False) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
        target_mean_sq = self.target_corr / self.corr_func(fan_in)
        target_variance = self.relu_scale * (1. - self.target_corr) / fan_in

        shift = fan_in * math.sqrt(target_mean_sq * self.target_var / (2 * math.pi))
        bias_var = 0.0
        if self.bias_noise > 0.:
            target_variance *= (1 - self.bias_noise)
            bias_var = self.bias_noise * (1. - self.target_corr) * self.target_var

        return (target_mean_sq, target_variance), (-shift, bias_var)

    def corr_func(self, fan_in: int) -> float:
        """ Helper function for correlation (cf. eq. 35). """
        rho = self.target_corr
        mix_mom = math.sqrt(1 - rho ** 2) + rho * math.acos(-rho)
        return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)
