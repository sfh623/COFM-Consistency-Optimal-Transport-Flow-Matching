import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np


class Activations:
    """
    A collection of custom activation functions and a factory method to retrieve them.
    """

    @staticmethod
    def symm_softplus(x, softplus_=F.softplus):
        """Symmetric softplus: smooth around zero, approximates Gaussian kernel"""
        return softplus_(x) - 0.5 * x

    @staticmethod
    def softplus(x):
        """Standard softplus activation"""
        return F.softplus(x)

    @staticmethod
    def gaussian_softplus(x):
        """Gaussian-inspired softplus using error function"""
        z = np.sqrt(np.pi / 2)
        return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / (2 * z)

    @staticmethod
    def gaussian_softplus2(x):
        """Normalized Gaussian-inspired softplus"""
        z = np.sqrt(np.pi / 2)
        return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / z

    @staticmethod
    def laplace_softplus(x):
        """Combines ReLU with Laplace-like tail"""
        return F.relu(x) + torch.exp(-torch.abs(x)) / 2

    @staticmethod
    def cauchy_softplus(x):
        """Cauchy-inspired softplus"""
        pi = np.pi
        return (x * pi - torch.log(x**2 + 1) + 2 * x * torch.atan(x)) / (2 * pi)

    @staticmethod
    def activation_shifting(activation):
        """
        Returns a shifted version of `activation` such that output(0)=0
        """
        def shifted(x):
            return activation(x) - activation(torch.zeros_like(x))
        return shifted

    @classmethod
    def get_act(cls, act_fn: str, shifted: bool = False):
        """
        Factory method to retrieve activation by name.

        Args:
            act_fn: one of {'elu', 'relu', 'tanh', 'lrelu', 'sp', 'ssp', 'gsp', 'gsp2', 'lsp', 'csp'}
            shifted: if True, shifts activation so that act(0)=0
        """
        name = act_fn.lower()
        if name == 'elu':
            act = lambda x: F.elu(x)
        elif name == 'relu':
            act = lambda x: F.relu(x)
        elif name == 'celu':
            act = lambda x: F.celu(x)
        elif name == 'tanh':
            act = lambda x: torch.tanh(x)
        elif name == 'lrelu':
            act = lambda x: F.leaky_relu(x)
        elif name == 'sp':
            act = cls.softplus
        elif name == 'ssp':
            act = cls.symm_softplus
        elif name == 'gsp':
            act = cls.gaussian_softplus
        elif name == 'gsp2':
            act = cls.gaussian_softplus2
        elif name == 'lsp':
            act = cls.laplace_softplus
        elif name == 'csp':
            act = cls.cauchy_softplus
        else:
            raise ValueError(f"Unsupported activation: {act_fn}")

        if shifted:
            act = cls.activation_shifting(act)
        return act


class ActNorm(torch.nn.Module):
    """ ActNorm layer with data-dependent init."""
    _scaling_min = 0.001

    def __init__(self, num_features, logscale_factor=1., scale=1., learn_scale=True, initialized=False):
        super(ActNorm, self).__init__()
        self.initialized = initialized
        self.num_features = num_features

        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + self._scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + self._scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f"{self.num_features}"


class ActNormNoLogdet(ActNorm):

    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]