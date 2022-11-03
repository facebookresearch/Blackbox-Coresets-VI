# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import operator as op
from functools import reduce

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


def gaussian_fn(loc=None, scale=None):
    return dist.normal.Normal(loc, scale)


def categorical_fn(logits=None, probs=None):
    return dist.Categorical(logits=logits, probs=probs)


def set_mc_samples(net, mc_samples):
    for module in net.modules():
        if isinstance(module, VIMixin):
            module.mc_samples = mc_samples


def inverse_softplus(x):
    if torch.is_tensor(x):
        return x.expm1().log()
    return np.log(np.expm1(x))


def prod(a):
    return reduce(op.mul, a)


def deep_getattr(obj, name):
    return reduce(getattr, name.split("."), obj)


def deep_delattr(obj, name):
    lpart, _, rpart = name.rpartition(".")
    if lpart:
        obj = deep_getattr(obj, lpart)
    delattr(obj, rpart)


def deep_setattr(obj, name, value):
    lpart, _, rpart = name.rpartition(".")
    if lpart:
        obj = deep_getattr(obj, lpart)
    setattr(obj, rpart, value)


class VIMixin(nn.Module):
    def __init__(self, *args, init_sd=0.01, prior_sd=1.0, mc_samples=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_sd = nn.Parameter(
            inverse_softplus(torch.full_like(self.weight, init_sd))
        )
        if self.bias is not None:
            self._bias_sd = nn.Parameter(
                nn.Parameter(inverse_softplus(torch.full_like(self.bias, init_sd)))
            )
        else:
            self.register_parameter("_bias_sd", None)
        (
            self.prior_sd,
            self.mc_samples,
            self._cached_weight,
            self._cached_bias,
            self._init_sd,
        ) = (
            prior_sd,
            mc_samples,
            None,
            None,
            init_sd,
        )
        self.reset_parameters_variational()

    def reset_parameters_variational(self) -> None:
        super().reset_parameters()  # pyre-ignore
        self._weight_sd.data.copy_(
            inverse_softplus(torch.full_like(self.weight, self._init_sd))
        )
        if self.bias is not None:
            self._bias_sd.data.copy_(
                inverse_softplus(torch.full_like(self.bias, self._init_sd))
            )
        self._cached_weight, self._cached_bias = (
            None,
            None,
        )

    def kl(self):
        w_kl = dist.kl_divergence(self.weight_dist, self.prior_weight_dist)
        b_kl = (
            dist.kl_divergence(self.bias_dist, self.prior_bias_dist)
            if self.bias is not None
            else 0.0
        )
        return w_kl + b_kl

    def sampled_nkl(self):
        w = self._cached_weight
        w_kl = self.prior_weight_dist.log_prob(w) - self.weight_dist.log_prob(w)
        b = self._cached_bias.squeeze(1) if self.mc_samples > 1 else self._cached_bias
        b_kl = self.prior_bias_dist.log_prob(b) - self.bias_dist.log_prob(b)
        return w_kl + b_kl

    @property
    def weight_dist(self):
        return dist.Independent(
            dist.Normal(self.weight, self.weight_sd), self.weight.ndim
        )

    @property
    def prior_weight_dist(self):
        return dist.Independent(
            dist.Normal(torch.zeros_like(self.weight), self.prior_sd), self.weight.ndim
        )

    @property
    def weight_sd(self):
        return F.softplus(self._weight_sd)

    @property
    def bias_dist(self):
        if self.bias is not None:
            return dist.Independent(
                dist.Normal(self.bias, self.bias_sd), self.bias.ndim
            )
        return None

    @property
    def prior_bias_dist(self):
        if self.bias is not None:
            return dist.Independent(
                dist.Normal(torch.zeros_like(self.bias), self.prior_sd), self.bias.ndim
            )
        return None

    @property
    def bias_sd(self):
        if self.bias is not None:
            return F.softplus(self._bias_sd)
        return None

    def rsample(self):
        weight = self.weight_dist.rsample(self.weight_batch_shape)
        bias = (
            self.bias_dist.rsample(self.bias_batch_shape)
            if self.bias is not None
            else None
        )
        return weight, bias

    @property
    def weight_batch_shape(self):
        return torch.Size((self.mc_samples,) if self.mc_samples > 1 else ())

    @property
    def bias_batch_shape(self):
        return torch.Size((self.mc_samples, 1) if self.mc_samples > 1 else ())

    def extra_repr(self):
        return f"{super().extra_repr()}, mc_samples={self.mc_samples}"


class VILinear(VIMixin, nn.Linear):
    def forward(self, x):
        self._cached_weight, self._cached_bias = self.rsample()
        return x.matmul(self._cached_weight.transpose(-2, -1)) + self._cached_bias


"""
class ResNet(torch.nn.Module):
    def __init__(self, module, skip_connection):
        super().__init__()
        self.module = module
        self.skip_connection = skip_connection

    def forward(self, x):
        return self.module(x) + self.skip_connection(x)
"""


class VIConv2d(VIMixin, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:
            raise ValueError(
                "Cannot use groups argument for variational conv layer as this is used for parallelizing across samples."
            )
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # x: S x N x C x H x W
        # or
        # x: N x C x H x W
        # reshape to: N x SC x H x W
        # so that when we convolve with
        # w: SK x C x h x w
        # we get an output with shape
        # N x SK x H' x W'
        # that we reshape to
        # S x N x K x H' x W'
        if self.mc_samples > 1:
            if x.ndim == 4:
                x = x.repeat(1, self.mc_samples, 1, 1)
            else:
                x = x.transpose(0, 1).flatten(1, 2)

        self._cached_weight, self._cached_bias = self.rsample()
        w = (
            self._cached_weight.flatten(0, 1)
            if self.mc_samples > 1
            else self._cached_weight
        )
        b = self._cached_bias.flatten()
        a = F.conv2d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.mc_samples,
        )
        if self.mc_samples > 1:
            return a.view(
                -1, self.mc_samples, self.out_channels, *a.shape[-2:]
            ).transpose(0, 1)
        return a


class BatchMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        if x.shape == 4:
            return super().forward(x)
        d0, d1 = x.shape[:2]
        x = super().forward(x.flatten(0, 1))
        return x.view(d0, d1, *x.shape[1:])


def make_fcnet(
    in_dim,
    h_dim,
    out_dim,
    n_layers=2,
    linear_class=None,
    nonl_class=None,
    mc_samples=4,
    residual=False,
    **kwargs,
):
    if linear_class is None:
        linear_class = VILinear
    if nonl_class is None:
        nonl_class = nn.ReLU

    net = nn.Sequential()
    for i in range(n_layers):
        net.add_module(
            f"lin{i}", linear_class(in_dim if i == 0 else h_dim, h_dim, **kwargs)
        )
        net.add_module(f"nonl{i}", nonl_class())
    """
    if residual:
        skip_connection = nn.Linear(in_dim, h_dim)
        net = ResNet(net, skip_connection)
    """
    net.add_module("classifier", linear_class(h_dim, out_dim, **kwargs))
    for module in net.modules():
        module.mc_samples = mc_samples
    return net


def make_regressor_net(
    in_dim,
    h_dim,
    out_dim=1,
    n_layers=2,
    linear_class=None,
    nonl_class=None,
    mc_samples=4,
    residual=False,
    **kwargs,
):
    if linear_class is None:
        linear_class = VILinear
    if nonl_class is None:
        nonl_class = nn.ReLU

    net = nn.Sequential()
    for i in range(n_layers):
        net.add_module(
            f"lin{i}",
            linear_class(in_dim if i == 0 else h_dim, h_dim, **kwargs),
        )
        net.add_module(f"nonl{i}", nonl_class())
    """
    if residual:
        skip_connection = nn.Linear(in_dim, h_dim)
        net = ResNet(net, skip_connection)
    """
    net.add_module("regressor", linear_class(h_dim, out_dim, **kwargs))
    for module in net.modules():
        module.mc_samples = mc_samples
    return net


def make_lenet(
    conv_class=None, linear_class=None, pool_class=None, nonl_class=None, **kwargs
):
    if conv_class is None:
        conv_class = VIConv2d
    if linear_class is None:
        linear_class = VILinear
    if pool_class is None:
        pool_class = BatchMaxPool2d
    if nonl_class is None:
        nonl_class = nn.ReLU

    return nn.Sequential(
        conv_class(1, 6, 5, padding=2, **kwargs),
        nonl_class(),
        pool_class(2, 2),
        conv_class(6, 16, 5, padding=0, **kwargs),
        nonl_class(),
        pool_class(2, 2),
        nn.Flatten(-3, -1),
        linear_class(400, 120, **kwargs),
        nonl_class(),
        linear_class(120, 84, **kwargs),
        nonl_class(),
        linear_class(84, 10),
    )


def make_alexnet(
    conv_class=None,
    linear_class=None,
    pool_class=None,
    nonl_class=None,
    local_response_norm_class=None,
    **kwargs,
):
    if conv_class is None:
        conv_class = VIConv2d
    if linear_class is None:
        linear_class = VILinear
    if pool_class is None:
        pool_class = BatchMaxPool2d
    if nonl_class is None:
        nonl_class = nn.ReLU
    if local_response_norm_class is None:
        local_response_norm_class = nn.LocalResponseNorm
    return nn.Sequential(
        conv_class(3, 64, 5, stride=1, padding=2),
        nonl_class(),
        pool_class(kernel_size=3, stride=2, padding=1),
        local_response_norm_class(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        conv_class(64, 64, kernel_size=5, padding=2, stride=1),
        nonl_class(),
        local_response_norm_class(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        pool_class(kernel_size=3, stride=2, padding=1),
        nn.Flatten(-3, -1),
        linear_class(
            4096, 384, **kwargs
        ),  # add kwargs so that mc_samples arg gets correctly passed
        nonl_class(),
        linear_class(384, 192, **kwargs),
        nonl_class(),
        linear_class(192, 10),
    )


class network(torch.nn.Module):
    def __init__(self, **kwargs):
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(963, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)


class MultivariateNormalVIMixin(nn.Module):
    def __init__(self, *args, init_sd=0.01, prior_sd=1., mc_samples=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.mc_samples = mc_samples
        self.prior_sd = prior_sd

        self.param_names = []
        self.param_shapes = []

        for n, p in list(self.named_parameters()):
            self.param_names.append(n)
            self.param_shapes.append(p.shape)
            deep_delattr(self, n)

        self.param_numels = list(map(prod, self.param_shapes))
        n = sum(self.param_numels)
        self.mean = nn.Parameter(p.new_zeros(n))
        self._sd = nn.Parameter(inverse_softplus(p.new_full((n,), init_sd)))
        n_corr = torch.tril_indices(n - 1, n - 1, offset=-1)[0].numel()
        self._corr = nn.Parameter(p.new_zeros(n_corr))

        self.num_params = n

    def reset_parameters_variational(self) -> None:
        raise NotImplementedError

    def kl(self):
        return dist.kl_divergence(self.param_dist, self.prior_dist)

    def sampled_nkl(self):
        x = torch.cat(
            [deep_getattr(self, n).flatten(1) for n in self.param_names], dim=1
        )
        return self.prior_dist.log_prob(x) - self.param_dist.log_prob(x)
    
    '''
    def sampled_nkl(self):
        w = self._cached_weight
        w_kl = self.prior_weight_dist.log_prob(w) - self.weight_dist.log_prob(w)
        b = self._cached_bias.squeeze(1) if self.mc_samples > 1 else self._cached_bias
        b_kl = self.prior_bias_dist.log_prob(b) - self.bias_dist.log_prob(b)
        return w_kl + b_kl
    '''
    @property
    def scale_tril(self):
        k = self.mean.new_zeros(self.num_params, self.num_params)
        k[torch.arange(self.num_params), torch.arange(self.num_params)] = F.softplus(
            self._sd
        )
        d = self.mean.size(-1) - 1
        i = torch.tril_indices(d, d, offset=-1)
        k[i[0], i[1]] = self._corr
        return k

    @property
    def param_dist(self):
        return dist.MultivariateNormal(self.mean, scale_tril=self.scale_tril)

    def rsample(self):
        x = self.param_dist.rsample((self.mc_samples,))
        return [
            xx.reshape(self.mc_samples, *shape)
            for xx, shape in zip(x.split(self.param_numels, dim=-1), self.param_shapes)
        ]

    def cached_rsample(self):
        for name, sample in zip(self.param_names, self.rsample()):
            deep_setattr(self, name, sample)

    @property
    def prior_dist(self):
        m = torch.zeros_like(self.mean)
        sd = torch.full_like(self.mean, self.prior_sd).diag_embed()
        return dist.MultivariateNormal(m, scale_tril=sd)


class VILinearMultivariateNormal(MultivariateNormalVIMixin, nn.Linear):
    def forward(self, x, **kwargs):
        super().cached_rsample()
        x = x.matmul(self.weight.transpose(-1, -2))
        if self.bias is not None:
            x = x + self.bias.unsqueeze(-2)
        return x


def make_fc2net(
    in_dim,
    h_dim,
    out_dim,
    n_layers=2,
    linear_class=None,
    nonl_class=None,
    mc_samples=4,
    residual=False,
    **kwargs,
):
    if linear_class is None:
        linear_class = VILinearMultivariateNormal
    if nonl_class is None:
        nonl_class = nn.ReLU

    net = nn.Sequential()
    for i in range(n_layers):
        net.add_module(
            f"lin{i}", linear_class(in_dim if i == 0 else h_dim, h_dim, mc_samples=mc_samples, **kwargs)
        )
        net.add_module(f"nonl{i}", nonl_class())
    """
    if residual:
        skip_connection = nn.Linear(in_dim, h_dim)
        net = ResNet(net, skip_connection)
    """
    net.add_module("classifier", linear_class(h_dim, out_dim, mc_samples=mc_samples, **kwargs))
    for module in net.modules():
        module.mc_samples = mc_samples
    return net
