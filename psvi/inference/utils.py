# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist
from psvi.models.neural_net import VILinear
from torch.utils.data import DataLoader

def pseudo_subsample_init(x, y, num_pseudo=20, nc=2, seed=0):
    r"""
    Initialize on random subsets from each class with approximately equal
    """
    torch.manual_seed(seed)
    N, _ = x.shape
    cnt = 0
    u, z = torch.Tensor([]), torch.Tensor([])
    for c in range(nc):
        idx_c, pts_with_c = (
            torch.arange(N)[y == c],
            num_pseudo // nc if c < nc - 1 else num_pseudo - cnt,
        )
        u, z = torch.cat(
            (u, x[idx_c[torch.randperm(len(idx_c))[:pts_with_c]]])
        ), torch.cat((z, c * torch.ones(pts_with_c)))
        cnt += num_pseudo // nc
    return u.requires_grad_(True), z


def pseudo_rand_init(x, y, num_pseudo=20, nc=2, seed=0, variance=0.1):
    r"""
    Initialize on noisy means of the observed datapoints and random labels equally split among classes
    """
    torch.manual_seed(seed)
    _, D = x.shape
    u = (
        (x[:, :].mean() + variance * torch.randn(num_pseudo, D))
        .clone()
        .requires_grad_(True)
    )
    z = torch.Tensor([])
    for c in range(nc):
        z = torch.cat(
            (
                z,
                c
                * torch.ones(
                    num_pseudo // nc
                    if c < nc - 1
                    else num_pseudo - (nc - 1) * (num_pseudo // nc)
                ),
            )
        )
    return u, z


r"""
Model specific computations for psvi variational objective used to estimate the coreset posterior over black-box sparsevi construction
"""


def elbo(net, u, z, w):
    r"""
    ELBO computed on (u,z): variational objective for posterior approximation using only the coreset datapoints
    """
    pseudo_nll = -dist.Bernoulli(logits=net(u).squeeze(-1)).log_prob(z).matmul(w)
    sampled_nkl = sum(m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear))
    return (pseudo_nll.sum() - sampled_nkl).sum()


def sparsevi_psvi_elbo(net, x, u, y, z, w, N):  # variational objective for
    r"""
    PSVI-ELBO: variational objective for true data conditioned on coreset data (called in outer optimization of the sparse-bbvi construction)
    """
    Nu, Nx = u.shape[0], x.shape[0]
    all_data, all_labels = torch.cat((u, x)), torch.cat((z, y))
    all_nlls = -dist.Bernoulli(logits=net(all_data).squeeze(-1)).log_prob(all_labels)
    pseudo_nll, data_nll = N / Nu * all_nlls[:, :Nu].matmul(w), all_nlls[:, Nu:].sum(-1)
    sampled_nkl = sum(m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear))
    log_weights = -pseudo_nll + sampled_nkl
    weights = log_weights.softmax(-1).squeeze()
    return weights.mul(N / Nx * data_nll - pseudo_nll).sum() - log_weights.mean()


def forward_through_coreset(net, u, x, z, y, w):
    r"""
    Likelihood computations for coreset next datapoint selection step
    """
    Nu = u.shape[0]
    with torch.no_grad():
        all_data, all_labels = torch.cat((u, x)), torch.cat((z, y))
        all_lls = dist.Bernoulli(logits=net(all_data).squeeze(-1)).log_prob(all_labels)
        core_ll, data_ll = all_lls[:, :Nu], all_lls[:, Nu:]
        sampled_nkl = sum(
            m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear)
        )
        log_weights = core_ll.matmul(w) + sampled_nkl
        weights = log_weights.softmax(-1).squeeze()
        return core_ll.T, data_ll.T, weights


def predict_through_coreset(net, xt, x, y, w=None):
    r"""
    Importance-weight correction for predictions using the coreset posterior
    """
    Ntest = xt.shape[0]
    with torch.no_grad():
        all_data = torch.cat((xt, x))
        all_logits = net(all_data).squeeze(-1)
        pnlls = -dist.Bernoulli(logits=all_logits[:, Ntest:]).log_prob(y)
        pseudo_nll = pnlls.matmul(w) if w is not None else pnlls.sum(-1)
        test_data_logits = all_logits[:, :Ntest]
        sampled_nkl = sum(
            m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear)
        )
        log_weights = -pseudo_nll + sampled_nkl
        weights = log_weights.softmax(-1).squeeze()
        return test_data_logits, weights


def make_dataloader(data, minibatch, shuffle=True):
    r"""
    Create pytorch dataloader from given dataset and minibatch size
    """
    return DataLoader(data, batch_size=minibatch, pin_memory=True, shuffle=shuffle)


def compute_empirical_mean(dloader):
    r"""
    Compute the mean of the observed data distribution
    """
    trainsum, nb_samples = 0., 0. # compute statistics of the training data
    for data, _ in dloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        trainsum += data.mean(2).sum(0) # use with caution: might raise overflow for large datasets
        nb_samples += batch_samples
    return trainsum / nb_samples 


def pred_on_grid(
    model,
    n_test_per_dim=250,
    device=None,
    **kwargs,
):
    r"""
    Predictifons over a 2-d grid for visualization of predictive posterior on 2-d synthetic datasets
    """
    _x0_test = torch.linspace(-3, 4, n_test_per_dim)
    _x1_test = torch.linspace(-2, 3, n_test_per_dim)
    x_test = torch.stack(torch.meshgrid(_x0_test, _x1_test), dim=-1).to(device)

    with torch.no_grad():
        return model(x_test.view(-1, 2)).squeeze(-1).softmax(-1).mean(0)