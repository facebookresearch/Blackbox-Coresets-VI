
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
import time
import numpy as np
import torch
import torch.distributions as dist
from torch.distributions.normal import Normal
from typing import Any, Dict
from psvi.models.logreg import model, laplace_precision, mcmc_sample, logreg_forward
from psvi.models.neural_net import categorical_fn, gaussian_fn, VILinear
from tqdm import tqdm
from psvi.experiments.experiments_utils import set_up_model, update_hyperparams_dict
from psvi.inference.utils import *
from torch.utils.data import DataLoader
from psvi.inference.psvi_classes import SubsetPreservingTransforms
from functools import partial

r"""
    Implementations of baseline inference methods.
"""

def run_laplace(
    theta,
    mu0,
    sigma0,
    x_core,
    y_core,
    w_core,
    optim_net,
    inner_it=1000,
    diagonal=True,
    mc_samples=4,
    seed=0,
    **kwargs,
):
    r"""
    Returns samples from Laplace approximation
    """
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    for _ in range(
        inner_it
    ):  # inner loop for Laplace approximation of current coreset iterate
        optim_net.zero_grad()
        ll_core, prior = model(theta, mu0, sigma0, x_core, y_core, single=True)
        loss = -w_core.dot(ll_core) - prior  # negative log-joint
        loss.backward()
        optim_net.step()
    optim_net.zero_grad()
    with torch.no_grad():
        # samples from coreset iterate
        prec = laplace_precision(x_core, theta, w_core, diagonal=diagonal)
        laplace_approx = (
            dist.MultivariateNormal(theta, precision_matrix=prec)
            if not diagonal
            else Normal(theta, prec**-0.5)
        )
        return laplace_approx.rsample((mc_samples,)).squeeze()


def run_random(
    x=None,
    y=None,
    xt=None,
    yt=None,
    mc_samples=4,
    num_epochs=100,
    log_every=10,
    N=None,
    D=None,
    seed=0,
    mcmc=False,
    lr0net=1e-3,  # initial learning rate for optimizer
    **kwargs,
) -> Dict[str, Any]:
    r"""
    Returns diagnostics from a Laplace or an MCMC fit on a random subset of the training data
    """
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    w = torch.zeros(N).clone().detach()  # coreset weights
    nlls_random, accs_random, idcs_random, times_random, core_idcs = [], [], [], [0], []
    x_test_aug = torch.cat((xt, torch.ones(xt.shape[0], 1)), dim=1)
    x_aug = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
    t_start = time.time()
    num_epochs = min(num_epochs, 2000) if mcmc else num_epochs
    for it in tqdm(range(num_epochs)):
        # Evaluate predictive performance of current coreset posterior
        if it % log_every == 0:
            if mcmc:
                param_samples = mcmc_sample(sml, core_idcs, x, y, w, seed=seed)
            else:
                # model params prior
                mu0, sigma0 = (
                    torch.zeros(D + 1),
                    torch.ones(D + 1),
                )
                theta0 = Normal(mu0, sigma0).rsample()
                theta = torch.nn.Parameter(theta0, requires_grad=True)
                optim_net = torch.optim.Adam([theta], lr0net)
                param_samples = run_laplace(
                    theta,
                    mu0,
                    sigma0,
                    x_aug[core_idcs, :],
                    y[core_idcs],
                    w[core_idcs],
                    optim_net,
                    inner_it=1000,
                    diagonal=True,
                    mc_samples=mc_samples,
                    seed=seed,
                )
            times_random.append(times_random[-1] + time.time() - t_start)
            test_probs = logreg_forward(param_samples, x_test_aug)
            test_acc = test_probs.gt(0.5).float().eq(yt).float().mean()
            test_nll = -dist.Bernoulli(probs=test_probs).log_prob(yt).mean()
            nlls_random.append(test_nll.item()), accs_random.append(
                test_acc.item()
            ), idcs_random.append(len(core_idcs))
            print(f"predictive accuracy: {(100*test_acc.item()):.2f}%")
        new_coreset_point = random.choice(
            tuple(set(range(N)).difference(set(core_idcs)))
        )
        core_idcs.append(new_coreset_point)  # attach a new random point
        w[core_idcs] = N / len(core_idcs)
    # store results
    return {
        "accs": accs_random,
        "nlls": nlls_random,
        "csizes": idcs_random,
        "times": times_random[1:],
    }



def run_giga(
    x=None,
    y=None,
    xt=None,
    yt=None,
    mc_samples=100,
    data_minibatch=512,
    num_epochs=100,
    log_every=10,
    N=None,
    D=None,
    seed=0,
    mcmc=False,
    subset_size=200,
    lr0net=1e-3,
    **kwargs,
) -> Dict[str, Any]:
    r"""
    Returns diagnostics of a fit using the GIGA coreset (Campbell & Broderick, 2018)
    """
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    mc_samples = max(
        mc_samples,
        50,  # overwrite arg for num of mc_samples for more fine grained vectors
    )
    w = (
        torch.zeros(N)
        .clone()
        .detach()
        .requires_grad_(
            requires_grad=False,
        )
    )  # coreset weights

    w_pred = (
        torch.zeros(N)
        .clone()
        .detach()
        .requires_grad_(
            requires_grad=False,
        )
    )  # rescaled weights for predictions

    # model params prior
    mu0, sigma0 = (
        torch.zeros(D + 1),
        torch.ones(D + 1),
    )

    nlls_giga, accs_giga, idcs_giga, times_giga = [], [], [], [0]
    x_aug, x_test_aug = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1), torch.cat(
        (xt, torch.ones(xt.shape[0], 1)), dim=1
    )

    core_idcs = []
    t_start = time.time()

    # Approximate the true posterior via MCMC sampling on a random subset
    # [this computation occurs once]
    sub_idcs, sum_scaling = (
        np.random.randint(x.shape[0], size=subset_size),
        x.shape[0] / data_minibatch,
    )  # sample minibatch when accessing full data and rescale corresponding log-likelihood

    if mcmc:
        with torch.no_grad():
            param_samples = mcmc_sample(
                sml,
                core_idcs,
                x[sub_idcs, :],
                y[sub_idcs],
                sum_scaling * torch.ones_like(y[sub_idcs]),
                n_samples=mc_samples,
            )
    else:
        theta0 = Normal(mu0, sigma0).rsample()
        theta = torch.nn.Parameter(theta0, requires_grad=True)
        param_samples = run_laplace(
            theta,
            mu0,
            sigma0,
            x_aug[sub_idcs, :],
            y[sub_idcs],
            sum_scaling * torch.ones_like(y[sub_idcs]),
            torch.optim.Adam([theta], lr0net),
            inner_it=1000,
            diagonal=True,
            mc_samples=mc_samples,
            seed=seed,
        )
    lw = torch.zeros(mc_samples)  # initial vector of weighted log-likelihood of coreset
    # Grow the coreset for a number of iterations
    for it in tqdm(range(num_epochs)):
        x_core, y_core = x_aug[core_idcs, :], y[core_idcs]
        sub_idcs, _ = (
            np.random.randint(x.shape[0], size=data_minibatch),
            x.shape[0] / data_minibatch,
        )  # sample minibatch when accessing full data and rescale corresponding log-likelihood

        ll_all, _ = model(
            param_samples,
            mu0,
            sigma0,
            torch.cat((x_aug[sub_idcs, :], x_core)),
            torch.cat((y[sub_idcs], y_core)),
        )
        ll_data, ll_core = ll_all[: len(sub_idcs), :], ll_all[len(sub_idcs) :, :]
        ll_data, ll_core = (
            ll_data - ll_data.mean(axis=1).repeat(ll_data.shape[1], 1).T,
            ll_core - ll_core.mean(axis=1).repeat(ll_core.shape[1], 1).T,
        )

        sum_lls = ll_data.sum(axis=0)
        norm_lls = torch.nn.functional.normalize(ll_data, dim=1)  # ell_n
        norm_sumlls = torch.nn.functional.normalize(sum_lls, dim=0)  # ell
        denom_sumlls = sum_lls.norm(p=2, dim=0)  # ||L||

        if it % log_every == 0:  # log predictive performance
            # Rescaling weights for unnormalized likelihoods in predictions
            if len(core_idcs) > 0:
                w_pred[core_idcs] = (
                    w[core_idcs]
                    * denom_sumlls
                    / ll_core.norm(p=2, dim=1)
                    * lw.dot(norm_sumlls)
                )
            if mcmc:
                predictive_samples = mcmc_sample(sml, core_idcs, x, y, w_pred)
            else:
                theta0 = Normal(mu0, sigma0).rsample()
                theta = torch.nn.Parameter(theta0, requires_grad=True)
                optim_net = torch.optim.Adam([theta], lr0net)
                predictive_samples = run_laplace(
                    theta,
                    mu0,
                    sigma0,
                    x_aug[core_idcs, :],
                    y[core_idcs],
                    w[core_idcs].detach(),
                    optim_net,
                    inner_it=100,
                    diagonal=True,
                    mc_samples=mc_samples,
                    seed=seed,
                )
            times_giga.append(times_giga[-1] + time.time() - t_start)
            test_probs = logreg_forward(predictive_samples, x_test_aug)
            test_acc = test_probs.gt(0.5).float().eq(yt).float().mean()
            test_nll = -dist.Bernoulli(probs=test_probs).log_prob(yt).mean()
            print(f"predictive accuracy: {(100*test_acc.item()):.2f}%")

            nlls_giga.append(test_nll.item())
            accs_giga.append(test_acc.item())
            idcs_giga.append(len(w[w > 0]))

            # Compute geodesic direction of each datapoint, make greedy next point selection and compute the step size
            d = torch.nn.functional.normalize(
                norm_sumlls - norm_sumlls.dot(lw) * lw, dim=0
            )
            lwr = lw.repeat(len(sub_idcs), 1)
            dns = torch.nn.functional.normalize(
                norm_lls
                - torch.einsum(
                    "n, ns -> ns", torch.einsum("ns, ns -> n", lwr, norm_lls), lwr
                ),
                dim=1,
            )
            # new datapoint selection
            pt_idx = sub_idcs[torch.argmax(torch.einsum("s, ns -> n", d, dns))]
            if pt_idx not in core_idcs:
                core_idcs.append(pt_idx)  # list of coreset point indices
                idx_new = -1
                x_core, y_core = (
                    x_aug[core_idcs, :],
                    y[core_idcs],
                )  # updated coreset support
                ll_all, _ = model(
                    param_samples,
                    mu0,
                    sigma0,
                    torch.cat((x_aug[sub_idcs, :], x_core)),
                    torch.cat((y[sub_idcs], y_core)),
                )
                ll_core = ll_all[len(sub_idcs) :, :]
                ll_core = ll_core - ll_core.mean(axis=1).repeat(ll_core.shape[1], 1).T
                norm_ll_core = torch.nn.functional.normalize(
                    ll_core, dim=1
                )  # ell_n_core
            else:
                idx_new = core_idcs.index(pt_idx)
            zeta0, zeta1, zeta2 = (
                norm_sumlls.dot(norm_ll_core[idx_new, :]),
                norm_sumlls.dot(lw),
                norm_ll_core[idx_new, :].dot(lw),
            )
            gamma = (zeta0 - zeta1 * zeta2) / (
                zeta0 - zeta1 * zeta2 + zeta1 - zeta0 * zeta2
            )
            lw = torch.nn.functional.normalize(
                (1 - gamma) * lw + gamma * norm_ll_core[idx_new, :], dim=0
            )
            # Optimal weight calibration
            w = (
                (1 - gamma) * w
                + gamma
                * torch.nn.functional.one_hot(torch.tensor(pt_idx), num_classes=N)
            ) / torch.norm((1 - gamma) * lw + gamma * norm_ll_core[idx_new, :])
            with torch.no_grad():
                torch.clamp_(w, min=0)

    # store results
    return {
        "accs": accs_giga,
        "nlls": nlls_giga,
        "csizes": idcs_giga,
        "times": times_giga[1:],
    }


def run_sparsevi(
    x=None,
    y=None,
    xt=None,
    yt=None,
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    N=None,
    D=None,
    diagonal=True,
    inner_it=10,
    outer_it=10,
    lr0net=1e-3,
    lr0v=1e-1,
    seed=0,
    mcmc=False,
    **kwargs,
) -> Dict[str, Any]:  # max coreset size
    r"""
    Returns diagnostics of a fit using Sparse VI (Campbell & Beronov, 2019)
    """
    def resc(N, w, core_idcs):
        return 1. #N/sum(w[core_idcs]) if sum(w[core_idcs])>0 else 1

    outer_it = min(outer_it, 500)  # cap to maximum value for num_epochs and outer_it
    num_epochs = min(num_epochs, 2000) if mcmc else num_epochs
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    w = (
        torch.zeros(N)
        .clone()
        .detach()
        .requires_grad_(
            requires_grad=True,
        )
    )  # coreset weights

    # model params prior
    mu0, sigma0 = (
        torch.zeros(D + 1),
        torch.ones(D + 1),
    )

    nlls_svi, accs_svi, idcs_svi, times_svi = [], [], [], [0]
    x_aug, x_test_aug = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1), torch.cat(
        (xt, torch.ones(xt.shape[0], 1)), dim=1
    )

    # Grow the coreset for a number of iterations
    core_idcs = []
    t_start = time.time()
    for it in tqdm(range(num_epochs)):
        # Evaluate predictive performance of current coreset posterior
        if it % log_every == 0:
            if mcmc:
                param_samples = mcmc_sample(sml, core_idcs, x, y, w)
            else:
                theta0 = Normal(mu0, sigma0).rsample()
                theta = torch.nn.Parameter(theta0, requires_grad=True)
                optim_net = torch.optim.Adam([theta], lr0net)
                param_samples = run_laplace(
                    theta,
                    mu0,
                    sigma0,
                    x_aug[core_idcs, :],
                    y[core_idcs],
                    resc(N, w.detach(), core_idcs)*w[core_idcs].detach(),
                    torch.optim.Adam([theta], lr0net),
                    inner_it=1000,
                    diagonal=True,
                    mc_samples=mc_samples,
                )

            times_svi.append(times_svi[-1] + time.time() - t_start)
            test_probs = logreg_forward(param_samples, x_test_aug)
            test_acc = test_probs.gt(0.5).float().eq(yt).float().mean()
            test_nll = -dist.Bernoulli(probs=test_probs).log_prob(yt).mean()

            nlls_svi.append(test_nll.item())
            accs_svi.append(test_acc.item())
            idcs_svi.append(len(core_idcs))
            print(f"predictive accuracy: {(100*test_acc.item()):.2f}%")

        # 1. Compute current coreset posterior using Laplace approximation on coreset points
        sub_idcs, sum_scaling = (
            np.random.randint(x.shape[0], size=data_minibatch),
            x.shape[0] / data_minibatch,
        )  # sample minibatch when accessing full data and rescale corresponding log-likelihood
        x_core, y_core = x_aug[core_idcs, :], y[core_idcs]

        theta0 = Normal(mu0, sigma0).rsample()
        theta = torch.nn.Parameter(theta0, requires_grad=True)
        optim_net = torch.optim.Adam([theta], lr0net)

        for _ in range(
            inner_it
        ):  # inner loop for Laplace approximation of current coreset iterate
            optim_net.zero_grad()
            ll_core, prior = model(theta, mu0, sigma0, x_core, y_core, single=True)
            loss = -resc(N, w, core_idcs)*w[core_idcs].dot(ll_core) - prior  # negative log-joint
            loss.backward()
            optim_net.step()
        with torch.no_grad():
            # samples from coreset iterate
            prec = laplace_precision(x_core, theta, resc(N, w, core_idcs)*w[core_idcs], diagonal=diagonal)
            laplace_approx = (
                dist.MultivariateNormal(theta, precision_matrix=prec)
                if not diagonal
                else Normal(theta, prec**-0.5)
            )
            param_samples = laplace_approx.rsample((mc_samples,)).squeeze()

            # 2. Compute loglikelihoods for each sample
            ll_all, _ = model(
                param_samples,
                mu0,
                sigma0,
                torch.cat((x_aug[sub_idcs, :], x_core)),
                torch.cat((y[sub_idcs], y_core)),
            )
            ll_data, ll_core = ll_all[: len(sub_idcs), :], ll_all[len(sub_idcs) :, :]
            cll_data, cll_core = (
                ll_data - ll_data.mean(axis=1).repeat(ll_data.shape[1], 1).T,
                ll_core - ll_core.mean(axis=1).repeat(ll_core.shape[1], 1).T,
            )

            # 3. Select point to attach to the coreset next
            resid = sum_scaling * cll_data.sum(axis=0) - resc(N, w, core_idcs)*w[core_idcs].matmul(cll_core)
            corrs = (
                cll_data.matmul(resid)
                / torch.sqrt((cll_data**2).sum(axis=1))
                / cll_data.shape[1]
            )
            corecorrs = (
                torch.abs(cll_core.matmul(resid))
                / torch.sqrt((cll_core**2).sum(axis=1))
                / cll_core.shape[1]
            )
            if corecorrs.shape[0] == 0 or corrs.max() > corecorrs.max():
                pt_idx = sub_idcs[torch.argmax(corrs)]
                print(f"\nAdding new point. Support increased to {len(core_idcs)+1} \n") if pt_idx not in core_idcs else print("\nImproving fit with current support \n")
                core_idcs.append(pt_idx) if pt_idx not in core_idcs else None
            else:
                print("\nImproving fit with current support \n")
            print(f"weights vector {(resc(N, w, core_idcs)*w[w>0]).sum()}")

        # 4. Sample for updated weights and take projected gradient descent steps on the weights
        # sample from updated model
        x_core, y_core = x_aug[core_idcs, :], y[core_idcs]
        optim_w = torch.optim.Adam([w], lr0v) #/(1. + it))
        theta0 = Normal(mu0, sigma0).rsample()
        theta = torch.nn.Parameter(theta0, requires_grad=True)
        for _ in range(outer_it):
            optim_net = torch.optim.Adam([theta], lr0net)
            for _ in range(
                inner_it
            ):  # inner loop for Laplace approximation of current coreset iterate
                # negative log-joint
                optim_net.zero_grad()
                ll, prior = model(theta, mu0, sigma0, x_core, y_core, single=True)
                loss = -resc(N, w, core_idcs)*w[core_idcs].dot(ll) - prior
                loss.backward()
                optim_net.step()
            with torch.no_grad():
                # samples from coreset iterate
                prec = laplace_precision(x_core, theta, resc(N, w, core_idcs)*w[core_idcs], diagonal=diagonal)
                laplace_approx = (
                    dist.MultivariateNormal(theta, precision_matrix=prec)
                    if not diagonal
                    else Normal(theta, prec**-0.5)
                )
                param_samples = laplace_approx.rsample((mc_samples,)).squeeze()

                sub_idcs, sum_scaling = (
                    np.random.randint(x_aug.shape[0], size=data_minibatch),
                    x.shape[0] / data_minibatch,
                )  # sample minibatch when accessing full data and rescale corresponding log-likelihood
                # compute w_grad
                ll_all, _ = model(
                    param_samples,
                    mu0,
                    sigma0,
                    torch.cat((x_aug[sub_idcs, :], x_core)),
                    torch.cat((y[sub_idcs], y_core)),
                )
                ll_data, ll_core = (
                    ll_all[: len(sub_idcs), :],
                    ll_all[len(sub_idcs) :, :],
                )
                cll_data, cll_core = (
                    ll_data - ll_data.mean(axis=1).repeat(ll_data.shape[1], 1).T,
                    ll_core - ll_core.mean(axis=1).repeat(ll_core.shape[1], 1).T,
                )
                resid = sum_scaling * cll_data.sum(axis=0) - resc(N, w, core_idcs) * w[core_idcs].matmul(
                    cll_core
                )
            w.grad.data[core_idcs] = (-cll_core.matmul(resid) / cll_core.shape[1]) / resc(N, w, core_idcs)
            optim_w.step()
            with torch.no_grad():
                torch.clamp_(w, 0)
    # store results
    return {
        "nlls": nlls_svi,
        "accs": accs_svi,
        "csizes": idcs_svi,
        "times": times_svi[1:],
    }



def run_opsvi(
    x=None,
    y=None,
    xt=None,
    yt=None,
    mc_samples=10,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    N=None,
    D=None,
    num_pseudo=10,
    inner_it=10,
    diagonal=True,
    lr0net=1e-3,
    lr0u=1e-3,
    lr0v=1e-3,
    register_elbos=False,
    init_args="subsample",
    seed=0,
    mcmc=False,
    log_pseudodata=False,
    **kwargs,
) -> Dict[str, Any]:
    r"""
    Returns diagnostics of a fit using the original PSVI construction (Manousakas et al, 2020)
    """
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    us, zs, ws, core_idcs_opsvi, elbos_opsvi = [], [], [], [], []
    nlls_opsvi, accs_opsvi, idcs_opsvi, times_opsvi = [], [], [], [0]
    with torch.no_grad():
        w = N / num_pseudo * (torch.ones(num_pseudo).clone().detach())
    w.requires_grad_(
        requires_grad=True,
    )  # coreset weights
    # model params prior
    mu0, sigma0 = (
        torch.zeros(D + 1),
        torch.ones(D + 1),
    )
    theta0 = Normal(mu0, sigma0).rsample()
    theta = torch.nn.Parameter(theta0, requires_grad=True)
    x_aug, x_test_aug = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1), torch.cat(
        (xt, torch.ones(xt.shape[0], 1)), dim=1
    )
    # initialization of pseudodata
    with torch.no_grad():
        u, z = (
            pseudo_rand_init(x, y, num_pseudo=num_pseudo, seed=seed)
            if init_args == "random"
            else pseudo_subsample_init(x, y, num_pseudo=num_pseudo, seed=seed)
        )
    u, z = (
        torch.cat((u, torch.ones(u.shape[0], 1)), dim=1)
        .clone()
        .detach()
        .requires_grad_(True)
    ).float(), z.float()

    optim_net = torch.optim.Adam([theta], lr0net)
    optim_u = torch.optim.Adam([u], lr0u)
    optim_w = torch.optim.Adam([w], lr0v * N)
    t_start = time.time()
    for it in tqdm(range(num_epochs)):
        # Evaluate predictive performance of current coreset posterior
        if it % log_every == 0:
            param_samples = (
                mcmc_sample(sml, list(range(num_pseudo)), u[:, :-1], z, w)
                if mcmc
                else run_laplace(
                    theta,
                    mu0,
                    sigma0,
                    u,
                    z,
                    w.detach(),
                    torch.optim.Adam([theta], lr0net),
                    inner_it=inner_it,
                    diagonal=True,
                    mc_samples=mc_samples,
                    seed=seed,
                )
            )
            times_opsvi.append(times_opsvi[-1] + time.time() - t_start)
            test_probs = logreg_forward(param_samples, x_test_aug)
            test_acc = test_probs.gt(0.5).float().eq(yt).float().mean()
            test_nll = -dist.Bernoulli(probs=test_probs).log_prob(yt).mean()
            core_idcs_opsvi.append(num_pseudo)
            nlls_opsvi.append(test_nll.item())
            accs_opsvi.append(test_acc.item())
            idcs_opsvi.append(num_pseudo)
            print(f"predictive accuracy: {(100*test_acc.item()):.2f}%")

            us.append(u.detach().numpy())
            zs.append(z.detach().numpy())
            ws.append(w.detach().numpy())

        # 1. Compute current coreset posterior using Laplace approximation on coreset points

        x_core, y_core = u, z
        # Sample for updated weights and take projected gradient descent steps on the weights
        optim_net = torch.optim.Adam([theta], lr0net)
        for in_it in range(
            inner_it
        ):  # inner loop for Laplace approximation of current coreset iterate
            # negative log-joint
            optim_net.zero_grad()
            ll, prior = model(theta, mu0, sigma0, x_core, y_core, single=True)
            loss = -w.dot(ll) - prior
            loss.backward()
            if register_elbos and in_it % log_every == 0:
                with torch.no_grad():
                    elbos_opsvi.append((1, -loss.item()))
            optim_net.step()
        optim_w.zero_grad()
        optim_u.zero_grad()
        with torch.no_grad():
            # samples from coreset iterate
            prec = laplace_precision(x_core, theta, w, diagonal=diagonal)
            laplace_approx = (
                dist.MultivariateNormal(theta, precision_matrix=prec)
                if not diagonal
                else Normal(theta, prec**-0.5)
            )
            param_samples = laplace_approx.rsample((mc_samples,)).squeeze()

            sub_idcs, sum_scaling = (
                np.random.randint(x_aug.shape[0], size=data_minibatch),
                x.shape[0] / data_minibatch,
            )  # sample minibatch when accessing full data and rescale corresponding log-likelihood
        # compute w_grad and u_grad
        ll_all, _ = model(
            param_samples,
            mu0,
            sigma0,
            torch.cat((x_aug[sub_idcs, :], x_core)),
            torch.cat((y[sub_idcs], y_core)),
        )
        ll_data, ll_core = (
            ll_all[: len(sub_idcs), :],
            ll_all[len(sub_idcs) :, :],
        )
        cll_data, cll_core = (
            ll_data - ll_data.mean(axis=1).repeat(ll_data.shape[1], 1).T,
            ll_core - ll_core.mean(axis=1).repeat(ll_core.shape[1], 1).T,
        )
        resid = sum_scaling * cll_data.sum(axis=0) - w.matmul(cll_core)
        w.grad.data = -cll_core.matmul(resid) / cll_core.shape[1]
        u_function = (
            torch.matmul(torch.einsum("m,ms->s", -w.detach(), cll_core), resid.detach())
            / cll_core.shape[1]
        )
        u.grad.data = torch.autograd.grad(u_function, u)[0]
        u.grad.data[:, -1] = 0  # zero gradient on the last column
        optim_w.step()
        optim_u.step()
        with torch.no_grad():
            torch.clamp_(w, 0)

    # store results
    results = {
        "accs": accs_opsvi,
        "nlls": nlls_opsvi,
        "csizes": core_idcs_opsvi,
        "times": times_opsvi[1:],
        "elbos": elbos_opsvi,
    }
    if log_pseudodata:
        results["us"], results["zs"], results["vs"] = us, zs, ws
    return results


def run_mfvi(
    xt=None,
    yt=None,
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    N=None,
    D=None,
    lr0net=1e-3,  # initial learning rate for optimizer
    mul_fact=2,  # multiplicative factor for total number of gradient iterations in classical vi methods
    seed=0,
    distr_fn=categorical_fn,
    architecture=None,
    n_hidden=None,
    nc=2,
    log_pseudodata=False,
    train_dataset=None,
    test_dataset=None,
    init_sd=None, 
    **kwargs,
) -> Dict[str, Any]:
    r"""
    Returns diagnostics using a mean-field VI fit on the full training dataset. Implementation supporting pytorch dataloaders
    (To be used only in the BNN experiment flows)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    nlls_mfvi, accs_mfvi, times_mfvi, elbos_mfvi, grid_preds = [], [], [0], [], []
    t_start = time.time()
    net = set_up_model(
        architecture=architecture, D=D, n_hidden=n_hidden, nc=nc, mc_samples=mc_samples, init_sd=init_sd, 
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_minibatch,
        pin_memory=True,
        shuffle=True,
    )
    n_train = len(train_loader.dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_minibatch,
        pin_memory=True,
        shuffle=True,
    )

    optim_vi = torch.optim.Adam(net.parameters(), lr0net)
    total_iterations = mul_fact * num_epochs
    checkpts = list(range(mul_fact * num_epochs))[::log_every]
    lpit = [checkpts[idx] for idx in [0, len(checkpts) // 2, -1]]
    for i in tqdm(range(total_iterations)):
        xbatch, ybatch = next(iter(train_loader))
        xbatch, ybatch = xbatch.to(device, non_blocking=True), ybatch.to(
            device, non_blocking=True
        )
        optim_vi.zero_grad()
        data_nll = -(
            n_train
            / xbatch.shape[0]
            * distr_fn(logits=net(xbatch).squeeze(-1)).log_prob(ybatch).sum()
        )
        kl = sum(m.kl() for m in net.modules() if isinstance(m, VILinear))
        mfvi_loss = data_nll + kl
        mfvi_loss.backward()
        optim_vi.step()
        with torch.no_grad():
            elbos_mfvi.append(-mfvi_loss.item())
        if i % log_every == 0 or i == total_iterations -1:
            total, test_nll, corrects = 0, 0, 0
            for xt, yt in test_loader:
                xt, yt = xt.to(device, non_blocking=True), yt.to(
                    device, non_blocking=True
                )
                with torch.no_grad():
                    test_logits = net(xt).squeeze(-1).mean(0)
                    corrects += test_logits.argmax(-1).float().eq(yt).float().sum()
                    total += yt.size(0)
                    test_nll += -distr_fn(logits=test_logits).log_prob(yt).sum()
                    if log_pseudodata and i in lpit:
                        grid_preds.append(pred_on_grid(net, device=device).detach().cpu().numpy().T)
            times_mfvi.append(times_mfvi[-1] + time.time() - t_start)
            nlls_mfvi.append((test_nll / float(total)).item())
            accs_mfvi.append((corrects / float(total)).item())
            print(f"predictive accuracy: {(100*accs_mfvi[-1]):.2f}%")
    # store results
    results = {
        "accs": accs_mfvi,
        "nlls": nlls_mfvi,
        "times": times_mfvi[1:],
        "elbos": elbos_mfvi,
        "csizes": None,
    }
    if log_pseudodata:
        results["grid_preds"] = grid_preds
    return results


def run_mfvi_subset(
    x=None,
    y=None,
    xt=None,
    yt=None,
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    D=None,
    lr0net=1e-3,  # initial learning rate for optimizer
    mul_fact=2,  # multiplicative factor for total number of gradient iterations in classical vi methods
    seed=0,
    distr_fn=categorical_fn,
    log_pseudodata=False,
    train_dataset=None,
    test_dataset=None,
    num_pseudo=100,  # constrain on random subset with size equal to the max coreset size in the experiment
    init_args="subsample",
    architecture=None,
    n_hidden=None,
    nc=2,
    dnm=None,
    init_sd=None,
    **kwargs,
) -> Dict[str, Any]:
    r"""
    Returns diagnostics using a mean-field VI fit on a random subset of the training dataset with specified size. Implementation supporting pytorch dataloaders
    (To be used only in the BNN experiment flows)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    nlls_mfvi, accs_mfvi, times_mfvi, elbos_mfvi, grid_preds = [], [], [0], [], []
    t_start = time.time()
    net = set_up_model(
        architecture=architecture, D=D, n_hidden=n_hidden, nc=nc, mc_samples=mc_samples, init_sd=init_sd,
    ).to(device)

    if dnm=="MNIST":
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_minibatch,
            # pin_memory=True,
            shuffle=True,
        )
        n_train = len(train_loader.dataset)

        points_per_class = [num_pseudo // nc] * nc  # split equally among classes
        points_per_class[-1] = num_pseudo - sum(points_per_class[:-1])
        ybatch = (
            torch.tensor(
                [
                    item
                    for sublist in [[i] * ppc for i, ppc in enumerate(points_per_class)]
                    for item in sublist
                ]
            )
            .float()
            .to(device, non_blocking=True)
        )

        def get_x_from_label(ipc, _l):
            indices = (
                torch.as_tensor(train_dataset.targets).clone().detach() == _l
            ).nonzero()
            return torch.utils.data.DataLoader(
                SubsetPreservingTransforms(train_dataset, indices=indices, dnm=dnm),
                batch_size=ipc,
                shuffle=True,
            )

        distilled_lst = []
        for c in range(nc):
            u0 = next(iter(get_x_from_label(points_per_class[c], c)))
            distilled_lst.append(u0.to(device=device, non_blocking=True))
        xbatch = torch.cat(distilled_lst).to(device, non_blocking=True)

    else:
        xbatch, ybatch = (
            pseudo_rand_init(x, y, num_pseudo=num_pseudo, seed=seed, nc=nc)
            if init_args == "random"
            else pseudo_subsample_init(x, y, num_pseudo=num_pseudo, seed=seed, nc=nc)
        )
    n_train = len(train_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_minibatch,
        pin_memory=True,
        shuffle=True,
    )
    optim_vi = torch.optim.Adam(net.parameters(), lr0net)
    sum_scaling = n_train / num_pseudo
    total_iterations = mul_fact * num_epochs
    checkpts = list(range(mul_fact * num_epochs))[::log_every]
    lpit = [checkpts[idx] for idx in [0, len(checkpts) // 2, -1]]
    for i in tqdm(range(total_iterations)):
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)
        optim_vi.zero_grad()

        data_nll = (
            -sum_scaling
            * distr_fn(logits=net(xbatch).squeeze(-1)).log_prob(ybatch).sum()
        )
        kl = sum(m.kl() for m in net.modules() if isinstance(m, VILinear))
        mfvi_loss = data_nll + kl
        mfvi_loss.backward()
        optim_vi.step()
        with torch.no_grad():
            elbos_mfvi.append(-mfvi_loss.item())
        if i % log_every == 0:
            total, test_nll, corrects = 0, 0, 0
            for xt, yt in test_loader:
                xt, yt = xt.to(device, non_blocking=True), yt.to(
                    device, non_blocking=True
                )
                with torch.no_grad():
                    test_logits = net(xt).squeeze(-1).mean(0)
                    corrects += test_logits.argmax(-1).float().eq(yt).float().sum()
                    total += yt.size(0)
                    test_nll += -distr_fn(logits=test_logits).log_prob(yt).sum()
                    if log_pseudodata and i in lpit:
                        grid_preds.append(pred_on_grid(net, device=
                        device).detach().cpu().numpy().T)
            times_mfvi.append(times_mfvi[-1] + time.time() - t_start)
            nlls_mfvi.append((test_nll / float(total)).item())
            accs_mfvi.append((corrects / float(total)).item())
            print(f"predictive accuracy: {(100*accs_mfvi[-1]):.2f}%")
    # store results
    results = {
        "accs": accs_mfvi,
        "nlls": nlls_mfvi,
        "times": times_mfvi[1:],
        "elbos": elbos_mfvi,
        "csizes": [num_pseudo] * (mul_fact * num_epochs),
    }
    if log_pseudodata:
        results["grid_preds"] = grid_preds
        results["us"], results["zs"], results["vs"] = xbatch.detach(), ybatch.detach(), [sum_scaling]*num_pseudo
    return results


# MFVI for BNN regression
def run_mfvi_regressor(
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    D=None,
    lr0net=1e-3,  # initial learning rate for optimizer
    seed=0,
    architecture=None,
    n_hidden=None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    nc=1,
    y_mean=None,
    y_std=None,
    taus=None,
    init_sd=1e-6,
    model_selection = True,
    dnm=None, 
    **kwargs,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    # normalized x train, normalized targets
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_minibatch,
        # pin_memory=True,
        shuffle=False,
    )
    # normalized x test, unnormalized targets
    test_loader, val_loader, n_train = (
        DataLoader(
            test_dataset,
            batch_size=data_minibatch,
            # pin_memory=True,
            shuffle=False,
        ),
        DataLoader(
            val_dataset,
            batch_size=data_minibatch,
            # pin_memory=True,
            shuffle=False,
        ),
        len(train_loader.dataset),
    )
    bpe = max(1, int(n_train / data_minibatch))  # batches per epoch
    def revert_norm(y_pred):
        return y_pred * y_std + y_mean
    best_tau, best_ll = taus[0], -float("inf")
    if model_selection:
        # model selection
        print("\nOptimizing precision hyperparameter")
        for tau in taus:
            print(f"\n\nTrying tau = {tau}")
            net = set_up_model(
                architecture=architecture,
                D=D,
                n_hidden=n_hidden,
                nc=nc,
                mc_samples=mc_samples,
                init_sd=init_sd,
            ).to(device)
            optim_vi = torch.optim.Adam(net.parameters(), lr0net)
            tau_fit = fit(
                net=net,
                optim_vi=optim_vi,
                train_loader=train_loader,
                pred_loader=val_loader,
                revert_norm=revert_norm,
                log_every=-1,
                tau=tau,
                epochs=num_epochs * bpe,
                device=device,
            )
            if tau_fit["lls"][-1] > best_ll:
                best_tau, best_ll = tau, tau_fit["lls"][-1]
                print(f"current best tau, best ll : {best_tau}, {best_ll}")
    else:
        best_tau = taus[0]
    print(f"\n\nselected tau : {best_tau}\n\n")
    update_hyperparams_dict(dnm, best_tau)
    net = set_up_model(
        architecture=architecture,
        D=D,
        n_hidden=n_hidden,
        nc=nc,
        mc_samples=mc_samples,
        init_sd=init_sd,
    ).to(device)
    optim_vi = torch.optim.Adam(net.parameters(), lr0net)
    results = fit(
        net=net,
        optim_vi=optim_vi,
        train_loader=train_loader,
        pred_loader=test_loader,
        revert_norm=revert_norm,
        log_every=log_every,
        tau=best_tau,
        epochs=num_epochs * bpe,
        device=device,
    )
    return results


# MFVI Subset for BNN regression
def run_mfvi_subset_regressor(
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    D=None,
    lr0net=1e-3,  # initial learning rate for optimizer
    seed=0,
    architecture=None,
    n_hidden=None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    nc=1,
    y_mean=None,
    y_std=None,
    init_sd=1e-6,
    num_pseudo=100,  # constrain on random subset with size equal to the max coreset size in the experiment
    taus=None,
    model_selection = False,
    **kwargs,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    # normalized x train, normalized targets
    sample_idcs = random.sample(range(len(train_dataset)), num_pseudo)
    subset_train_dataset = torch.utils.data.Subset(train_dataset, sample_idcs)
    subset_train_loader = DataLoader(
        subset_train_dataset,
        batch_size=num_pseudo,
        # pin_memory=True,
        shuffle=False,
    )
    # normalized x test, unnormalized targets
    test_loader, val_loader, n_train = (
        DataLoader(
            test_dataset,
            batch_size=data_minibatch,
            # pin_memory=True,
            shuffle=False,
        ),
        DataLoader(
            val_dataset,
            batch_size=data_minibatch,
            # pin_memory=True,
            shuffle=False,
        ),
        len(train_dataset),
    )
    bpe = max(1, int(n_train / data_minibatch))  # batches per epoch
    def revert_norm(y_pred):
        return y_pred * y_std + y_mean
    best_tau, best_ll = taus[0], -float("inf")
    if model_selection:
        # model selection
        print("\nOptimizing precision hyperparameter")
        for tau in taus:
            print(f"\n\nTrying tau = {tau}")
            net = set_up_model(
                architecture=architecture,
                D=D,
                n_hidden=n_hidden,
                nc=nc,
                mc_samples=mc_samples,
                init_sd=init_sd,
            ).to(device)
            optim_vi = torch.optim.Adam(net.parameters(), lr0net)
            tau_fit = fit(
                net=net,
                optim_vi=optim_vi,
                train_loader=subset_train_loader,
                pred_loader=val_loader,
                revert_norm=revert_norm,
                log_every=-1,
                tau=tau,
                epochs=num_epochs * bpe,
                device=device,
            )
            if tau_fit["lls"][-1] > best_ll:
                best_tau, best_ll = tau, tau_fit["lls"][-1]
                print(f"current best tau, best ll : {best_tau}, {best_ll}")
    else:
        best_tau = taus[0]
    print(f"\n\nselected tau : {best_tau}\n\n")
    net = set_up_model(
        architecture=architecture,
        D=D,
        n_hidden=n_hidden,
        nc=nc,
        mc_samples=mc_samples,
        init_sd=init_sd,
    ).to(device)
    optim_vi = torch.optim.Adam(net.parameters(), lr0net)
    results = fit(
        net=net,
        optim_vi=optim_vi,
        train_loader=subset_train_loader,
        pred_loader=test_loader,
        revert_norm=revert_norm,
        log_every=log_every,
        tau=best_tau,
        epochs=num_epochs * bpe,
        device=device,
    )
    results["csizes"] = [num_pseudo]
    return results



# fit mean-field BNN using the standard ELBO and log predictive performance
def fit(
    net=None,
    optim_vi=None,
    train_loader=None,
    pred_loader=None,
    revert_norm=None,
    log_every=-1,
    tau=1e-2,
    epochs=40,
    device=None,
):
    distr_fn = partial(gaussian_fn, scale=1.0 / np.sqrt(tau))
    logging_checkpoint = (
        lambda it: (it % log_every) == 0 if log_every > 0 else it == (epochs - 1)
    )  # if log_every==-1 then log pred perf only at the end of training
    lls, rmses, times, elbos = [], [], [0], []
    t_start = time.time()
    n_train = len(train_loader.dataset)
    for e in tqdm(range(epochs)):
        xbatch, ybatch = next(iter(train_loader))
        xbatch, ybatch = xbatch.to(device, non_blocking=True), ybatch.to(
            device, non_blocking=True
        )
        optim_vi.zero_grad()

        data_nll = -(
            n_train
            / xbatch.shape[0]
            * distr_fn(net(xbatch).squeeze(-1)).log_prob(ybatch.squeeze()).sum()
        )
        kl = sum(m.kl() for m in net.modules() if isinstance(m, VILinear))
        loss = data_nll + kl
        loss.backward()
        optim_vi.step()

        with torch.no_grad():
            elbos.append(-loss.item())
            if logging_checkpoint(e):
                total, test_ll, rmses_unnorm = 0, 0, 0
                for (xt, yt) in pred_loader:
                    xt, yt = (
                        xt.to(device, non_blocking=True),
                        yt.to(device, non_blocking=True).squeeze(),
                    )
                    with torch.no_grad():
                        y_pred = net(xt).squeeze(-1)
                        y_pred = revert_norm(y_pred).mean(0).squeeze()
                        rmses_unnorm += (y_pred - yt).square().sum()
                        total += yt.size(0)
                        test_ll += distr_fn(y_pred).log_prob(yt.squeeze()).sum()
                times.append(times[-1] + time.time() - t_start)
                lls.append((test_ll / float(total)).item())
                rmses.append((rmses_unnorm / float(total)).sqrt().item())
                print(
                    f"  \n\n\n  Predictive rmse {rmses[-1]:.2f} | pred ll {lls[-1]:.2f}"
                )
    results = {
        "rmses": rmses,
        "lls": lls,
        "times": times[1:],
        "elbos": elbos,
        "scale": 1.0 / np.sqrt(tau),
    }
    return results

