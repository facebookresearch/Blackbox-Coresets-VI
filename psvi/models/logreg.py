# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import stan
import torch
from torch.distributions.normal import Normal


def logreg_forward(thetas, x):
    return x.matmul(thetas.T).sigmoid().mean(axis=1).squeeze()


def model(thetas, mu0, sigma0, x, y, single=False):
    prior_val = Normal(mu0, sigma0).log_prob(thetas).sum()
    if single:
        return -torch.nn.BCEWithLogitsLoss(reduction="none")(x @ thetas, y), prior_val
    return (
        -torch.nn.BCEWithLogitsLoss(reduction="none")(
            x.matmul(thetas.T).squeeze(), y.repeat(thetas.shape[0], 1).T
        ),
        prior_val,
    )


def prior(D):
    mu0_w, sigma0_w, mu0_b, sigma0_b = (
        torch.zeros(D),
        torch.ones(D),
        torch.zeros(1),
        torch.ones(1),
    )
    return mu0_w, sigma0_w, mu0_b, sigma0_b


def inverse_softplus(x):
    if torch.is_tensor(x):
        return x.expm1().log()
    return np.log(np.expm1(x))


# Stan model used for coreset posterior sampling in the original Sparse VI implementation
stan_representation = """
    data {
        int<lower=0> d; // 1 + dimensionality of x
        int<lower=0> n; // number of observations
        matrix[n,d] x; // inputs
        int<lower=0,upper=1> y[n]; // outputs in {0, 1}
        vector[n] w; // weights
    }
    parameters {
        real theta0; // intercept
        vector[d] theta; // logreg params
    }
    model {
        theta0 ~ normal(0, 1);
        theta ~ normal(0, 1);
        for(i in 1:n){
          target += w[i]*bernoulli_logit_lpmf(y[i]| theta0 + x[i]*theta);
        }
    }
"""


def mcmc_sample(sml, core_idcs, x, y, w, N_per=2000, seed=42, n_samples=5):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    sampler_data = {
        "x": x[core_idcs, :].detach().cpu().numpy(),
        "y": y[core_idcs].detach().cpu().numpy().astype(int),
        "d": x.shape[1],
        "n": len(core_idcs),
        "w": w[core_idcs].detach().cpu().numpy(),
    }
    sml = stan.build(stan_representation, data=sampler_data, seed=seed)
    sampling_output = sml.sample(
        num_samples=N_per,
        chains=1,
        control={"adapt_delta": 0.9, "max_treedepth": 15},
        verbose=False,
    )[:, -n_samples:]
    param_samples = torch.cat(
        (
            torch.tensor([d["theta"] for d in sampling_output]),
            torch.tensor([d["theta0"] for d in sampling_output]).unsqueeze(axis=1),
        ),
        axis=1,
    )
    return param_samples


def laplace_precision(z_core, theta, w, diagonal=False):
    with torch.no_grad():
        m = z_core @ theta
        idcs = w > 0
        p = m[idcs].sigmoid()
        d = p * (1 - p) * w[idcs]
        a = z_core[idcs].T * d.sqrt()
        if diagonal:
            return a.pow(2).sum(1) + 1
        else:
            nll_hessian = a.matmul(a.T)
            negative_log_prior_hessian = torch.eye(z_core.shape[1])
            return negative_log_prior_hessian + nll_hessian
