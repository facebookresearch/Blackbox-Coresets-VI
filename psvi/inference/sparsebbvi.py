# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
    Incremental variational coreset utilising the PSVI objective
"""

import time

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from psvi.inference.utils import (
    elbo,
    forward_through_coreset,
    predict_through_coreset,
    sparsevi_psvi_elbo,
)

from psvi.models.neural_net import make_fcnet, VILinear
from tqdm import tqdm


def run_sparsevi_with_bb_elbo(
    n_layers=1,
    logistic_regression=True,
    n_hidden=40,
    log_every=10,
    lr0=1e-3,
    register_elbos=False,
    seed=0,
    **kwargs,
):
    r"""
    Inremental variational coreset construction, with greedy selection step and coreset points weight vector optimization using our generalized ELBO
    """
    saved_args = locals()
    print("saved_args is", saved_args)
    np.random.seed(seed), torch.manual_seed(seed)
    elbos = []
    results = {}
    num_epochs, inner_it, outer_it = (
        kwargs["num_epochs"],
        kwargs["inner_it"],
        kwargs["outer_it"],
    )
    x, y, xt, yt, mc_samples, data_minibatch = (
        kwargs["x"],
        kwargs["y"],
        kwargs["xt"],
        kwargs["yt"],
        kwargs["mc_samples"],
        kwargs["data_minibatch"],
    )
    N, D = x.shape
    net = (
        nn.Sequential(
            VILinear(D, 1, mc_samples=mc_samples),
        )
        if logistic_regression
        else make_fcnet(
            D,
            n_hidden,
            1,
            n_layers=n_layers,
            linear_class=VILinear,
            nonl_class=nn.ReLU,
            mc_samples=mc_samples,
        )
    )
    w = (
        torch.zeros(N).clone().detach().requires_grad_(True)
    )  # coreset weights initialised to 0

    nlls_sbbvi, accs_sbbvi, core_idcs_sbbvi = [], [], []
    optim_net0 = torch.optim.Adam(
        list(net.parameters()), lr0
    )  # optimizer for ELBO on coreset datapoints
    optim_w = torch.optim.Adam([w], lr0)  # optimizer for PSVI-ELBO
    core_idcs = []

    times = [0]
    t_start = time.time()
    # Grow the coreset for num_epochs iterations
    for it in tqdm(range(num_epochs)):
        # Evaluate coreset posterior
        if it % log_every == 0:
            with torch.no_grad():
                test_data_logits, weights = predict_through_coreset(net, xt, x, y, w)
                test_probs = torch.clamp(weights @ (test_data_logits.sigmoid()), max=1)

                test_acc = test_probs.gt(0.5).float().eq(yt).float().mean()
                test_nll = -dist.Bernoulli(probs=test_probs).log_prob(yt).mean()

                nlls_sbbvi.append(test_nll.item())
                accs_sbbvi.append(test_acc.item())
                print(f"predictive accuracy: {(100*test_acc.item()):.2f}%")

                core_idcs_sbbvi.append(len(core_idcs))
                times.append(times[-1] + time.time() - t_start)

        if kwargs["scatterplot_coreset"]:
            if it == num_epochs - 1:
                test_data_logits, weights = predict_through_coreset(
                    net, kwargs["xgrid"], x, y, w
                )
                test_probs = torch.clamp(weights @ (test_data_logits.sigmoid()), max=1)
                r = (
                    test_probs.reshape(
                        (
                            int(np.sqrt(kwargs["xgrid"].shape[0])),
                            int(np.sqrt(kwargs["xgrid"].shape[0])),
                        )
                    ),
                    xt,
                    kwargs["plot_data"],
                    kwargs["plot_preds"],
                    x[w > 0],
                    y[w > 0],
                )
                kwargs["plot_classification_with_coreset"](*r, 1, "sparse bbvi")

        x_core, y_core = x[core_idcs, :], y[core_idcs]
        sub_idcs, sum_scaling = (
            np.random.randint(x.shape[0], size=data_minibatch),
            x.shape[0] / data_minibatch,
        )  # sample minibatch when accessing full data and rescale corresponding log-likelihood

        # 1. Approximate current coreset posterior via minimizing the ELBO on the coreset support
        optim_net0.zero_grad()
        for in_it in range(inner_it):
            loss = elbo(net, x_core, y_core, w[core_idcs])
            if register_elbos and in_it % log_every == 0:
                with torch.no_grad():
                    elbos.append((1, -loss.item()))
            loss.backward()
            optim_net0.step()

        with torch.no_grad():
            # 2. Compute loglikelihoods for each sample using samples from the approximation to the coreset posterior
            ll_core, ll_data, weights = forward_through_coreset(
                net, x_core, x[sub_idcs, :], y_core, y[sub_idcs], w[core_idcs]
            )
            cll_data, cll_core = ll_data - torch.einsum(
                "s, ns ->ns", weights, ll_data
            ), ll_core - torch.einsum("s, ms ->ms", weights, ll_core)

            # 3. Select point to attach to the coreset next via max correlation with residual error
            resid = sum_scaling * cll_data.sum(axis=0) - torch.einsum(
                "m, ms ->s", w[core_idcs], cll_core
            )
            corrs = (
                cll_data.matmul(resid)
                / torch.sqrt((cll_data**2).sum(axis=1))
                / cll_data.shape[1]
            )
            corecorrs = (
                torch.abs(cll_core.matmul(resid))
                / torch.sqrt((cll_core**2).sum(axis=1))
                / cll_core.shape[1]
                if len(core_idcs) > 0
                else None
            )
            if corecorrs is None or corrs.max() > corecorrs.max():
                pt_idx = sub_idcs[torch.argmax(torch.max(corrs))]
                core_idcs.append(pt_idx) if pt_idx not in core_idcs else None

            # 4. Sample for updated weights and take projected gradient descent steps on the weights
            x_core, y_core = x[core_idcs, :], y[core_idcs]
            sub_idcs, sum_scaling = (
                np.random.randint(x.shape[0], size=data_minibatch),
                x.shape[0] / data_minibatch,
            )  # sample minibatch when accessing full data and rescale corresponding log-likelihood

        for out_it in range(outer_it):
            optim_w.zero_grad()
            loss_joint = sparsevi_psvi_elbo(
                net, x[sub_idcs, :], x_core, y[sub_idcs], y_core, w[core_idcs], N
            )
            if register_elbos and out_it % log_every == 0:
                with torch.no_grad():
                    elbos.append((0, -loss_joint.item()))
            loss_joint.backward()
            optim_w.step()
            with torch.no_grad():
                torch.clamp_(w, 0)

    # store results
    results["accs"] = accs_sbbvi
    results["nlls"] = nlls_sbbvi
    results["csizes"] = core_idcs_sbbvi
    results["times"] = times[1:]
    results["elbos"] = elbos
    return results
