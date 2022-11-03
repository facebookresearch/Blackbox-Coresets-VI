# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import libraries
import itertools
import os
import pickle as pk
import random
import tempfile
import time
from functools import partial
from typing import Any, Dict

import fblearner.flow.api as flow
import numpy as np
import torch
import torch.nn as nn
from fblearner.flow.api import Capability

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler
from pandora.experimental.psvi.inference.psvi_dl_regression import (
    PSVIR_DL,
    PSVIRAV_DL,
    PSVIRLearnV_DL,
)

from pandora.experimental.psvi.inference.utils import (
    get_regression_benchmark,
    hyperparams_for_regression,
)
from pandora.experimental.psvi.models.neural_net import (
    gaussian_fn,
    make_regressor_net,
    VILinear,
)

from torch.utils.data import DataLoader, Dataset

# custom datasetf
class BaseDataset(Dataset):
    def __init__(self, x, y=None, randomize=False):
        self.data = (
            x.mean() + 1.0 * torch.randn_like(x) if randomize else x
        )  # if randomize return a randomized replica of the data centered around the mean of the empirical distribution
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def set_up_model(
    D=None,
    n_hidden=None,
    nc=1,
    mc_samples=None,
    architecture=None,
    **kwargs,
):
    if architecture in {
        "regressor_net",
    }:  # feed forward VI BNN for regression with diagonal covariance (optional arg for residual connections)
        return make_regressor_net(
            D,
            n_hidden,
            nc,
            linear_class=VILinear,
            nonl_class=nn.ReLU,
            mc_samples=mc_samples,
            residual=(architecture == "residual_fn"),
            **kwargs,
        )


## BB PSVI  DL
@flow.flow_async(
    resource_requirements=flow.ResourceRequirements(
        gpu=1, memory="32g", capabilities=[Capability.GPU_V100_32G_HOST]
    ),
    max_run_count=1,
)
@flow.typed()
def run_psvi_dl(**kwargs) -> Dict[str, Any]:
    return PSVIR_DL(**kwargs).run_psvi(**kwargs)


## BB PSVI learn v DL
@flow.flow_async(
    resource_requirements=flow.ResourceRequirements(
        gpu=1, memory="32g", capabilities=[Capability.GPU_V100_32G_HOST]
    ),
    max_run_count=1,
)
@flow.typed()
def run_psvi_learn_v_dl(**kwargs) -> Dict[str, Any]:
    return PSVIRLearnV_DL(**kwargs).run_psvi(**kwargs)


## BB PSVI alpha v DL
@flow.flow_async(
    resource_requirements=flow.ResourceRequirements(
        gpu=1, memory="32g", capabilities=[Capability.GPU_V100_32G_HOST]
    ),
    max_run_count=1,
)
@flow.typed()
def run_psvi_alpha_v_dl(**kwargs) -> Dict[str, Any]:
    return PSVIRAV_DL(**kwargs).run_psvi(**kwargs)


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
    for e in range(epochs):
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
    results = {
        "rmses": rmses,
        "lls": lls,
        "times": times[1:],
        "elbos": elbos,
        "scale": 1.0 / np.sqrt(tau),
    }
    return results


# MFVI DL
@flow.flow_async(
    resource_requirements=flow.ResourceRequirements(
        gpu=1,
        memory="16g",
    ),
    max_run_count=1,
)
def run_mfvi_dl(
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    D=None,
    lr0net=1e-3,  # initial learning rate for optimizer
    seed=0,
    architecture=None,
    n_hidden=None,
    log_pseudodata=False,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    nc=1,
    dnm=None,
    y_mean=None,
    y_std=None,
    taus=None,
    init_sd=1e-6,
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
    # model selection
    for tau in taus:
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
        print(f"current loglik {tau_fit['lls'][-1]}")
        print(f"val dataset shape {len(val_loader.dataset)}")
        if tau_fit["lls"][-1] > best_ll:
            best_tau, best_ll = tau, tau_fit["lls"][-1]
            print(f"current best tau, best ll : {best_tau}, {best_ll}")
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
        train_loader=train_loader,
        pred_loader=test_loader,
        revert_norm=revert_norm,
        log_every=log_every,
        tau=best_tau,
        epochs=num_epochs * bpe,
        device=device,
    )
    print("final lls : ", ["%.4f" % el for el in results["lls"]])
    return results


# MFVI Subset DL
@flow.flow_async(
    resource_requirements=flow.ResourceRequirements(
        gpu=1,
        memory="16g",
    ),
    max_run_count=1,
)
def run_mfvi_subset_dl(
    mc_samples=4,
    data_minibatch=128,
    num_epochs=100,
    log_every=10,
    D=None,
    lr0net=1e-3,  # initial learning rate for optimizer
    seed=0,
    architecture=None,
    n_hidden=None,
    log_pseudodata=False,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    nc=1,
    dnm=None,
    y_mean=None,
    y_std=None,
    taus=None,
    init_sd=1e-6,
    num_pseudo=100,  # constrain on random subset with size equal to the max coreset size in the experiment
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

    print(f"subset size : {len(subset_train_dataset)}")

    bpe = max(1, int(n_train / data_minibatch))  # batches per epoch

    def revert_norm(y_pred):
        return y_pred * y_std + y_mean

    best_tau, best_ll = taus[0], -float("inf")
    # model selection
    for tau in taus:
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
        print(f"current loglik {tau_fit['lls'][-1]}")
        print(f"val dataset shape {len(val_loader.dataset)}")
        if tau_fit["lls"][-1] > best_ll:
            best_tau, best_ll = tau, tau_fit["lls"][-1]
            print(f"current best tau, best ll : {best_tau}, {best_ll}")
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
    print("rmses : ", ["%.4f" % el for el in results["rmses"]])
    print("lls : ", ["%.4f" % el for el in results["lls"]])
    print("elbos : ", ["%.4f" % el for el in results["elbos"]])
    results["csizes"] = [num_pseudo]
    return results


## Run experiment
@flow.flow_async(resource_requirements=flow.ResourceRequirements(gpu=1, memory="8g"))
@flow.typed()
def experiment_driver(
    methods: Dict[str, bool],
    method_args: Dict[str, Any],
    fnm: str,
) -> None:
    results = {}
    BUCKET, pathnm = "dionm_ml", "tree/regression"
    MANIFOLD_BASEPATH = f"manifold://{BUCKET}/{pathnm}"
    for dnm in method_args["datasets"]:

        (X, Y), indices = get_regression_benchmark(
            dnm,
            data_dir=MANIFOLD_BASEPATH,
            seed=method_args["seed"],
            p_split=(-1, 0.1, method_args["num_test"]),
        )
        taus = hyperparams_for_regression()[dnm]

        # split into training and test sets
        x, y, xv, yv, xt, yt = (
            X[indices["train"]],
            Y[indices["train"]],
            X[indices["val"]],
            Y[indices["val"]],
            X[indices["test"]],
            Y[indices["test"]],
        )

        N, D = x.shape
        # compute training set statistics for normalization
        x_mean, y_mean, x_std, y_std = (
            np.mean(x, 0),
            np.mean(y),
            np.std(x, 0),
            np.std(y),
        )
        # Parse in torch dataloaders
        train_dataset, val_dataset, test_dataset, y_mean, y_std = (
            BaseDataset(
                torch.from_numpy(
                    ((x - np.full(x.shape, x_mean)) / np.full(x.shape, x_std)).astype(
                        np.float32
                    )
                ),
                torch.from_numpy(((y - y_mean) / y_std).astype(np.float32)),
            ),
            BaseDataset(
                torch.from_numpy(
                    (
                        (xv - np.full(xv.shape, x_mean)) / np.full(xv.shape, x_std)
                    ).astype(np.float32)
                ),
                torch.from_numpy(yv.astype(np.float32)),
            ),
            BaseDataset(
                torch.from_numpy(
                    (
                        (xt - np.full(xt.shape, x_mean)) / np.full(xt.shape, x_std)
                    ).astype(np.float32)
                ),
                torch.from_numpy(yt.astype(np.float32)),
            ),
            torch.tensor(y_mean),
            torch.tensor(y_std),
        )

        # Run methods
        results[dnm] = {}
        inf_dict = {
            "psvi_dl": run_psvi_dl,
            "psvi_learn_v_dl": run_psvi_learn_v_dl,
            "psvi_alpha_v_dl": run_psvi_alpha_v_dl,
            "mfvi_dl": run_mfvi_dl,
            "mfvi_subset_dl": run_mfvi_subset_dl,
        }

        for nm_alg in methods:
            logistic_regression = method_args.get(
                "regression", method_args.get("architecture") == "regression"
            )

            if methods[nm_alg]:
                inf_alg = inf_dict[nm_alg]
            else:
                continue

            # store dictionary with results
            results[dnm][nm_alg] = {}
            if not isinstance(method_args["mc_samples"], list):
                method_args["mc_samples"] = [
                    method_args["mc_samples"]
                ]  # make compatible with older json that do not support lists for the mc_samples argument
            for e, (lr0net, lr0u, lr0v, in_it, trainer, mc_samples) in enumerate(
                itertools.product(
                    *(
                        method_args["lr0net"],
                        method_args["lr0u"],
                        method_args["lr0v"],
                        method_args["inner_it"],
                        method_args["trainer"],
                        method_args["mc_samples"],
                    )
                )
            ):
                hyp = (lr0net, lr0u, lr0v, in_it, trainer, mc_samples)
                results[dnm][nm_alg][hyp] = {}
                (
                    lls_list,
                    rmses_list,
                    wents_list,
                    nesses_list,
                    vents_list,
                    lls_full_list,
                    rmses_full_list,
                    wents_full_list,
                    nesses_full_list,
                    vents_full_list,
                    csize_list,
                    times_list,
                    elbos_list,
                    us_list,
                    zs_list,
                    vs_list,
                    alphas_list,
                ) = [[] for _ in range(17)]
                for t in range(method_args["num_trials"]):
                    (
                        csizes_,
                        lls_,
                        rmses_,
                        rmses_full_,
                        lls_full_,
                        times_,
                        elbos_,
                        wents_,
                        nesses_,
                        wents_full_,
                        nesses_full_,
                        vents_,
                        vents_full_,
                        us_,
                        zs_,
                        vs_,
                        alphas_,
                    ) = [[] for _ in range(17)]
                    if (
                        nm_alg.startswith("psvi")
                        or nm_alg.startswith("opsvi")
                        or nm_alg.startswith("mfvi_subset")
                    ):
                        compute_weights_entropy = (
                            False
                            if nm_alg.startswith("opsvi")
                            or nm_alg.startswith("mfvi_subset")
                            else method_args["compute_weights_entropy"]
                        )
                        for ps in method_args[
                            "coreset_sizes"
                        ]:  # range of pseudocoreset sizes tested over the experiment
                            results_inf_alg = inf_alg(
                                mc_samples=mc_samples,
                                num_epochs=method_args["num_epochs"],
                                data_minibatch=method_args["data_minibatch"],
                                D=D,
                                N=N,
                                tr=t,
                                diagonal=method_args.get("diagonal", None),
                                inner_it=in_it,
                                scatterplot_coreset=method_args.get(
                                    "scatterplot_coreset"
                                ),  # not parsed for some methods atm
                                logistic_regression=logistic_regression,
                                trainer=trainer,
                                log_every=method_args["log_every"],
                                register_elbos=method_args["register_elbos"],
                                lr0u=lr0u,
                                lr0net=lr0net,
                                lr0v=lr0v,
                                init_args=method_args["init_args"],
                                init_sd=method_args[
                                    "init_sd"
                                ],  # initialization of variance in variational model
                                num_pseudo=ps,
                                seed=t,  # map random seed to the trial number for reproducibility of inference result at the beginning of each of the baseline
                                compute_weights_entropy=compute_weights_entropy,
                                reset=method_args.get("reset"),
                                reset_interval=method_args.get("reset_interval"),
                                architecture=method_args.get("architecture"),
                                log_pseudodata=method_args.get("log_pseudodata"),
                                n_hidden=method_args.get(
                                    "n_hidden"
                                ),  # hidden units in nn architecture
                                n_layers=method_args.get("n_layers", 1),
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                test_dataset=test_dataset,
                                y_mean=y_mean,
                                y_std=y_std,
                                dnm=dnm,
                                learn_z=method_args.get("learn_z", True),
                                prune=method_args.get("prune", False),
                                prune_interval=method_args.get("prune_interval", None),
                                prune_sizes=method_args.get("prune_sizes", None),
                                gamma=method_args.get(
                                    "gamma", 1.0
                                ),  # multiplier at steps of lr scheduler
                                lr0alpha=method_args.get(
                                    "lr0alpha", 1e-3
                                ),  # lr for alpha
                                taus=taus,
                            )  # .catch(lambda _: None)
                            if results_inf_alg is not None:
                                lls, rmses, csizes, times, elbos = (
                                    results_inf_alg["lls"],
                                    results_inf_alg["rmses"],
                                    results_inf_alg["csizes"],
                                    results_inf_alg["times"],
                                    results_inf_alg["elbos"],
                                )
                                if nm_alg == "psvi_alpha_v_dl":
                                    alphas_.append(results_inf_alg["alpha"])
                                csizes_.append(csizes[-1])
                                lls_.append(lls[-1])
                                rmses_.append(rmses[-1])
                                rmses_full_.append(rmses)
                                lls_full_.append(lls)
                                times_.append(times)
                                elbos_.append(elbos)
                                us_.append(results_inf_alg.get("us", None))
                                zs_.append(results_inf_alg.get("zs", None))
                                vs_.append(results_inf_alg.get("vs", None))
                                if compute_weights_entropy and nm_alg not in {
                                    "psvi_no_iw",
                                    "mfvi_subset",
                                    "mfvi_subset_dl",
                                }:  # for inference methods using multi-sample VI objectives, store IW statistics
                                    wents = results_inf_alg["went"]
                                    wents_.append(wents[-1])
                                    wents_full_.append(wents)
                                    ness = results_inf_alg["ness"]
                                    nesses_.append(ness[-1])
                                    nesses_full_.append(ness)
                                if nm_alg not in {
                                    "mfvi_subset",
                                    "mfvi_subset_dl",
                                    "opsvi",
                                }:
                                    vents = results_inf_alg["vent"]
                                    vents_.append(vents[-1])
                                    vents_full_.append(vents)
                        csize_list.append(csizes_)
                        lls_list.append(lls_)
                        rmses_list.append(rmses_)
                        alphas_list.append(alphas_)
                        rmses_full_list.append(rmses_full_)
                        lls_full_list.append(lls_full_)
                        times_list.append(times_)
                        elbos_list.append(elbos_)
                        us_list.append(us_)
                        zs_list.append(zs_)
                        vs_list.append(vs_)
                        vents_list.append(vents_)
                        vents_full_list.append(vents_full_)
                        results[dnm][nm_alg][hyp]["rmses_full"] = rmses_full_list
                        results[dnm][nm_alg][hyp]["lls_full"] = lls_full_list
                        if nm_alg not in {"mfvi_subset", "mfvi_subset_dl", "opsvi"}:
                            results[dnm][nm_alg][hyp]["vent"] = vents_
                            results[dnm][nm_alg][hyp]["vent_full"] = vents_full_list
                        if nm_alg == "psvi_alpha_v_dl":
                            results[dnm][nm_alg][hyp]["alphas"] = alphas_list
                        if compute_weights_entropy:
                            wents_list.append(wents_)
                            wents_full_list.append(wents_full_)
                            results[dnm][nm_alg][hyp]["went"] = wents_list
                            results[dnm][nm_alg][hyp]["went_full"] = wents_full_list
                            nesses_list.append(nesses_)
                            nesses_full_list.append(nesses_full_)
                            results[dnm][nm_alg][hyp]["ness"] = nesses_list
                            results[dnm][nm_alg][hyp]["ness_full"] = nesses_full_list
                    else:
                        results_inf_alg = inf_alg(
                            mc_samples=mc_samples,
                            num_epochs=method_args["num_epochs"],
                            data_minibatch=method_args["data_minibatch"],
                            D=D,
                            N=N,
                            tr=t,
                            diagonal=method_args.get("diagonal", None),
                            inner_it=in_it,
                            outer_it=method_args.get("outer_it", None),
                            scatterplot_coreset=method_args.get("scatterplot_coreset"),
                            logistic_regression=logistic_regression,
                            trainer=trainer,
                            log_every=method_args["log_every"],
                            register_elbos=method_args["register_elbos"],
                            lr0net=lr0net,
                            lr0v=lr0v,
                            seed=t,  # set random seed equal to the trial number for reproducibility of inference result at the beginning of each of the baseline
                            subset_size=method_args.get("coreset_sizes")[-1],
                            architecture=method_args.get("architecture"),
                            log_pseudodata=method_args.get("log_pseudodata"),
                            n_hidden=method_args.get(
                                "n_hidden"
                            ),  # hidden units in nn architecture
                            dnm=dnm,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset,
                            y_mean=y_mean,
                            y_std=y_std,
                            init_sd=method_args["init_sd"],
                            taus=taus,
                        )
                        lls, rmses, times = (
                            results_inf_alg["lls"],
                            results_inf_alg["rmses"],
                            results_inf_alg["times"],
                        )
                        lls_list.append(lls), rmses_list.append(
                            rmses
                        ), times_list.append(times)
                        if nm_alg.startswith("sparse-bbvi") or nm_alg.startswith(
                            "mfvi"
                        ):  # methods returning ELBOS
                            elbos = results_inf_alg["elbos"]
                            elbos_list.append(elbos)
                        us = results_inf_alg.get("us")
                        zs = results_inf_alg.get("zs")
                        vs = results_inf_alg.get("vs")
                        us_list.append(us), zs_list.append(zs), vs_list.append(vs)
                results[dnm][nm_alg][hyp]["lls"] = lls_list
                results[dnm][nm_alg][hyp]["rmses"] = rmses_list
                results[dnm][nm_alg][hyp]["times"] = times_list
                results[dnm][nm_alg][hyp]["elbos"] = elbos_list
                if not nm_alg.startswith("mfvi"):
                    results[dnm][nm_alg][hyp]["vs"] = vs_list
                if method_args.get("log_pseudodata"):
                    if nm_alg not in {"mfvi", "mfvi_dl"}:
                        results[dnm][nm_alg][hyp]["us"] = us_list
                        results[dnm][nm_alg][hyp]["zs"] = zs_list
    return write_to_manifold(results, fnm)


## Write results to manifold
@flow.flow_async(resource_requirements=flow.ResourceRequirements(cpu=1, memory="8g"))
@flow.typed()
def write_to_manifold(results: Dict[str, Any], fnm: str) -> None:
    BUCKET, pathnm = "dionm_ml", "tree/logreg"
    MANIFOLD_BASEPATH = f"manifold://{BUCKET}"
    # Store results locally at a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        pk.dump(results, tmp_file, pk.HIGHEST_PROTOCOL)
    tmp_filename = tmp_file.name
    # Copy results on manifold
    pathmgr = PathManager()
    pathmgr.register_handler(ManifoldPathHandler())
    pathmgr.copy_from_local(
        tmp_filename,
        os.path.join(MANIFOLD_BASEPATH, pathnm, fnm + ".pk"),
        overwrite=True,
    )


## Entry point
@flow.flow_async()
@flow.registered(owners=["oncall+pandora"])
@flow.typed()
def experiment(
    methods: Dict[str, bool],
    method_args: Dict[str, Any],
    fnm: str,
    **kwargs,
) -> None:

    return experiment_driver(
        methods,
        method_args,
        fnm,
    )  # run experiment