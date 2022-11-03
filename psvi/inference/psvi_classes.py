# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Black-box PSVI parent and children classes accessing the dataset via pytorch dataloaders.
"""

import time
import random
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from PIL import Image
from psvi.experiments.experiments_utils import SynthDataset
from psvi.hypergrad.diff_optimizers import DifferentiableAdam, GradientDescent
from psvi.hypergrad.hypergradients import CG_normaleq, fixed_point
from psvi.models.neural_net import (
    set_mc_samples,
    categorical_fn,
    gaussian_fn,
    make_fcnet,
    make_fc2net,
    make_lenet,
    make_alexnet,
    make_regressor_net,
    VILinear,
    VILinearMultivariateNormal,
)
from psvi.robust_higher import innerloop_ctx
from psvi.robust_higher.patch import monkeypatch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from functools import partial 
from psvi.inference.utils import make_dataloader, compute_empirical_mean

class SubsetPreservingTransforms(Dataset):
    r"""
    Subset of a dataset at specified indices with a specified list of transforms.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices=None, dim=2, dnm="Cifar10"):
        self.dataset = dataset
        self.indices = indices
        self.dnm = dnm
        self.dim = dim

    def __getitem__(self, idx):
        if self.dnm not in {"MNIST", "Cifar10"}:
            return self.dataset.data[self.indices[idx]].reshape((self.dim,))
        im = (
            Image.fromarray(self.dataset.data[self.indices[idx]])  # Cifar10
            if not self.dnm == "MNIST"
            else Image.fromarray(
                np.reshape(self.dataset.data[self.indices[idx]].numpy(), (28, 28)),
                mode="L",  # MNIST
            )  # TBC: Supporting only Cifar10 and MNIST
        )
        return self.dataset.transform(im)

    def __len__(self):
        return len(self.indices)


class PSVI(object):
    r"""
    PSVI 
        - with fixed rescaled coefficients on pseudodata supporting pytorch dataloaders
    """

    def __init__(
        self,
        u=None,  # pseudo x-coordinates
        z=None,  # pseudo y-coordinates
        train_dataset=None,  # true training data
        test_dataset=None,  # test data
        N=None,  # size of training data
        D=None,  # dimensionality of training data
        model=None,  # statistical model
        optim=None,  # joint variational model/pseudodata optimizer
        optim_u=None,  # optimizer for pseudodata
        optim_net=None,  # optimizer for variational model parameters
        optim_v=None,  # optimizer for log-likelihood rescaling vector
        optim_z=None,  # optimizer for soft labels on distilled data
        register_elbos=True,  # register values of objectives over inference
        num_pseudo=None,  # number of pseudodata
        seed=0,  # random seed for instantiation of the method (for reproducibility)
        compute_weights_entropy=True,  # compute the entropy of weights distribution used in importance sampling
        mc_samples=None,  # number of MC samples for computation of variational objectives and predictions on unseen data
        reset=False,  # reset variational parameters to initialization
        reset_interval=10,  # number of outer gradient steps between reinitializations
        learn_v=False,  # boolean indicating if the v vector is learnable
        f=lambda *x: x[0],  # transformation applied on the v vector
        distr_fn=categorical_fn,  # distribution of last nn layer
        dnm="MNIST",  # dataset name
        nc=10,  # number of classes (argument supported only for the psvi dataloader subclasses)
        init_dataset=None,  # populated when picking initializations from a disturbed version of the original datapoints
        parameterised=False,
        learn_z=False,  # optimize in the label space
        prune=False,  # apply prunning over coreset training
        prune_interval=None,  # corresponding number of outer updates for prunning
        prune_sizes=None,  # list with budgets for pruned coreset
        increment=False, # incremental learning setting
        increment_interval=None, # corresponding number of outer updates between incrementing with new learning task
        increment_sizes=None, # list of increasing coreset sizes after incrementally introducing new learning tasks 
        lr0alpha=1e-3,
        retrain_on_coreset=False, # retrain variational parameters only on coreset datapoints after extracting a coreset using joint optimizer on the PSVI ELBO
        device_id=None,
        **kwargs,
    ):
        np.random.seed(seed), torch.manual_seed(seed)
        self.device = torch.device( f"cuda:{device_id}" if device_id else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.u, self.z = u, z
        self.train_dataset, self.test_dataset = (
            train_dataset,
            test_dataset,
        )
        self.N, self.D, self.dnm = N, D, dnm
        self.nc = nc  # number of classes
        self.distr_fn = distr_fn
        (
            self.model,
            self.optim,
            self.optim_u,
            self.optim_net,
            self.optim_v,
            self.optim_z,
        ) = (
            model,
            optim,
            optim_u,
            optim_net,
            optim_v,
            optim_z,
        )
        self.register_elbos, self.compute_weights_entropy = (
            register_elbos,
            compute_weights_entropy,
        )
        self.elbos = []
        self.num_pseudo, self.mc_samples = num_pseudo if not increment else increment_sizes[0], mc_samples
        self.reset, self.reset_interval, self.learn_v, self.learn_z = (
            reset,
            reset_interval,
            learn_v,
            learn_z,
        )
        with torch.no_grad():
            self.v = (
                1.0 / self.num_pseudo * torch.ones(self.num_pseudo, device=self.device)
            )
        self.v.requires_grad_(
            self.learn_v
        )  # initialize weights of coreset pseudodata to uniform and set to differentiable or not according to attribute learn_v
        self.f, self.parameterised = f, parameterised
        self.init_dataset = init_dataset
        self.results = {}
        self.prune, self.prune_interval, self.prune_sizes = (
            prune,
            prune_interval,
            prune_sizes,
        )
        self.increment, self.increment_interval, self.increment_sizes = (
            increment,
            increment_interval,
            increment_sizes,
        )
        if self.increment:
            self.historical_coresets = []
        self.lr0alpha = lr0alpha
        self.retrain_on_coreset = retrain_on_coreset

    def pseudo_subsample_init(self):
        r"""
        Initialization of pseudodata on random data subset with equal number of datapoints from each class
        """
        chosen_dataset = (
            self.train_dataset if self.init_dataset is None else self.init_dataset
        )
        # set up pseudodata by initializing to random subset from the existing dataset
        points_per_class = [
            self.num_pseudo // self.nc
        ] * self.nc  # split equally among classes
        points_per_class[-1] = self.num_pseudo - sum(
            points_per_class[:-1]
        )  # assigning the remainder to the last class
        with torch.no_grad():
            self.z = (
                torch.tensor(
                    [
                        item
                        for sublist in [
                            [i] * ppc for i, ppc in enumerate(points_per_class)
                        ]
                        for item in sublist
                    ]
                )
                .float()
                .to(self.device, non_blocking=True)
            )
            if self.learn_z:
                self.z = torch.nn.functional.one_hot(
                    self.z.to(torch.int64),
                    num_classes=self.nc,
                ).float()  # initialize target logits close to one-hot-encoding [0,..., class, ..., 0]-vectors
                self.z.requires_grad_(True)

        def get_x_from_label(ipc, _l):
            indices = (
                torch.as_tensor(chosen_dataset.targets).clone().detach() == _l
            ).nonzero()
            return torch.utils.data.DataLoader(
                SubsetPreservingTransforms(
                    chosen_dataset,
                    indices=indices,
                    dnm=self.dnm,
                    dim=self.D,
                ),
                batch_size=ipc,
                shuffle=True,
            )

        distilled_lst = []
        for c in range(self.nc):
            u0 = next(iter(get_x_from_label(points_per_class[c], c)))
            distilled_lst.append(u0.to(device=self.device, non_blocking=True))
        self.u = torch.cat(distilled_lst).requires_grad_(True)

    def pseudo_rand_init(self, variance=1.):
        r"""
        Initialize on noisy means of the observed datapoints and random labels equally split among classes
        """
        # print(f"is leaf : {self.u.is_leaf}")
        self.u = (
            (compute_empirical_mean(self.train_loader) + variance * torch.randn(self.num_pseudo, self.D))
            .clone()
        ).to(self.device).requires_grad_(True)
        self.z = torch.Tensor([])
        for c in range(self.nc):
            self.z = torch.cat(
                (
                    self.z.to(self.device),
                    c
                    * torch.ones(
                        self.num_pseudo // self.nc
                        if c < self.nc - 1
                        else self.num_pseudo - (self.nc - 1) * (self.num_pseudo // self.nc)
                    ).to(self.device),
                )
            )

    def psvi_elbo(self, xbatch, ybatch, model=None, params=None, hyperopt=False):
        r"""
        PSVI objective computation [negative PSVI-ELBO]
        """
        assert self.mc_samples > 1
        Nu, Nx = self.u.shape[0], xbatch.shape[0]
        all_data, all_labels = torch.cat((self.u, xbatch)), torch.cat(
            (
                self.z,
                ybatch
                if not self.learn_z
                else self.nc
                * torch.nn.functional.one_hot(
                    ybatch.to(torch.int64),
                    num_classes=self.nc,
                ).float(),
            )
        )
        logits = model(all_data) if not hyperopt else model(all_data, params=params)
        log_probs = (nn.LogSoftmax(dim=-1)(logits)).permute(1, 2, 0)
        all_nlls = (
            -self.distr_fn(logits=logits.squeeze(-1)).log_prob(all_labels)
            if not self.learn_z
            else torch.nn.KLDivLoss(reduction="none")(
                log_probs,
                all_labels.softmax(0).unsqueeze(-1).expand(log_probs.shape),
            )
            .sum(1)
            .T
        )
        pseudo_nll = (
            all_nlls[:, :Nu].matmul(self.N * self.f(self.v, 0)) if Nu > 0 else 0.0
        )
        data_nll = self.N / Nx * all_nlls[:, Nu:].sum(-1)
        sampled_nkl = sum(
            m.sampled_nkl()
            for m in model.modules()
            if (isinstance(m, VILinear) or isinstance(m, VILinearMultivariateNormal))
        )
        log_weights = -pseudo_nll + sampled_nkl
        weights = log_weights.softmax(0)
        return weights.mul(data_nll - pseudo_nll).sum() - log_weights.mean()

    def inner_elbo(self, model=None, params=None, hyperopt=False):
        r"""
        Inner VI objective computation [negative ELBO]
        """
        logits = model(self.u) if not hyperopt else model(self.u, params=params)
        if len(logits.shape)==2:
            logits.unsqueeze_(1)
        log_probs = (nn.LogSoftmax(dim=-1)(logits)).permute(1, 2, 0)
        pseudodata_nll = (
            -self.distr_fn(logits=logits.squeeze(-1)).log_prob(self.z)
            if not self.learn_z
            else torch.nn.KLDivLoss(reduction="none")(
                log_probs,
                self.z.softmax(0).unsqueeze(-1).expand(log_probs.shape),
            )
            .sum(1)
            .T
        ).matmul(self.N * self.f(self.v, 0))
        kl = sum(
            m.kl()
            for m in model.modules()
            if (isinstance(m, VILinear) or isinstance(m, VILinearMultivariateNormal))
        )
        return pseudodata_nll.sum() + kl if self.u.shape[0] > 0 else kl

    r"""
    Optimization methods
    """

    def joint_step(self, xbatch, ybatch):
        self.optim.zero_grad()
        loss = self.psvi_elbo(xbatch, ybatch, model=self.model)
        with torch.no_grad():
            if self.register_elbos:
                self.elbos.append((2, -loss.item()))
        loss.backward()
        self.optim.step()
        return loss

    def alternating_step(self, xbatch, ybatch):
        for i in range(2):
            self.optim = self.optim_net if i == 0 else self.optim_u
            self.optim.zero_grad()
            loss = self.psvi_elbo(xbatch, ybatch, model=self.model)
            with torch.no_grad():
                if self.register_elbos:
                    self.elbos.append(
                        (1, -loss.item())
                    ) if i == 1 else self.elbos.append((0, -loss.item()))
            loss.backward()
            self.optim.step()
        return loss

    def nested_step(self, xbatch, ybatch, truncated=False, K=5):
        self.optim_u.zero_grad()
        self.optim_net.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        if self.learn_z:
            self.optim_z.zero_grad()
        if not truncated:
            with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
                for in_it in range(self.inner_it):
                    mfvi_loss = self.inner_elbo(model=fmodel)
                    with torch.no_grad():
                        if self.register_elbos and in_it % self.log_every == 0:
                            self.elbos.append((1, -mfvi_loss.item()))
                    diffopt.step(mfvi_loss)
                psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
                with torch.no_grad():
                    if self.register_elbos:
                        self.elbos.append((0, -psvi_loss.item()))
                psvi_loss.backward()
        else:
            inner_opt = torch.optim.Adam(list(self.model.parameters()), 1e-4)
            for in_it in range(self.inner_it - K):
                mfvi_loss = self.inner_elbo(model=self.model)
                with torch.no_grad():
                    if self.register_elbos and in_it % self.log_every == 0:
                        self.elbos.append((1, -mfvi_loss.item()))
                mfvi_loss.backward()
                inner_opt.step()
            print('done non-differentiable part')
            inner_opt.zero_grad()
            with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
                for in_it in range(K):
                    mfvi_loss = self.inner_elbo(model=fmodel)
                    with torch.no_grad():
                        if self.register_elbos and in_it % self.log_every == 0:
                            self.elbos.append((1, -mfvi_loss.item()))
                    diffopt.step(mfvi_loss)
                psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
                with torch.no_grad():
                    if self.register_elbos:
                        self.elbos.append((0, -psvi_loss.item()))
                psvi_loss.backward()
        self.optim_u.step()
        if self.learn_v:
            self.optim_v.step()
            if not self.parameterised:
                with torch.no_grad():
                    torch.clamp_(
                        self.v, min=0.0
                    )  # clamp weights of coreset data point to be non-negative
        if self.scheduler_optim_net:
            self.scheduler_optim_net.step()
        if self.learn_z:
            self.optim_z.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(list(fmodel.parameters())),
            self.model.parameters(),
        )
        return psvi_loss

    def hyper_step(
        self,
        xbatch,
        ybatch,
        T=50,  # iterations for inner problem solver
        inner_opt_class=DifferentiableAdam,  # optimizer type for inner problem solver
        K=30,  # iterations for linear system solver (in approximate implicit differentiation methods)
        linsys_lr=1e-4,  # lr for the SGD optimizer used to solve the linear system on the Jacobian-vector products
        hypergrad_approx="CG_normaleq",
        **kwargs,
    ):
        T = self.inner_it
        inner_opt_kwargs = {"step_size": self.optim_net.param_groups[0]["lr"]}
        fmodel = monkeypatch(self.model, copy_initial_weights=True)
        self.optim_u.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        if self.learn_z:
            raise NotImplementedError

        def inner_loop(hparams, params, optim, n_steps, create_graph=False):
            params_history = [optim.get_opt_params(params)]
            for _ in range(n_steps):
                params_history.append(
                    optim(params_history[-1], hparams, create_graph=create_graph)
                )
            return params_history

        def get_inner_opt(train_loss):
            return inner_opt_class(train_loss, **inner_opt_kwargs)

        def inner_loss_function(p, hp, hyperopt=True):
            if self.learn_v:
                self.u, self.v = hp[0], hp[1]
            else:
                self.u = hp[0]
            return self.inner_elbo(model=fmodel, params=p, hyperopt=hyperopt)

        def outer_loss_function(p, hp):
            if self.learn_v:
                self.u, self.v = hp[0], hp[1]
            else:
                self.u = hp[0]
            return self.psvi_elbo(xbatch, ybatch, model=fmodel, params=p, hyperopt=True)

        inner_opt = get_inner_opt(inner_loss_function)
        params = [p.detach().clone().requires_grad_(True) for p in fmodel.parameters()]
        params_history = inner_loop(
            [self.u] + [self.v] if self.learn_v else [self.u],
            params,
            inner_opt,
            T,
        )
        last_param = params_history[-1][: len(params)]
        linear_opt = GradientDescent(loss_f=inner_loss_function, step_size=linsys_lr)

        if hypergrad_approx == "fixed_point":  # fixed-point AID
            fixed_point(
                last_param,
                [self.u] + [self.v] if self.learn_v else [self.u],
                K=K,
                fp_map=linear_opt,
                outer_loss=outer_loss_function,
                stochastic=True,
            )
        elif hypergrad_approx == "CG_normaleq":  # CG on normal equations AID
            CG_normaleq(
                last_param,
                [self.u] + [self.v] if self.learn_v else [self.u],
                K=K,
                fp_map=linear_opt,
                outer_loss=outer_loss_function,
                set_grad=True,
            )
        self.optim_u.step()
        if self.learn_v:
            self.optim_v.step()
            if not self.parameterised:
                with torch.no_grad():
                    torch.clamp_(self.v, min=0.0)
        ll = outer_loss_function(last_param, [self.u] + [self.v])
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(last_param),
            self.model.parameters(),
        )
        return ll.item()

    def set_up_model(self):
        r"""
        Specify the statistical model
        """
        print("SETTING UP THE MODEL \n\n")
        if self.logistic_regression:
            self.model = nn.Sequential(
                VILinear(
                    self.D, self.nc, init_sd=self.init_sd, mc_samples=self.mc_samples
                ),
            ).to(self.device)
        elif self.architecture=="logistic_regression_fullcov":
            self.model = nn.Sequential(
                VILinearMultivariateNormal(
                    self.D, self.nc, init_sd=self.init_sd, mc_samples=self.mc_samples
                ),
            ).to(self.device)
        elif self.architecture in {"fn", "residual_fn"}:
            self.model = make_fcnet(
                self.D,
                self.n_hidden,
                self.nc,
                n_layers=self.n_layers,
                linear_class=VILinear,
                nonl_class=nn.ReLU,
                mc_samples=self.mc_samples,
                residual=(self.architecture == "residual_fn"),
                init_sd=self.init_sd,
            ).to(self.device)
        elif self.architecture == "fn2":
            print(f"architecture : {self.architecture}")
            self.model = make_fc2net(
                self.D,
                self.n_hidden,
                self.nc,  # does not support argument on the number of channels
                linear_class=VILinearMultivariateNormal,
                nonl_class=nn.ReLU,
                mc_samples=self.mc_samples,
                init_sd=self.init_sd,
            ).to(self.device)
        elif self.architecture == "lenet":
            self.model = make_lenet(
                linear_class=VILinear,
                nonl_class=nn.ReLU,
                mc_samples=self.mc_samples,
                init_sd=self.init_sd,
            ).to(self.device)
        elif self.architecture == "alexnet":
            self.model = make_alexnet(
                linear_class=VILinear,
                nonl_class=nn.ReLU,
                mc_samples=self.mc_samples,
                init_sd=self.init_sd,
            ).to(self.device)
        elif self.architecture == "regressor_net":
            self.model =  make_regressor_net(
                self.D,
                self.n_hidden,
                self.nc,
                linear_class=VILinear,
                nonl_class=nn.ReLU,
                mc_samples=self.mc_samples,
                residual=(self.architecture == "residual_fn"),
                init_sd=self.init_sd,
            ).to(self.device)         


    def run_psvi(
        self,
        init_args="subsample",
        trainer="nested",
        n_layers=1,
        logistic_regression=True,
        n_hidden=None,
        architecture=None,
        log_every=10,
        inner_it=10,
        data_minibatch=None,
        lr0net=1e-3,
        lr0u=1e-3,
        lr0joint=1e-3,
        lr0v=1e-2,
        lr0z=1e-2,
        init_sd=1e-3,
        num_epochs=1000,
        log_pseudodata=False,
        prune_idx=0,
        increment_idx=0,
        gamma=1.0,
        **kwargs,
    ):
        r"""
        Run inference
        """
        # experiment-specific hyperparameters
        self.init_args = init_args
        self.trainer = trainer
        self.logistic_regression = logistic_regression
        self.architecture, self.n_hidden, self.n_layers, self.init_sd = (
            architecture,
            n_hidden,
            n_layers,
            init_sd,
        )
        self.log_every, self.log_pseudodata = log_every, log_pseudodata
        self.data_minibatch = data_minibatch
        self.inner_it, self.num_epochs = inner_it, num_epochs
        self.scheduler_optim_net = None
        self.gamma = gamma
        epoch_quarter = (self.N // self.data_minibatch) // 4
        scheduler_kwargs = {
            "step_size": epoch_quarter if epoch_quarter > 0 else 10000,
            "gamma": self.gamma,
        }

        # load the training and test data on dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=True, 
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=False,  
        )

        # setup for training and test sets in incremental learning: we start with 2 classes, and keep adding 1 new class at a time
        if self.increment:
            self.incremental_train_datasets, self.incremental_test_datasets = [None]*(self.nc - 1), [None]*(self.nc - 1)
            for c in range(1, self.nc):
                self.incremental_train_datasets[c-1] = self.train_dataset.subset_where(cs=list(range(c+1)) if c==1 else [c])
                self.incremental_test_datasets[c-1] = self.test_dataset.subset_where(cs=list(range(c+1)))
            (self.train_loader, self.test_loader) = (make_dataloader(self.incremental_train_datasets[0], self.data_minibatch), 
                                                    make_dataloader(self.incremental_test_datasets[0], self.data_minibatch, shuffle=False))
            self.train_data_so_far = len(self.train_loader.dataset)
            self.nc = 2 # in the incremental learning case start with a 2-class classification problem
        self.set_up_model()

        # initialization of results data structures
        (
            nlls_psvi,
            accs_psvi,
            core_idcs_psvi,
            iws_entropy,
            nesses,
            vs_entropy,
            us,
            zs,
            vs,
            grid_preds,
            times,
        ) = ([], [], [], [], [], [], [], [], [], [], [0])


        # initialization of pseudodata
        pseudodata_init = {
            "random": self.pseudo_rand_init,  # different transformations applied on `train_dataset`
            "subsample": self.pseudo_subsample_init,
        }
        pseudodata_init[self.init_args]()
        # optimization method
        self.optim_net, self.optim_u = (
            torch.optim.Adam(list(self.model.parameters()), lr0net),
            torch.optim.Adam([self.u], lr0u),
        )
        self.scheduler_optim_net = torch.optim.lr_scheduler.StepLR(
            self.optim_net, **scheduler_kwargs
        )
        if self.learn_v:
            self.optim_v = torch.optim.Adam([self.v], lr0v)
        if self.learn_z:
            self.optim_z = torch.optim.Adam([self.z], lr0z)
        optimizers = {
            "alternating": self.alternating_step,
            "nested": self.nested_step,
            "hyper": self.hyper_step,
        }
        if self.trainer == "joint":
            variational_params = (
                list(self.model.parameters()) + [self.u] + [self.v]
                if self.learn_v
                else list(self.model.parameters()) + [self.u]
            )
            self.optim = torch.optim.Adam(variational_params, lr0joint)
            psvi_step = self.joint_step
        else:
            psvi_step = optimizers[self.trainer]
        t_start = time.time()
        # training loop
        total_checkpts = list(range(self.num_epochs))[::log_every]
        downsample = 1 # downsample checkpoints for logging predictive uncertainty over a grid
        lpit = total_checkpts[::downsample]
        for it in tqdm(range(self.num_epochs)):
            xbatch, ybatch = next(iter(self.train_loader))
            xbatch, ybatch = xbatch.to(self.device, non_blocking=True), ybatch.to(
                self.device, non_blocking=True
            )
            # evaluation
            if it % self.log_every == 0:
                test_acc, test_nll, iw_ent, ness, v_ent = self.evaluate()
                if (
                    self.log_pseudodata
                    and it in lpit
                    and self.dnm not in {"MNIST", "Cifar10", "adult", "phishing", "webspam"}
                ):
                    print(f"\nlogging predictive grid at {it}")
                    grid_preds.append(self.pred_on_grid().detach().cpu().numpy().T)
                with torch.no_grad():
                    nlls_psvi.append(test_nll.item())
                    accs_psvi.append(test_acc.item())
                    print(f"\npredictive accuracy: {(100*test_acc.item()):.2f}%")
                    core_idcs_psvi.append(self.num_pseudo)
                    times.append(times[-1] + time.time() - t_start)
                    vs.append((self.f(self.v, 0)).clone().cpu().detach().numpy())
                    if iw_ent is not None:
                        iws_entropy.append(iw_ent.item())
                    if ness is not None:
                        nesses.append(ness.item())
                    if v_ent is not None:
                        vs_entropy.append(v_ent.item())
                    if self.log_pseudodata:
                        us.append(self.u.clone().cpu().detach().numpy())
                        zs.append(self.z.clone().cpu().detach().numpy())

            # variational nn reinitialization
            if self.reset and it % self.reset_interval == 0:
                self.weight_reset()

            # take a single optimization step
            psvi_step(xbatch, ybatch)

            # prune coreset to smaller sizes
            if self.prune and it > 0 and it % self.prune_interval == 0:
                if prune_idx < len(self.prune_sizes):
                    self.prune_coreset(
                        to_size=self.prune_sizes[prune_idx], lr0v=lr0v, lr0net=lr0net
                    )
                    prune_idx += 1
                    self.weight_reset()
                    # reset model upon pruning

            # add new learning task and increment coreset to enable fitting it
            if self.increment and it > 0 and it % self.increment_interval == 0:
                if increment_idx < len(self.increment_sizes)-1:
                    # self.historical_coresets.append({'v': self.v, 'u':self.u, 'z':self.z})
                    increment_idx += 1
                    #samples_from_coresets = [torch.multinomial(self.f(self.historical_coresets[_i]['v'], 0), self.train_data_so_far//increment_idx, replacement=True) for _i in range(increment_idx)] # sample summarising data from tasks so far using coreset points weighting
                    samples_from_coreset = torch.multinomial(self.f(self.v, 0), self.train_data_so_far, replacement=True) # sample summarising data from tasks so far using coreset points weighting
                    self.nc += 1 # added new class in training dataset
                    self.set_up_model() # reset model 
                    self.increment_coreset(
                        to_size=self.increment_sizes[increment_idx], lr0v=lr0v, lr0u=lr0u, lr0net=lr0net, new_class=increment_idx+1, increment_idx=increment_idx
                    )
                    #self.train_loader = make_dataloader(self.incremental_train_datasets[increment_idx].concatenate(torch.cat([self.historical_coresets[_i]['u'][samples_from_coresets[_i]].clone().detach() for _i in range(increment_idx)], axis=0), 
                    #                                                                                               torch.cat([self.historical_coressets[_i]['z'][samples_from_coresets[_i]].clone().detach() for _i in range(increment_idx)], axis=0)), 
                    #                                                                            self.data_minibatch) # augment with new training data
                    self.train_loader = make_dataloader(self.incremental_train_datasets[increment_idx].concatenate(self.u[samples_from_coreset].clone().detach(), 
                                                                                                                    self.z[samples_from_coreset].clone().detach()), 
                                                                                                self.data_minibatch) # augment with new training data

                    self.test_loader = make_dataloader(self.incremental_test_datasets[increment_idx], self.data_minibatch, shuffle=False) # augment with new test data
                    self.train_data_so_far = len(self.train_loader.dataset)

        # retrain restricting only on coreset datapoints
        if self.retrain_on_coreset:
            print("\n\nRetrain on the extracted coreset for the same number of epochs")
            self.weight_reset()
            self.optim_retrain =  torch.optim.Adam(list(self.model.parameters()), lr0joint)
            for it in tqdm(range(self.num_epochs)):
                # evaluation
                if it % self.log_every == 0:
                    test_acc, test_nll, iw_ent, ness, v_ent = self.evaluate(correction=False)
                    if (
                        self.log_pseudodata
                        and it in lpit
                        and self.dnm not in {"MNIST", "Cifar10", "adult", "phishing", "webspam"}
                    ):
                        print(f"\nlogging predictive grid at {it}")
                        grid_preds.append(self.pred_on_grid(correction=False).detach().cpu().numpy().T)
                    with torch.no_grad():
                        nlls_psvi.append(test_nll.item())
                        accs_psvi.append(test_acc.item())
                        print(f"\npredictive accuracy: {(100*test_acc.item()):.2f}%")
                        core_idcs_psvi.append(self.num_pseudo)
                        times.append(times[-1] + time.time() - t_start)
                        vs.append((self.f(self.v, 0)).clone().cpu().detach().numpy())
                        if iw_ent is not None:
                            iws_entropy.append(iw_ent.item())
                        if ness is not None:
                            nesses.append(ness.item())
                        if v_ent is not None:
                            vs_entropy.append(v_ent.item())
                        if self.log_pseudodata:
                            us.append(self.u.clone().cpu().detach().numpy())
                            zs.append(self.z.clone().cpu().detach().numpy())

                self.optim_retrain.zero_grad()
                loss = self.inner_elbo(model=self.model)
                loss.backward()
                self.optim_retrain.step()

        # store results
        self.results["accs"] = accs_psvi
        self.results["nlls"] = nlls_psvi
        self.results["csizes"] = core_idcs_psvi
        self.results["times"] = times[1:]
        self.results["elbos"] = self.elbos
        self.results["went"] = iws_entropy
        self.results["ness"] = nesses
        self.results["vent"] = vs_entropy
        self.results["vs"] = vs

        if self.log_pseudodata:
            self.results["us"], self.results["zs"], self.results["grid_preds"] = (
                us,
                zs,
                grid_preds,
            )
        return self.results

    ## Compute predictive metrics
    def evaluate(
        self,
        correction=True,
        **kwargs,
    ):
        assert self.mc_samples > 1
        total, test_nll, corrects = 0, 0, 0
        for xt, yt in self.test_loader:
            xt, yt = xt.to(self.device, non_blocking=True), yt.to(
                self.device, non_blocking=True
            )
            with torch.no_grad():
                all_data = torch.cat((self.u, xt))
                all_logits = self.model(all_data)
                pseudo_logits = all_logits[:, : self.num_pseudo]
                log_probs = (nn.LogSoftmax(dim=-1)(pseudo_logits)).permute(1, 2, 0)
                pseudo_nll = (
                    (
                        (
                            self.distr_fn(logits=pseudo_logits).log_prob(self.z)
                            if not self.learn_z
                            else torch.nn.KLDivLoss(reduction="none")(
                                log_probs,
                                self.z.softmax(0).unsqueeze(-1).expand(log_probs.shape),
                            ).sum((1, 2))
                        ).matmul(self.N * self.f(self.v, 0))
                    )
                    if self.num_pseudo > 0
                    else 0.0
                )
                test_data_logits = all_logits[:, self.num_pseudo :]
                sampled_nkl = sum(
                    m.sampled_nkl()
                    for m in self.model.modules()
                    if (
                        isinstance(m, VILinear)
                        or isinstance(m, VILinearMultivariateNormal)
                    )
                )
                log_weights = -pseudo_nll + sampled_nkl
                weights = log_weights.softmax(0)
                test_probs = (
                    (
                        test_data_logits.softmax(-1)
                        .mul(weights.unsqueeze(-1).unsqueeze(-1))
                        .sum(0)
                    )
                    if correction
                    else test_data_logits.softmax(-1).mean(0)
                )
                corrects += test_probs.argmax(-1).float().eq(yt).float().sum()
                total += yt.size(0)
                test_nll += -self.distr_fn(probs=test_probs).log_prob(yt).sum()

        iw_entropy = (
            -weights[weights > 0].log().mul(weights[weights > 0]).sum()
            if self.compute_weights_entropy
            else None
        )  # entropy of the importance weighting distribution
        ness = (
            weights.sum().square() / weights.square().sum() / weights.shape[0]
        )  # normalized effective sample size

        vs = self.f(self.v, 0)
        v_entropy = (
            vs.sum().square()
            / vs.square().sum()
            / self.num_pseudo  # normalize entropy with coreset size
            if self.compute_weights_entropy
            else None
        )
        return (
            corrects / float(total),
            test_nll / float(total),
            iw_entropy,
            ness,
            v_entropy,
        )

    def weight_reset(self):
        r"""
        Reset variational parameters to initialization
        """
        for layer in self.model.modules():
            if (
                isinstance(layer, VILinear)
                or isinstance(layer, VILinearMultivariateNormal)
            ) and hasattr(layer, "reset_parameters_variational"):
                layer.reset_parameters_variational()
            elif (
                isinstance(layer, nn.Conv2d)
                or (
                    isinstance(layer, VILinear)
                    or isinstance(layer, VILinearMultivariateNormal)
                )
                and hasattr(layer, "reset_parameters")
            ):
                layer.reset_parameters()

    def pred_on_grid(
        self,
        n_test_per_dim=250,
        correction=True,
        **kwargs,
    ):
        r"""
        Predictions over a 2-d grid for visualization of predictive posterior on 2-d synthetic datasets
        """
        _x0_test = torch.linspace(-3, 4, n_test_per_dim)
        _x1_test = torch.linspace(-2, 3, n_test_per_dim)
        x_test = torch.stack(torch.meshgrid(_x0_test, _x1_test), dim=-1).to(self.device)

        with torch.no_grad():
            all_data = torch.cat((self.u, x_test.view(-1, 2)))
            all_logits = self.model(all_data).squeeze(-1)
            pseudo_nll = (
                (
                    self.distr_fn(logits=all_logits[:, : self.num_pseudo])
                    .log_prob(self.z)
                    .matmul(self.N * self.f(self.v, 0))
                )
                if self.num_pseudo > 0
                else 0.0
            )
            grid_data_logits = all_logits[:, self.num_pseudo :]
            sampled_nkl = sum(
                m.sampled_nkl()
                for m in self.model.modules()
                if (
                    isinstance(m, VILinear) or isinstance(m, VILinearMultivariateNormal)
                )
            )

            log_weights = -pseudo_nll + sampled_nkl
            weights = log_weights.softmax(0)
            grid_probs = (
                (
                    grid_data_logits.softmax(-1)
                    .mul(weights.unsqueeze(-1).unsqueeze(-1))
                    .sum(0)
                )
                if correction
                else grid_data_logits.softmax(-1).mean(0)
            )
        return grid_probs

    def prune_coreset(
        self,
        to_size,
        lr0v=1e-3,
        lr0net=1e-4,
    ):  # designed to work only for the fixed u methods
        r"""
        Prune coreset to a given smaller size
        """
        self.num_pseudo = to_size
        keep_v = torch.multinomial(self.f(self.v, 0), to_size, replacement=False) # torch.topk(self.v, to_size)
        self.v = torch.zeros_like(self.v[keep_v]).clone().detach().requires_grad_(True)
        self.optim_v = torch.optim.Adam([self.v], lr0v)
        self.u = torch.index_select(self.u, 0, keep_v)
        self.z = torch.index_select(self.z, 0, keep_v)
        self.optim_net = torch.optim.Adam(list(self.model.parameters()), lr0net)

    def increment_coreset(
        self,
        to_size,
        lr0v=1e-3,
        lr0u=1e-3,
        lr0net=1e-4,
        variance=1., # variance for random initialization of coreset for new class 
        new_class=2,
        increment_idx=1,
    ):  
        r"""
        Increment coreset to a given larger size
        """
        self.num_pseudo, num_extra_points = to_size, to_size - len(self.v)
        extra_weights = torch.ones(num_extra_points, device=self.device)
        self.v = torch.cat(( self.v, 1. / (len(self.v) + num_extra_points) * self.v.sum() * extra_weights )).detach().requires_grad_(True)
        self.optim_v = torch.optim.Adam([self.v], lr0v)
        (new_us, new_zs) = (
            ((compute_empirical_mean(self.train_loader) + variance * torch.randn(num_extra_points, self.D)).clone(), new_class * torch.ones(num_extra_points)) 
                if self.init_args == "random"  
                else self.incremental_train_datasets[increment_idx][torch.randperm(len(self.incremental_train_datasets[increment_idx]))[:num_extra_points]])
        self.u, self.z = torch.cat((self.u, new_us)).detach().requires_grad_(True), torch.cat((self.z, new_zs))
        self.optim_u = torch.optim.Adam([self.u], lr0u)
        self.optim_net = torch.optim.Adam(list(self.model.parameters()), lr0net)


class PSVILearnV(PSVI):
    r"""
    PSVI 
        - with learnable v on a simplex (with constant sum constraint)
    """

    def __init__(self, learn_v=True, parameterised=True, **kwargs):
        super().__init__(**kwargs)
        self.learn_v, self.parameterised = learn_v, parameterised
        with torch.no_grad():
            self.v = torch.zeros(self.num_pseudo, device=self.device)
        self.v.requires_grad_(
            True
        )  # initialize learnable weights of coreset pseudodata to uniform
        self.f = (
            torch.softmax
        )  # transform v via softmax to keep the sum over the pseudodata fixed


class PSVI_No_Rescaling(PSVI):
    r"""
    PSVI 
        - with no fixed or learnable coefficients on coreset datapoints whatsoever
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v *= (
            1.0 / self.N
        )  # we remove log-likelihood rescaling dependency on the true dataset size N


class PSVIFreeV(PSVI):
    r"""
    PSVI 
        - with learnable v (subject only to non-negativity constraints)
    """

    def __init__(self, learn_v=True, **kwargs):
        super().__init__(**kwargs)
        self.learn_v = True
        self.v.requires_grad_(True)


class PSVI_Ablated(PSVILearnV):
    r"""
    PSVI 
        - with ablated importance sampling from coreset variational posterior
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def psvi_elbo(self, xbatch, ybatch, model=None, params=None, hyperopt=False):
        r"""
        Ablated PSVI objective computation
        """
        Nx = xbatch.shape[0]
        logits = model(xbatch) if not hyperopt else model(xbatch, params=params)
        nlls = -self.distr_fn(logits=logits.squeeze(-1)).log_prob(ybatch)
        data_nll = self.N / Nx * nlls.sum(-1)  # multi-sample training
        sampled_nkl = sum(
            m.sampled_nkl() for m in model.modules() if isinstance(m, VILinear)
        )
        return data_nll.mean() - sampled_nkl.mean()


class PSVI_No_IW(PSVI_Ablated):
    r"""
    PSVI 
        - with single-sample training / multi-sample testing
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mc_samples = 1

    def evaluate(
        self,
        correction=True,
        mc_samples_eval=5,
        mc_samples_train=1,
        **kwargs,
    ):
        r"""
        Compute predictive metrics
        """
        with torch.no_grad():
            self.mc_samples = mc_samples_eval
            set_mc_samples(
                self.model, self.mc_samples
            )  # set to multi-sample for testing
            test_acc, test_nll, iw_entropy, ness, v_entropy = super().evaluate(
                correction=True,
                **kwargs,
            )
            self.mc_samples = 1
            set_mc_samples(
                self.model, mc_samples_train
            )  # set to single-sample for training
        return test_acc, test_nll, iw_entropy, ness, v_entropy

    def pred_on_grid(
        self,
        correction=True,
        n_test_per_dim=250,
        mc_samples_eval=5,
        mc_samples_train=1,
        **kwargs,
    ):
        r"""
        Predictions over a 2-d grid for visualization of predictive posterior on 2-d synthetic datasets
        """
        # TODO: fix for correction via importance weighting 
        with torch.no_grad():
            self.mc_samples = mc_samples_eval
            set_mc_samples(
                self.model, self.mc_samples
            )  # set to multi-sample for testing
            test_probs = super().pred_on_grid(
                correction=correction,
                n_test_per_dim=n_test_per_dim,
                **kwargs,
            )
            self.mc_samples = mc_samples_train
            set_mc_samples(
                self.model, mc_samples_train
            )  # set to single-sample for training
        return test_probs


class PSVIAV(PSVILearnV):
    r"""
    PSVI subclass with
       - learnable coreset point weights on a simplex,
       - learnable rescaling of total coreset evidence
    """

    def __init__(self, learn_v=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.tensor([0.0], device=self.device)
        self.alpha.requires_grad_(True)
        self.f = lambda *x: (
            torch.exp(self.alpha) * torch.softmax(x[0], x[1])
        )  # transform v via softmax to keep the sum over the pseudodata fixed and multiply by a learnable non-negative coefficient
        self.optim_alpha = torch.optim.Adam([self.alpha], self.lr0alpha)
        self.results["alpha"] = []

    def evaluate(self, **kwargs):
        self.results["alpha"].append(
            self.alpha.clone()
            .cpu()
            .detach()
            .numpy()  # store the extra variational parameter
        )
        return super().evaluate(**kwargs)
    
    def increment_coreset(self, lr0alpha=1e-3, **kwargs):
        super().increment_coreset(**kwargs)
        self.optim_alpha = torch.optim.Adam([self.alpha], lr0alpha)

    def hyper_step(
        self,
        xbatch,
        ybatch,
        T=10,  # iterations for inner problem solver
        inner_opt_class=DifferentiableAdam,  # optimizer type for inner problem solver
        K=10,  # iterations for linear system solver (in approximate implicit differentiation methods)
        linsys_lr=1e-1,  # lr for the SGD optimizer used to solve the linear system on the Jacobian-vector products
        hypergrad_approx="CG_normaleq",
        **kwargs,
    ):
        T = self.inner_it
        inner_opt_kwargs = {"step_size": self.optim_net.param_groups[0]["lr"]}
        fmodel = monkeypatch(self.model, copy_initial_weights=True)
        self.optim_u.zero_grad()
        self.optim_v.zero_grad()
        self.optim_alpha.zero_grad()
        if self.optim_z:
            raise NotImplementedError

        def inner_loop(hparams, params, optim, n_steps, create_graph=False):
            params_history = [optim.get_opt_params(params)]
            for _ in range(n_steps):
                params_history.append(
                    optim(params_history[-1], hparams, create_graph=create_graph)
                )
            return params_history

        def get_inner_opt(train_loss):
            return inner_opt_class(train_loss, **inner_opt_kwargs)

        def inner_loss_function(p, hp, hyperopt=True):
            self.u, self.v, self.alpha = hp[0], hp[1], hp[2]
            return self.inner_elbo(model=fmodel, params=p, hyperopt=hyperopt)

        def outer_loss_function(p, hp):
            self.u, self.v, self.alpha = hp[0], hp[1], hp[2]
            return self.psvi_elbo(xbatch, ybatch, model=fmodel, params=p, hyperopt=True)

        inner_opt = get_inner_opt(inner_loss_function)
        params = [p.detach().clone().requires_grad_(True) for p in fmodel.parameters()]
        params_history = inner_loop(
            [self.u] + [self.v] + [self.alpha],
            params,
            inner_opt,
            T,
        )
        last_param = params_history[-1][: len(params)]
        linear_opt = GradientDescent(
            loss_f=inner_loss_function, step_size=linsys_lr
        )  # GradientDescent(loss_f=inner_loss_function, step_size=linsys_lr)

        if hypergrad_approx == "fixed_point":  # fixed-point AID
            fixed_point(
                last_param,
                [self.u] + [self.v] + [self.alpha],
                K=K,
                fp_map=linear_opt,
                outer_loss=outer_loss_function,
                stochastic=True,
            )
        elif hypergrad_approx == "CG_normaleq":  # CG on normal equations AID
            CG_normaleq(
                last_param,
                [self.u] + [self.v] + [self.alpha],
                K=K,
                fp_map=linear_opt,
                outer_loss=outer_loss_function,
                set_grad=True,
            )
        self.optim_u.step()
        if self.learn_v:
            self.optim_v.step()
            self.optim_alpha.step()

        ll = outer_loss_function(last_param, [self.u] + [self.v] + [self.alpha])
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(last_param),
            self.model.parameters(),
        )
        return ll.item()

    def nested_step(self, xbatch, ybatch):
        self.optim_u.zero_grad()
        self.optim_net.zero_grad()
        self.optim_alpha.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        if self.learn_z:
            self.optim_z.zero_grad()
        with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
            for in_it in range(self.inner_it):
                mfvi_loss = self.inner_elbo(model=fmodel)
                with torch.no_grad():
                    if self.register_elbos and in_it % self.log_every == 0:
                        self.elbos.append((1, -mfvi_loss.item()))
                diffopt.step(mfvi_loss)
            psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
            with torch.no_grad():
                if self.register_elbos:
                    self.elbos.append((0, -psvi_loss.item()))
            psvi_loss.backward()
        self.optim_u.step()
        if self.learn_v:
            self.optim_v.step()
            self.optim_alpha.step()
        if self.learn_z:
            self.optim_z.step()
        if self.scheduler_optim_net:
            self.scheduler_optim_net.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(list(fmodel.parameters())),
            self.model.parameters(),
        )
        return psvi_loss


class PSVIFixedU(PSVILearnV):
    r"""
    PSVI subclass
        - with fixed coreset point locations
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def nested_step(self, xbatch, ybatch):
        self.u.requires_grad_(False)
        self.optim_net.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
            for in_it in range(self.inner_it):
                mfvi_loss = self.inner_elbo(model=fmodel)
                with torch.no_grad():
                    if self.register_elbos and in_it % self.log_every == 0:
                        self.elbos.append((1, -mfvi_loss.item()))
                diffopt.step(mfvi_loss)
            psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
            with torch.no_grad():
                if self.register_elbos:
                    self.elbos.append((0, -psvi_loss.item()))
            psvi_loss.backward()
        if self.learn_v:
            self.optim_v.step()
        if self.scheduler_optim_net:
            self.scheduler_optim_net.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(list(fmodel.parameters())),
            self.model.parameters(),
        )
        return psvi_loss

    def hyper_step(
        self,
        xbatch,
        ybatch,
        T=20,  # iterations for inner problem solver
        inner_opt_class=DifferentiableAdam,  # optimizer type for inner problem solver
        K=20,  # iterations for linear system solver (in approximate implicit differentiation methods)
        linsys_lr=1e-3,  # lr for the SGD optimizer used to solve the linear system on the Jacobian-vector products
        hypergrad_approx="CG_normaleq",
        **kwargs,
    ):
        self.u.requires_grad_(False)
        T = self.inner_it
        inner_opt_kwargs = {"step_size": self.optim_net.param_groups[0]["lr"]}
        fmodel = monkeypatch(self.model, copy_initial_weights=True)
        if self.learn_v:
            self.optim_v.zero_grad()

        def inner_loop(hparams, params, optim, n_steps, create_graph=False):
            params_history = [optim.get_opt_params(params)]
            for _ in range(n_steps):
                params_history.append(
                    optim(params_history[-1], hparams, create_graph=create_graph)
                )
            return params_history

        def get_inner_opt(train_loss):
            return inner_opt_class(train_loss, **inner_opt_kwargs)

        def inner_loss_function(p, hp, hyperopt=True):
            if self.learn_v:
                self.v = hp[0]
            else:
                pass
            return self.inner_elbo(model=fmodel, params=p, hyperopt=hyperopt)

        def outer_loss_function(p, hp):
            if self.learn_v:
                self.v = hp[0]
            else:
                pass
            return self.psvi_elbo(xbatch, ybatch, model=fmodel, params=p, hyperopt=True)

        inner_opt = get_inner_opt(inner_loss_function)
        params = [p.detach().clone().requires_grad_(True) for p in fmodel.parameters()]
        params_history = inner_loop(
            [self.v] if self.learn_v else None,
            params,
            inner_opt,
            T,
        )
        last_param = params_history[-1][: len(params)]
        fp_map = DifferentiableAdam(
            loss_f=inner_loss_function, step_size=linsys_lr
        )  # GradientDescent(loss_f=inner_loss_function, step_size=linsys_lr)
        if hypergrad_approx == "fixed_point":  # fixed-point AID
            fixed_point(
                last_param,
                [self.v] if self.learn_v else None,
                K=K,
                fp_map=fp_map,
                outer_loss=outer_loss_function,
                stochastic=True,
            )
        elif hypergrad_approx == "CG_normaleq":  # CG on normal equations AID
            CG_normaleq(
                last_param,
                [self.v] if self.learn_v else None,
                K=K,
                fp_map=fp_map,
                outer_loss=outer_loss_function,
                set_grad=True,
            )
        if self.learn_v:
            self.optim_v.step()
        ll = outer_loss_function(last_param, [self.v])
        if self.scheduler_optim_net:
            self.scheduler_optim_net.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(last_param),
            self.model.parameters(),
        )
        return ll.item()


class PSVIAFixedU(PSVILearnV):
    r"""
    PSVI subclass with
        - fixed coreset point locations
        - learnable coreset weights on a simplex
        - learnable rescaling of total coreset evidence
    """

    def __init__(self, learn_v=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.tensor([0.0], device=self.device)
        self.alpha.requires_grad_(True)
        self.f = lambda *x: (
            torch.exp(self.alpha) * torch.softmax(x[0], x[1])
        )  # transform v via softmax to keep the sum over the pseudodata fixed and multiply by a learnable non-negative coefficient
        self.optim_alpha = torch.optim.Adam([self.alpha], self.lr0alpha)
        self.results["alpha"] = []

    def evaluate(self, **kwargs):
        self.results["alpha"].append(
            self.alpha.clone()
            .cpu()
            .detach()
            .numpy()  # store the extra variational parameter
        )
        return super().evaluate(**kwargs)

    def hyper_step(
        self,
        xbatch,
        ybatch,
        T=5,  # iterations for inner problem solver
        inner_opt_class=DifferentiableAdam,  # optimizer type for inner problem solver
        K=5,  # iterations for linear system solver (in approximate implicit differentiation methods)
        linsys_lr=1e-1,  # lr for the SGD optimizer used to solve the linear system on the Jacobian-vector products
        hypergrad_approx="CG_normaleq",
        **kwargs,
    ):
        T = self.inner_it
        inner_opt_kwargs = {"step_size": self.optim_net.param_groups[0]["lr"]}
        fmodel = monkeypatch(self.model, copy_initial_weights=True)
        self.optim_v.zero_grad()
        self.optim_alpha.zero_grad()
        self.u.requires_grad_(False)

        def inner_loop(hparams, params, optim, n_steps, create_graph=False):
            params_history = [optim.get_opt_params(params)]
            for _ in range(n_steps):
                params_history.append(
                    optim(params_history[-1], hparams, create_graph=create_graph)
                )
            return params_history

        def get_inner_opt(train_loss):
            return inner_opt_class(train_loss, **inner_opt_kwargs)

        def inner_loss_function(p, hp, hyperopt=True):
            self.v, self.alpha = hp[0], hp[1]
            return self.inner_elbo(model=fmodel, params=p, hyperopt=hyperopt)

        def outer_loss_function(p, hp):
            self.v, self.alpha = hp[0], hp[1]
            return self.psvi_elbo(xbatch, ybatch, model=fmodel, params=p, hyperopt=True)

        inner_opt = get_inner_opt(inner_loss_function)
        params = [p.detach().clone().requires_grad_(True) for p in fmodel.parameters()]
        params_history = inner_loop(
            [self.v] + [self.alpha],
            params,
            inner_opt,
            T,
        )
        last_param = params_history[-1][: len(params)]
        linear_opt = GradientDescent(loss_f=inner_loss_function, step_size=linsys_lr)

        if hypergrad_approx == "fixed_point":  # fixed-point AID
            fixed_point(
                last_param,
                [self.v] + [self.alpha],
                K=K,
                fp_map=linear_opt,
                outer_loss=outer_loss_function,
                stochastic=True,
            )
        elif hypergrad_approx == "CG_normaleq":  # CG on normal equations AID
            CG_normaleq(
                last_param,
                [self.v] + [self.alpha],
                K=K,
                fp_map=linear_opt,
                outer_loss=outer_loss_function,
                set_grad=True,
            )
        if self.learn_v:
            self.optim_v.step()
            self.optim_alpha.step()

        ll = outer_loss_function(last_param, [self.v] + [self.alpha])
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(last_param),
            self.model.parameters(),
        )
        return ll.item()

    def nested_step(self, xbatch, ybatch):
        self.optim_net.zero_grad()
        self.optim_alpha.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        self.u.requires_grad_(False)

        with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
            for in_it in range(self.inner_it):
                mfvi_loss = self.inner_elbo(model=fmodel)
                with torch.no_grad():
                    if self.register_elbos and in_it % self.log_every == 0:
                        self.elbos.append((1, -mfvi_loss.item()))
                diffopt.step(mfvi_loss)
            psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
            with torch.no_grad():
                if self.register_elbos:
                    self.elbos.append((0, -psvi_loss.item()))
            psvi_loss.backward()
        if self.learn_v:
            self.optim_v.step()
            self.optim_alpha.step()
        if self.scheduler_optim_net:
            self.scheduler_optim_net.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(list(fmodel.parameters())),
            self.model.parameters(),
        )
        return psvi_loss


## PSVI subclass supporting regression 
class PSVI_regressor(PSVI):
    def __init__(
        self,
        u=None,  # pseudo x-coordinates
        z=None,  # pseudo y-coordinates
        train_dataset=None,  # true training data
        val_dataset=None,
        test_dataset=None,  # test data
        y_mean=None,
        y_std=None,
        N=None,  # size of training data
        D=None,  # dimensionality of training data
        optim=None,  # joint variational model/pseudodata optimizer
        optim_u=None,  # optimizer for pseudodata
        optim_net=None,  # optimizer for variational model parameters
        optim_v=None,  # optimizer for log-likelihood rescaling vector
        optim_z=None,  # optimizer for outputs on distilled data
        register_elbos=False,  # register values of objectives over inference
        num_pseudo=None,  # number of pseudodata
        seed=0,  # random seed for instantiation of the method (for reproducibility)
        compute_weights_entropy=True,  # compute the entropy of weights distribution used in importance sampling
        mc_samples=None,  # number of MC samples for computation of variational objectives and predictions on unseen data
        learn_v=False,  # boolean indicating if the v vector is learnable
        f=lambda *x: x[0],  # transformation applied on the v vector
        dnm=None,  # dataset name
        nc=1,  # dimension of output space
        init_dataset=None,  # populated when picking initializations from a disturbed version of the original datapoints
        parameterised=False,
        learn_z=True,  # optimize in the label space
        lr0alpha=1e-3,
        tau=0.1,
        logistic_regression=False,
        **kwargs,
    ):
        np.random.seed(seed), torch.manual_seed(seed)
        print(f'device id {device_id} ')
        self.device = torch.device( f"cuda:{device_id}" if device_id else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.u, self.z = u, z
        self.train_dataset, self.val_dataset, self.test_dataset = (
            train_dataset,
            val_dataset,
            test_dataset,
        )
        self.logistic_regression = logistic_regression
        self.N, self.D, self.dnm = N, D, dnm
        self.nc = nc  # dimensionality of output
        self.distr_fn = partial(gaussian_fn, scale=1.0 / np.sqrt(tau))
        (self.optim, self.optim_u, self.optim_net, self.optim_v, self.optim_z,) = (
            optim,
            optim_u,
            optim_net,
            optim_v,
            optim_z,
        )
        self.register_elbos, self.compute_weights_entropy = (
            register_elbos,
            compute_weights_entropy,
        )
        if self.register_elbos:
            self.elbos = []
        self.num_pseudo, self.mc_samples = num_pseudo, mc_samples
        self.learn_v, self.learn_z = (
            learn_v,
            learn_z,
        )
        with torch.no_grad():
            self.v = (
                1.0 / self.num_pseudo * torch.ones(self.num_pseudo, device=self.device)
            )
        self.v.requires_grad_(
            self.learn_v
        )  # initialize weights of coreset pseudodata to uniform and set to differentiable or not according to attribute learn_v
        self.f, self.parameterised = f, parameterised
        self.init_dataset = init_dataset
        self.results = {}
        self.lr0alpha = lr0alpha
        self.y_mean, self.y_std = y_mean, y_std

    ### Initialization methods for the pseudodata
    def pseudo_subsample_init(self):
        sample_idcs = random.sample(range(len(self.train_dataset)), self.num_pseudo)
        subset_train_dataset = torch.utils.data.Subset(self.train_dataset, sample_idcs)
        self.cs_support = DataLoader(
            subset_train_dataset,
            batch_size=self.num_pseudo,
            # pin_memory=True,
            shuffle=False,
        )
        with torch.no_grad():
            self.u, self.z = next(iter(self.cs_support))
            self.u, self.z = self.u.to(self.device), self.z.to(self.device)
        self.u.requires_grad_(True), self.z.requires_grad_(True)

    ## PSVI objective computation [negative PSVI-ELBO]
    def psvi_elbo(self, xbatch, ybatch, model=None, params=None, hyperopt=False):
        assert self.mc_samples > 1
        Nu, Nx = self.u.shape[0], xbatch.shape[0]
        all_xs, all_ys = torch.cat((self.u, xbatch)), torch.cat((self.z, ybatch))
        all_nlls = -self.distr_fn(model(all_xs).squeeze(-1)).log_prob(all_ys.squeeze())
        pseudo_nll = (
            all_nlls[:, :Nu].matmul(self.N * self.f(self.v, 0)) if Nu > 0 else 0.0
        )
        data_nll = self.N / Nx * all_nlls[:, Nu:].sum(-1)
        sampled_nkl = sum(
            m.sampled_nkl() for m in model.modules() if isinstance(m, VILinear)
        )
        log_weights = -pseudo_nll + sampled_nkl
        weights = log_weights.softmax(0)
        return weights.mul(data_nll - pseudo_nll).sum() - log_weights.mean()

    ## Inner VI objective computation [negative ELBO]
    def inner_elbo(self, model=None, params=None, hyperopt=False):
        pseudodata_nll = (
            -self.distr_fn(model(self.u).squeeze(-1)).log_prob(self.z.squeeze())
        ).matmul(self.N * self.f(self.v, 0))
        kl = sum(m.kl() for m in model.modules() if isinstance(m, VILinear))
        return pseudodata_nll.sum() + kl if self.u.shape[0] > 0 else kl

    ## Optimization methods
    def nested_step(self, xbatch, ybatch):
        self.optim_u.zero_grad()
        self.optim_net.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        if self.learn_z:
            self.optim_z.zero_grad()
        with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
            for in_it in range(self.inner_it):
                mfvi_loss = self.inner_elbo(model=fmodel)
                with torch.no_grad():
                    if self.register_elbos and in_it % self.log_every == 0:
                        self.elbos.append((1, -mfvi_loss.item()))
                diffopt.step(mfvi_loss)
            psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
            with torch.no_grad():
                if self.register_elbos:
                    self.elbos.append((0, -psvi_loss.item()))
            psvi_loss.backward()
        self.optim_u.step()
        if self.learn_v:
            self.optim_v.step()
            if not self.parameterised:
                with torch.no_grad():
                    torch.clamp_(
                        self.v, min=0.0
                    )  # clamp weights of coreset data point to be non-negative
        if self.learn_z:
            self.optim_z.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(list(fmodel.parameters())),
            self.model.parameters(),
        )
        return psvi_loss

    ## Execution of inference
    def run_psvi(
        self,
        init_args="subsample",
        trainer="nested",
        n_layers=1,
        n_hidden=None,
        architecture=None,
        log_every=10,
        inner_it=10,
        data_minibatch=None,
        lr0net=1e-3,
        lr0u=1e-3,
        lr0v=1e-2,
        lr0z=1e-2,
        init_sd=1e-3,
        num_epochs=1000,
        log_pseudodata=False,
        **kwargs,
    ):
        # experiment-specific hyperparameters
        self.init_args = init_args
        self.trainer = trainer
        self.architecture, self.n_hidden, self.n_layers, self.init_sd = (
            architecture,
            n_hidden,
            n_layers,
            init_sd,
        )
        self.log_every, self.log_pseudodata = log_every, log_pseudodata
        self.data_minibatch = data_minibatch
        self.inner_it, self.num_epochs = inner_it, num_epochs
        self.set_up_model()

        # initialization of results data structures
        (
            lls_psvi,
            rmses_psvi,
            core_idcs_psvi,
            iws_entropy,
            nesses,
            vs_entropy,
            us,
            zs,
            vs,
            times,
        ) = ([], [], [], [], [], [], [], [], [], [0])

        # load the training and test data on dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=True,
        ) 
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=False,
        )

        # initialization of pseudodata
        pseudodata_init = {
            "subsample": self.pseudo_subsample_init,
        }
        pseudodata_init[self.init_args]()
        # optimization method
        self.optim_net, self.optim_u = (
            torch.optim.Adam(list(self.model.parameters()), lr0net),
            torch.optim.Adam([self.u], lr0u),
        )
        if self.learn_v:
            self.optim_v = torch.optim.Adam([self.v], lr0v)
        if self.learn_z:
            self.optim_z = torch.optim.Adam([self.z], lr0z)
        optimizers = {
            "nested": self.nested_step,
        }
        psvi_step = optimizers[self.trainer]
        t_start = time.time()
        # training loop
        for it in tqdm(range(self.num_epochs)):
            xbatch, ybatch = next(iter(self.train_loader))
            xbatch, ybatch = xbatch.to(self.device, non_blocking=True), ybatch.to(
                self.device, non_blocking=True
            )
            # evaluation
            if it % self.log_every == 0:
                test_rmse, test_ll = self.evaluate(**kwargs)
                with torch.no_grad():
                    lls_psvi.append(test_ll.item())
                    rmses_psvi.append(test_rmse.item())
                    core_idcs_psvi.append(self.num_pseudo)
                    times.append(times[-1] + time.time() - t_start)
                    vs.append((self.f(self.v, 0)).clone().cpu().detach().numpy())
                    if self.log_pseudodata:
                        us.append(self.u.clone().cpu().detach().numpy())
                        zs.append(self.z.clone().cpu().detach().numpy())
            # take a single optimization step
            outer_loss = psvi_step(xbatch, ybatch)
            if it % self.log_every == 0:
                print(
                    f"  \n\n\n  Predictive rmse {test_rmse.item():.2f} | pred ll {test_ll.item():.2f}| outer loss {outer_loss:.0f}"
                )
            
        # store results
        self.results["rmses"] = rmses_psvi
        self.results["lls"] = lls_psvi
        self.results["csizes"] = core_idcs_psvi
        self.results["times"] = times[1:]
        self.results["went"] = iws_entropy
        self.results["ness"] = nesses
        self.results["vent"] = vs_entropy
        self.results["vs"] = vs

        print("rmses : ", ["%.4f" % el for el in self.results["rmses"]])
        print("lls : ", ["%.4f" % el for el in self.results["lls"]])
        return self.results

    ## Compute predictive metrics
    def evaluate(
        self,
        correction=True,
        **kwargs,
    ):
        def revert_norm(y_pred):
            return y_pred * self.y_std + self.y_mean

        assert self.mc_samples > 1
        total, test_ll, rmses_unnorm = 0, 0, 0
        for xt, yt in self.test_loader:
            xt, yt = (
                xt.to(self.device, non_blocking=True),
                yt.to(self.device, non_blocking=True).squeeze(),
            )

            with torch.no_grad():
                all_data = torch.cat((self.u, xt)).squeeze(-1)
                model_out = self.model(all_data).squeeze(-1)
                pseudo_out = model_out[:, : self.num_pseudo]
                pseudo_ll = (
                    self.distr_fn(pseudo_out)
                    .log_prob(self.z.squeeze())
                    .mul(self.N * self.f(self.v, 0))
                    if self.num_pseudo > 0
                    else 0.0
                ).sum()
                test_data_out = model_out[:, self.num_pseudo :]
                sampled_nkl = sum(
                    m.sampled_nkl()
                    for m in self.model.modules()
                    if isinstance(m, VILinear)
                )

                log_weights = -pseudo_ll + sampled_nkl
                weights = log_weights.softmax(0)
                y_pred = torch.matmul(revert_norm(test_data_out).T, weights)
                rmses_unnorm += (y_pred - yt).square().sum()
                total += yt.size(0)
                test_ll += self.distr_fn(y_pred).log_prob(yt.squeeze()).sum()
        return (
            (rmses_unnorm / float(total)).sqrt(),
            test_ll / float(total),
        )


## PSVI with learnable v on a simplex (with constant sum constraint)
class PSVILearnV_regressor(PSVI_regressor):
    def __init__(self, learn_v=True, parameterised=True, **kwargs):
        super().__init__(**kwargs)
        self.learn_v, self.parameterised = learn_v, parameterised
        with torch.no_grad():
            self.v = torch.zeros(self.num_pseudo, device=self.device)
        self.v.requires_grad_(
            True
        )  # initialize learnable weights of coreset pseudodata to uniform
        self.f = (
            torch.softmax
        )  # transform v via softmax to keep the sum over the pseudodata fixed


## PSVI with learnable v on a simplex and learnable rescaling on total coreset likelihood
class PSVIAV_regressor(PSVILearnV_regressor):
    def __init__(self, learn_v=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.tensor([0.0], device=self.device)
        self.alpha.requires_grad_(True)
        self.f = lambda *x: (
            torch.exp(self.alpha) * torch.softmax(x[0], x[1])
        )  # transform v via softmax to keep the sum over the pseudodata fixed and multiply by a learnable non-negative coefficient
        self.optim_alpha = torch.optim.Adam([self.alpha], self.lr0alpha)
        self.results["alpha"] = []

    def evaluate(self, **kwargs):
        self.results["alpha"].append(
            self.alpha.clone()
            .cpu()
            .detach()
            .numpy()  # store the extra variational parameter
        )
        return super().evaluate(**kwargs)

    def nested_step(self, xbatch, ybatch):
        self.optim_u.zero_grad()
        self.optim_net.zero_grad()
        self.optim_alpha.zero_grad()
        if self.learn_v:
            self.optim_v.zero_grad()
        if self.learn_z:
            self.optim_z.zero_grad()
        with innerloop_ctx(self.model, self.optim_net) as (fmodel, diffopt):
            for in_it in range(self.inner_it):
                mfvi_loss = self.inner_elbo(model=fmodel)
                with torch.no_grad():
                    if self.register_elbos and in_it % self.log_every == 0:
                        self.elbos.append((1, -mfvi_loss.item()))
                diffopt.step(mfvi_loss)
            psvi_loss = self.psvi_elbo(xbatch, ybatch, model=fmodel)
            with torch.no_grad():
                if self.register_elbos:
                    self.elbos.append((0, -psvi_loss.item()))
            psvi_loss.backward()
        self.optim_u.step()
        if self.learn_v:
            self.optim_v.step()
            self.optim_alpha.step()
        if self.learn_z:
            self.optim_z.step()
        if self.scheduler_optim_net:
            self.scheduler_optim_net.step()
        nn.utils.vector_to_parameters(
            nn.utils.parameters_to_vector(list(fmodel.parameters())),
            self.model.parameters(),
        )
        return psvi_loss