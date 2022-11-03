# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ADAPTATION OF flow_psvi FOR MULTI-GPU PLATFORMS
"""

r"""
Experiment execution script: Users can specify the dataset, the statistical model and the inference methods,
and this script will generate a dictionary with the predictive performance.
"""

# Import libraries
import argparse
import os
import pickle
from collections import defaultdict
from platform import architecture
from typing import Any, Dict, List
import concurrent
import tqdm
import random

from psvi.inference.baselines import (
    run_giga,
    run_mfvi,
    run_mfvi_subset,
    run_opsvi,
    run_random,
    run_sparsevi,
    run_mfvi_regressor,
    run_mfvi_subset_regressor
)
from psvi.inference.psvi_classes import (
    PSVI,
    PSVIAFixedU,
    PSVIAV,
    PSVIFixedU,
    PSVILearnV,
    PSVI_Ablated,
    PSVI_No_IW,
    PSVI_No_Rescaling,
    PSVIFreeV,
    PSVI_regressor,
    PSVILearnV_regressor,
    PSVIAV_regressor,
)
from psvi.inference.sparsebbvi import run_sparsevi_with_bb_elbo
from psvi.models.logreg import *
from experiments_utils import read_dataset, read_regression_dataset
import multiprocessing
from multiprocessing import set_start_method
torch.autograd.set_detect_anomaly(True)

NUM_GPUS = 8 

parser = argparse.ArgumentParser()

# Arguments for the experiment workflow
parser.add_argument(
    "--fnm", default="results", type=str, help="Filename where results are stored"
)
parser.add_argument(
    "--datasets",
    default=["phishing"],
    nargs="+",
    choices=["webspam", "phishing", "adult", "MNIST", "halfmoon", "four_blobs", "sinus", "concrete", "energy", "power", "kin8nm", "protein", "naval", "yacht", "boston", "wine", "year", "synth_lr_10", "synth_lr_50", "synth_lr_200"],
    type=str,
    help="List of dataset names",
)
parser.add_argument(
    "--methods",
    default=["psvi_learn_v", "mfvi", "mfvi_subset"],
    nargs="+",
    type=str,
    help="List of inference method names",
)
parser.add_argument("--mc_samples", default=10, type=int, help="Monte Carlo samples")
parser.add_argument("--num_epochs", default=301, type=int, help="Training epochs")
parser.add_argument(
    "--num_trials",
    default=3,
    type=int,
    help="Trials executed for each inference method",
)
parser.add_argument(
    "--data_minibatch", default=128, type=int, help="Data minibatch size"
)
parser.add_argument(
    "--inner_it",
    default=100,
    type=int,
    help="Gradient steps in the inner problem of nested optimization",
)
parser.add_argument(
    "--outer_it",
    default=100,
    type=int,
    help="Gradient steps in the outer problem of nested optimization",
)
parser.add_argument(
    "--trainer",
    default="nested",
    choices=["nested", "hyper", "joint"],
    type=str,
    help="Method for computation of hypergradient",
)
parser.add_argument(
    "--diagonal",
    action=argparse.BooleanOptionalAction,
    help="Diagonal approximation of Gaussian covariance matrices used",
)
parser.add_argument(
    "--architecture",
    default="logistic_regression",
    choices=["logistic_regression", "logistic_regression_fullcov", "fn", "fn2", "lenet", "regressor_net"],
    type=str,
    help="Model architecture",
)
parser.add_argument(
    "--n_hidden",
    default=40,
    type=int,
    help="Number of hidden units in feedforward neural architectures",
)
parser.add_argument(
    "--n_layers",
    default=1,
    type=int,
    help="Number of layers in feedforward neural architectures",
)
parser.add_argument(
    "--log_every",
    default=150,
    type=int,
    help="Frequency of logging evaluation results throughout training (in number of outer gradient iterations)",
)
parser.add_argument(
    "--register_elbos",
    action=argparse.BooleanOptionalAction,
    help="Saving variational objectives values throughout inference for plotting",
)
parser.add_argument(
    "--init_sd",
    default=1e-6,
    type=float,
    help="Initialization of standard deviation for variational parameters",
)
parser.add_argument(
    "--lr0net",
    default=1e-3,
    type=float,
    help="Initial learning rate for model parameters optimizer",
)
parser.add_argument(
    "--lr0u",
    default=1e-4,
    type=float,
    help="Initial learning rate for optimizer of pseudocoreset point input coordinates u",
)
parser.add_argument(
    "--lr0v",
    default=1e-3,
    type=float,
    help="Initial learning rate for optimizer of coreset support coefficients",
)
parser.add_argument(
    "--lr0z",
    default=1e-3,
    type=float,
    help="Initial learning rate for optimizer of coreset points labels",
)
parser.add_argument(
    "--lr0alpha",
    default=1e-3,
    type=float,
    help="Initial learning rate for coreset likelihood rescaling coefficient",
)
parser.add_argument(
    "--init_at",
    default="subsample",
    choices=["subsample", "random"],
    type=str,
    help="Method for coreset points initialization",
)
parser.add_argument(
    "--compute_weights_entropy",
    action=argparse.BooleanOptionalAction,
    help="Comput entropy of weights for plotting",
)
parser.add_argument(
    "--coreset_sizes",
    default=[100],
    nargs="+",
    type=int,
    help="List of sizes for coresets computed throughout the experiment, or subsamples used for baselines mfvi_subset and random",
)
parser.add_argument(
    "--reset",
    action=argparse.BooleanOptionalAction,
    help="Reset model parameters over intervals during training",
)
parser.add_argument(
    "--prune",
    action=argparse.BooleanOptionalAction,
    help="Prune to coreset of smaller size",
)
parser.add_argument(
    "--prune_interval",
    default=400,
    type=int,
    help="Gradient steps in the outer problem of nested optimization between prunning steps",
)
parser.add_argument(
    "--prune_sizes",
    default=[20],
    nargs="+",
    type=int,
    help="List of sizes for coresets in a pruning experiment (decreasing)",
)
parser.add_argument(
    "--increment",
    action=argparse.BooleanOptionalAction,
    help="Learn tasks incrementally",
)
parser.add_argument(
    "--increment_interval",
    default=1000,
    type=int,
    help="Gradient steps in the outer problem of nested optimization between incremental learning stages",
)
parser.add_argument(
    "--increment_sizes",
    default=[20],
    nargs="+",
    type=int,
    help="List of sizes for coresets in the incremental learning setting (non-decreasing)",
)
parser.add_argument(
    "--retrain_on_coreset",
    action=argparse.BooleanOptionalAction,
    help="Retrain the variational model restricted only on the extracted coreset datapoints for the same number of epochs",
)
parser.add_argument(
    "--save_input_data",
    action=argparse.BooleanOptionalAction,
    help="Save input dataset",
)
parser.add_argument(
    "--test_ratio", default=0.2, type=float, help="Ratio of test dataset size"
)
parser.add_argument(
    "--log_pseudodata",
    action=argparse.BooleanOptionalAction,
    help="Store pseudodata for visualisation",
)
parser.add_argument(
    "--data_folder",
    default="../data",
    type=str,
    help="Folder where dataset gets stored",
)
parser.add_argument(
    "--results_folder",
    default="../results",
    type=str,
    help="Folder where evaluation files get stored",
)
parser.add_argument(
    "--learn_z",
    action=argparse.BooleanOptionalAction,
    help="Learn soft labels for distilled data",
)
parser.add_argument(
    "--gamma", default=1., type=float, help="Decay factor of learning rate"
)

parser.set_defaults(
    diagonal=True,
    reset=False,
    compute_weights_entropy=False,
    register_elbos=False,
    save_input_data=False,
    prune=False,
    increment=False,
    log_pseudodata=False,
    retrain_on_coreset=False, 
    learn_z=False,
)

parsed_args = parser.parse_args()

method_args = vars(parsed_args)
datasets, methods = method_args["datasets"], method_args["methods"]
method_args["logistic_regression"] = method_args['architecture'] == 'logistic_regression'

[
    os.makedirs(fold)
    for fold in [method_args["data_folder"], method_args["results_folder"]]
    if not os.path.exists(fold)
]  # make folders for data and results storage

def pass_dict(d, f):
    return f(**d)

def rec_dd():
    return defaultdict(rec_dd)


results = rec_dd()  # recursive dictionary for storage of inference results

# Specify  inference methods
def inf_alg(**kwargs): 
    if kwargs["nm_alg"]=="psvi":
        return PSVI(**kwargs).run_psvi( **kwargs)
    elif kwargs["nm_alg"]=="psvi_ablated": 
        return PSVI_Ablated( **kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_learn_v": 
        return PSVILearnV( **kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_alpha_v": 
        return PSVIAV(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_no_iw": 
        return PSVI_No_IW(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_free_v": 
        return PSVIFreeV(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_no_rescaling": 
        return PSVI_No_Rescaling(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_fixed_u": 
        return PSVIFixedU(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_alpha_fixed_u": 
        return PSVIAFixedU(**kwargs).run_psvi(**kwargs )
    elif kwargs["nm_alg"]=="psvi_regressor": 
        return PSVI_regressor(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_alpha_v_regressor": 
        return  PSVIAV_regressor(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="psvi_learn_v_regressor": 
        return  PSVILearnV_regressor(**kwargs).run_psvi(**kwargs)
    elif kwargs["nm_alg"]=="sparsebbvi": 
        return run_sparsevi_with_bb_elbo
    elif kwargs["nm_alg"]=="opsvi": 
        return run_opsvi
    elif kwargs["nm_alg"]=="random": 
        return run_random
    elif kwargs["nm_alg"]=="sparsevi": 
        return run_sparsevi
    elif kwargs["nm_alg"]=="giga": 
        return run_giga
    elif kwargs["nm_alg"]=="mfvi": 
        return run_mfvi
    elif kwargs["nm_alg"]=="mfvi_subset": 
        return run_mfvi_subset
    elif kwargs["nm_alg"]=="mfvi_regressor": 
        return run_mfvi_regressor
    elif kwargs["nm_alg"]=="mfvi_subset_regressor": 
        return run_mfvi_subset_regressor



def experiment_driver(
    datasets: List[str],
    methods: Dict[str, bool],
    method_args: Dict[str, Any],
) -> None:
    r"""
    Run experiment
    """
    job_args = list()
    for dnm in datasets:
        # Read the dataset
        print(f"\nReading/Generating the dataset {dnm.upper()}")
        x, y, xt, yt, N, D, train_dataset, test_dataset, num_classes = read_dataset(
            dnm, method_args
        )
        print(
            f"\nBayesian {'logistic regression' if method_args['logistic_regression'] else 'neural network'} experiment.\nInference via {' '.join(map(lambda x:x.upper(), methods))} on {dnm} data over {method_args['num_trials']} {'independent trials.' if method_args['num_trials']>1 else 'trial.'}\n\n\n"
        )
        for nm_alg in methods:
            print(f"\n\nRunning {nm_alg}\n")
            logistic_regression = method_args.get(
                "logistic_regression", method_args.get("architecture") == "logreg"
            )
            compute_weights_entropy = (
                not nm_alg.startswith(("opsvi", "mfvi_subset"))
            ) and method_args["compute_weights_entropy"]
            tps = (
                method_args["coreset_sizes"]
                if nm_alg.startswith(("psvi", "opsvi", "mfvi_subset"))
                else [-1]
            )  # alias for baselines with no explicit constraint on dataset size
            for t in range(method_args["num_trials"]):
                print(f"Trial #{t}")
                for (
                    ps
                ) in tps:  # range of pseudocoreset sizes tested over the experiment
                    print(
                        f"Coreset/Subset with {ps if not method_args['increment'] else method_args['increment_sizes'][0]} datapoints"
                    ) if ps != -1 else print("Unconstrained data access")
                    idx = len(job_args)
                    job_args.append({"mc_samples":method_args["mc_samples"],
                                            "num_epochs":method_args["num_epochs"],
                                            "data_minibatch":method_args["data_minibatch"],
                                            "D":D,
                                            "N":N,
                                            "tr":t,
                                            "diagonal":method_args["diagonal"],
                                            "x":x,
                                            "y":y,
                                            "xt":xt,
                                            "yt":yt,
                                            "inner_it":method_args["inner_it"],
                                            "outer_it":method_args["outer_it"],
                                            "scatterplot_coreset":method_args.get(
                                                "scatterplot_coreset"
                                            ),  # not parsed for some methods atm
                                            "logistic_regression":logistic_regression,
                                            "trainer":method_args["trainer"],
                                            "log_every":method_args["log_every"],
                                            "register_elbos":method_args["register_elbos"],
                                            "lr0u":method_args["lr0u"],
                                            "lr0net":method_args["lr0net"],
                                            "lr0v":method_args["lr0v"],
                                            "lr0z":method_args["lr0z"],
                                            "lr0alpha":method_args["lr0alpha"],
                                            "init_args":method_args["init_at"],
                                            "init_sd":method_args[
                                                "init_sd"
                                            ],  # initialization of variance in variational model
                                            "num_pseudo":ps,
                                            "seed":t,  # map random seed to the trial number for reproducibility of inference result at the beginning of each of the baseline
                                            "compute_weights_entropy":compute_weights_entropy,
                                            "reset":method_args.get("reset"),
                                            "reset_interval":method_args.get("reset_interval"),
                                            "architecture":method_args.get("architecture"),
                                            "log_pseudodata":method_args.get("log_pseudodata"),
                                            "n_hidden":method_args.get(
                                                "n_hidden", 40
                                            ),  # hidden units in nn architecture
                                            "n_layers":method_args.get("n_layers", 1),
                                            "train_dataset":train_dataset,
                                            "test_dataset":test_dataset,
                                            "dnm":dnm,
                                            "nc":num_classes,
                                            "prune":method_args.get("prune"),
                                            "prune_interval":method_args.get("prune_interval"),
                                            "prune_sizes":method_args.get("prune_sizes"),
                                            "increment":method_args.get("increment"),
                                            "increment_interval":method_args.get("increment_interval"),
                                            "increment_sizes":method_args.get("increment_sizes"),
                                            "retrain_on_coreset":method_args.get("retrain_on_coreset"),
                                            "learn_z":method_args["learn_z"],
                                            "nm_alg":nm_alg,
                                            "device_id":idx % NUM_GPUS,
                                    })
    pool = multiprocessing.Pool(NUM_GPUS) # first arg is the number of workers 
    results_pool = [pool.apply_async(inf_alg, kwds=job_arg) for job_arg in job_args]
    ii=0
    for result in results_pool:
        _job_arg = job_args[ii]
        ii+=1
        results[_job_arg["dnm"]][_job_arg["nm_alg"]][_job_arg["num_pseudo"]][_job_arg["tr"]] = result.get()
    return write_to_files(results, method_args["fnm"])


def write_to_files(results: Dict[str, Any], fnm: str) -> None:
    r"""
    Write results to pk files
    """
    res_fnm = f"{method_args['results_folder']}/{fnm}.pk"
    print(f"Storing results in {res_fnm}")
    with open(res_fnm, "wb") as outfile:
        pickle.dump(results, outfile)


## Entry point
if __name__ == "__main__":
    set_start_method('spawn')
    (experiment_driver(
        datasets,
        methods,
        method_args,
    ) if method_args.get("architecture") != "regressor_net" 
                else  regressor_experiment_driver(
                                        datasets,
                                        methods,
                                        method_args))# run experiment
