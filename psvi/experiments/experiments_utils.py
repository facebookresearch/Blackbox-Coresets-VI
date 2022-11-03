# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import requests
import urllib.request
import zipfile
from collections import namedtuple
from io import BytesIO
import arff
import json 

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from psvi.models.neural_net import (
    make_fc2net,
    make_fcnet,
    make_lenet,
    make_regressor_net,
    VILinear,
    VILinearMultivariateNormal,
)
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

r"""
    Statistics used for normalization of some benchmark vision datasets
"""
dataset_normalization = dict(
    MNIST=((0.1307,), (0.3081,)),
    Cifar10=((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
)

r"""
    Classes of some benchmark vision datasets
"""
dataset_labels = dict(
    MNIST=list(range(10)),
    Cifar10=(
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "monkey",
        "horse",
        "ship",
        "truck",
    ),
)
DatasetStats = namedtuple(
    "DatasetStats", " ".join(["num_channels", "real_size", "num_classes"])
)

r"""
    Dimensions of vision benchmark datasets
"""
dataset_stats = dict(
    MNIST=DatasetStats(1, 28, 10),
    Cifar10=DatasetStats(3, 32, 10),
)


class SynthDataset(Dataset):
    r"""
    Custom torch dataset class supporting transforms
    """

    def __init__(self, x, y=None, transforms=None):
        self.data = x
        self.targets = y
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def subset_where(self, cs=[0, 1]): 
        r"""
        Returns subset of data corresponding to given list of classes
        """
        idcs = torch.isin(self.targets, torch.tensor(cs))
        return SynthDataset(self.data[idcs], self.targets[idcs])

    def concatenate(self, u, z):
        return SynthDataset(torch.cat((self.data, u)), y=torch.cat((self.targets, z)))

def split_data(N, p_split=(0.6, 0.2, 0.2), n_split=None, shuffle=True, seed=None):
    r"""
    Helper function for splitting data into train / validation / test
    """
    if seed is not None:
        np.random.seed(seed)

    if n_split is None:
        p_split = np.array(p_split)
        assert np.sum(p_split == -1) <= 1
        p_split[p_split == -1] = 1 - (np.sum(p_split) + 1)
        assert np.sum(p_split) == 1.0

        p_train, p_val, p_test = p_split
        train_idx = int(np.ceil(p_train * N))
        val_idx = int(np.ceil(train_idx + p_val * N))
    else:
        n_split = np.array(n_split)
        assert np.sum(n_split == -1) <= 1
        n_split[n_split == -1] = N - (np.sum(n_split) + 1)
        assert np.sum(n_split) == N

        n_train, n_val, n_test = n_split
        train_idx = int(n_train)
        val_idx = int(train_idx + n_val)

    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    return {
        "train": idx[:train_idx],
        "val": idx[train_idx:val_idx],
        "test": idx[val_idx:],
    }

# custom dataset
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



def read_regression_dataset(dnm, method_args):
        (X, Y), indices = get_regression_benchmark(
            dnm,
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
        return x, y, xv, yv, xt, yt, N, D, train_dataset, val_dataset, test_dataset, y_mean, y_std, taus


def get_regression_benchmark(name, seed=111, data_dir="psvi/data/", **kwargs):
    r"""
    Return data from UCI sets
        - param name: (str) Name of dataset to be used
        - param seed: (int) Random seed for splitting data into train and test
        - param kwargs: (dict) Additional arguments for splits
        - return: Inputs, outputs, and data-splits
    """
    np.random.seed(seed)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    urllinks = {"concrete": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
                "energy": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
                "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
                "kin8nm": "https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff",
                "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
                "naval": "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
                "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
                "boston": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "year": "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"}

    filename = urllinks[name].split('/')[-1]
    if not os.path.exists(data_dir + filename):
        urllib.request.urlretrieve(
            urllinks[name], data_dir + filename)

    if name in ["concrete", "energy"]:
        data = np.array(pd.read_excel(data_dir + filename))
    elif name == "power":
        zipfile.ZipFile(data_dir + filename).extractall(data_dir)
        data = pd.read_excel(data_dir + 'CCPP/Folds5x2_pp.xlsx', header=0).values
    elif name == "kin8nm":
        dataset = arff.load(open(data_dir + filename))
        data = np.array(dataset['data'])
    elif name == "protein":
        data = np.array(pd.read_csv(data_dir + filename))
    elif name == "naval":
        zipfile.ZipFile(data_dir + filename).extractall(data_dir)
        data = np.loadtxt(data_dir + "UCI CBM Dataset/data.txt") 
    elif name in ["yacht", "boston"]:
        data = np.loadtxt(data_dir + filename)
    elif name == "wine":
        data = np.array(pd.read_csv(data_dir + filename, delimiter=";"))
    elif name == "year":
        zipfile.ZipFile(data_dir + "/YearPredictionMSD.txt.zip").extractall(data_dir)
        data = np.loadtxt(data_dir + "/YearPredictionMSD.txt" , delimiter=",")
    elif name == "sinus":
        X = np.random.rand(10**3) * 2 * np.pi
        Y = np.sin(X)
        data = np.stack((X, Y), axis=-1)
    else:
        raise ValueError("Unsupported dataset: {}".format(data_dir, name))

    if name in ["energy", "naval"]:  # dataset has 2 response values
        X = data[:, :-2]
        Y = data[:, -2:-1]  # pick first response value
    else:
        X = data[:, :-1]
        Y = data[:, -1:]
    return (X, Y), split_data(len(X), **kwargs)


def hyperparams_for_regression():
    r"""
    Grid search space for precision in the regression BNN model
    """
    return {
        "concrete": [0.025, 0.05, 0.075],
        "energy": [0.25, 0.5, 0.75],
        "power": [0.05, 0.1, 0.15],
        "kin8nm": [150, 200, 250],
        "protein": [0.025, 0.05, 0.075],
        "naval": [30000, 40000, 50000],
        "yacht": [0.25, 0.5, 0.75],
        "boston": [0.1, 0.15, 0.2],
        "wine": [2.5, 3.0, 3.5],
        "year":[0.1, 1., 10.]
    }


def make_four_class_dataset(N_K=250):
    r"""
    Return two-dimensional four_blobs dataset with datapoints equally distributed among 4 classes
    :param N_K (int): number of datapoints per class
    """
    X1 = torch.cat(
        [
            0.8 + 0.4 * torch.randn(N_K, 1),
            1.5 + 0.4 * torch.randn(N_K, 1),
        ],
        dim=-1,
    )
    Y1 = 0 * torch.ones(X1.size(0)).long()
    X2 = torch.cat(
        [
            0.5 + 0.6 * torch.randn(N_K, 1),
            -0.2 - 0.1 * torch.randn(N_K, 1),
        ],
        dim=-1,
    )
    Y2 = 1 * torch.ones(X2.size(0)).long()
    X3 = torch.cat(
        [
            2.5 - 0.1 * torch.randn(N_K, 1),
            1.0 + 0.6 * torch.randn(N_K, 1),
        ],
        dim=-1,
    )
    Y3 = 2 * torch.ones(X3.size(0)).long()
    X4 = torch.distributions.MultivariateNormal(
        torch.Tensor([-0.5, 1.5]),
        covariance_matrix=torch.Tensor([[0.2, 0.1], [0.1, 0.1]]),
    ).sample(torch.Size([N_K]))
    Y4 = 3 * torch.ones(X4.size(0)).long()

    X = torch.cat([X1, X2, X3, X4], dim=0)
    X[:, 1] -= 1
    X[:, 0] -= 0.5
    Y = torch.cat([Y1, Y2, Y3, Y4])

    rows_permutations = torch.randperm(X.size()[0])
    return (
        X[rows_permutations, :],
        Y[rows_permutations],
    )  # shuffle rows for our train/test split


def set_up_model(
    D=None,
    n_hidden=None,
    nc=None,
    mc_samples=None,
    architecture=None,
    **kwargs,
):
    r"""
    Return torch nn model with the desired architecture
    :param D (int): dimensionality of input data
    :param n_hidden (int): number of units in each hidden layer
    :param nc (int): dimensionality of last layer
    :param mc_samples (int): number of samples produced at each forward pass through the nn
    :param architecture (str): nn architecture
                                - "fn": fully connected feedforward network with diagonal Gaussian on variational layers
                                - "residual_fn": fn with residual connections
                                - "fn2": fn with full covariance matrix on variational layers
                                - "lenet": LeNet architecture
                                - "logistic_regression": single layer nn (no hidden layers) implementing the logistic regression model
                                - "logistic_regression_fullcov": single layer nn (no hidden layers) implementing the logistic regression model with full covariance variational approximations
    """
    if architecture in {"fn", "residual_fn"}:
        return make_fcnet(
            D,
            n_hidden,
            nc,
            linear_class=VILinear,
            nonl_class=nn.ReLU,
            mc_samples=mc_samples,
            residual=(architecture == "residual_fn"),
            **kwargs,
        )
    elif architecture in {"fn2"}:
        return make_fc2net(
            D,
            n_hidden,
            nc,  # does not support argument on the number of chanells
            linear_class=VILinearMultivariateNormal,
            nonl_class=nn.ReLU,
            mc_samples=mc_samples,
            **kwargs,
        )
    elif architecture == "lenet":
        return make_lenet(
            linear_class=VILinear, nonl_class=nn.ReLU, mc_samples=mc_samples
        )
    elif architecture in {
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
    elif architecture == "logistic_regression":
        return nn.Sequential(VILinear(D, nc, mc_samples=mc_samples))    
    elif architecture == "logistic_regression_fullcov":
        return nn.Sequential(VILinearMultivariateNormal(D, nc, mc_samples=mc_samples))
    else:
        raise ValueError(
            "Architecture should be one of \n'lenet', 'logistic_regression', 'logistic_regression_fullcov', 'fn', 'fn2', 'residual_fn'"
        )


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


def get_torchvision_info(name):
    r"""
    Returns statistical information for specified torchvision benchmark dataset
    """
    assert name in dataset_stats, "Unsupported dataset: {}".format(name)
    num_channels, input_size, num_classes = dataset_stats[name]
    normalization = dataset_normalization[name]
    labels = dataset_labels[name]
    return num_channels, input_size, num_classes, normalization, labels


def load_dataset(path, urls):
    r"""
    Writes on a file a dataset living on a given URL
    """
    if not os.path.exists(path):
        os.mkdir(path)
    for url in urls:
        data = requests.get(url).content
        filename = os.path.join(path, os.path.basename(url))
        with open(filename, "wb") as file:
            file.write(data)
    return


def read_adult(data_folder):
    r"""
    Returns the adult dataset for logistic regression
    """
    urls = [
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    ]
    load_dataset(data_folder, urls)

    columns = [
        "age",
        "workClass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    train_data = pd.read_csv(
        data_folder + "/adult.data",
        names=columns,
        sep=" *, *",
        na_values="?",
        engine="python",
    ).dropna()
    test_data = pd.read_csv(
        data_folder + "/adult.test",
        names=columns,
        sep=" *, *",
        skiprows=1,
        na_values="?",
        engine="python",
    ).dropna()

    X, Xt = train_data[columns[::-1]], test_data[columns[::-1]]
    Y = np.array([0 if s == "<=50K" else 1 for s in train_data["income"]])
    Yt = np.array([0 if s == "<=50K." else 1 for s in test_data["income"]])

    # numerical columns : standardize
    numcols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    ss = StandardScaler()
    ss.fit(X[numcols])
    Xnum, Xtnum = ss.transform(X[numcols]), ss.transform(Xt[numcols])

    # categorical columns: apply 1-hot-encoding
    catcols = [
        "workClass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    enc = OneHotEncoder()
    enc.fit(X[catcols])
    Xcat, Xtcat = (
        enc.transform(X[catcols]).toarray(),
        enc.transform(Xt[catcols]).toarray(),
    )
    X, Xt = np.concatenate((Xnum, Xcat), axis=1), np.concatenate((Xtnum, Xtcat), axis=1)

    pca = PCA(n_components=10)
    pca.fit(X)
    X = pca.transform(X)
    Xt = pca.transform(Xt)
    X = np.c_[X, np.ones(X.shape[0])]
    Xt = np.c_[Xt, np.ones(Xt.shape[0])]
    return X, Y, Xt, Yt


def read_phishing(data_folder, dnm="phishing"):
    r"""
    Returns the phishing dataset for logistic regression
    """
    filename, urllink = (
        f"{data_folder}/{dnm}.npz",
        f"https://github.com/trevorcampbell/bayesian-coresets/blob/master/examples/data/{dnm}.npz?raw=true",
    )
    if not os.path.isfile(filename):
        response = requests.get(urllink)
        response.raise_for_status()
        data = np.load(BytesIO(response.content))
    else:
        data = np.load(filename)
    return data["X"], data["y"]

def read_webspam(data_folder, dnm="webspam"):
    r"""
    Returns the webspam dataset for logistic regression
    """
    import sklearn.datasets as skl_ds
    from sklearn import preprocessing
    import scipy.sparse as sp
    import numpy as np

    fnm_train, urllink_train = (
        f"{data_folder}/{dnm}_train.svm",
        "https://bitbucket.org/jhhuggins/lrcoresets/raw/cdcda24b5854ef380795ec11ab5321d0ec53c3fe/data/webspam_train.svm",
    )

    fnm_test, urllink_test = (
        f"{data_folder}/{dnm}_test.svm",
        "https://bitbucket.org/jhhuggins/lrcoresets/raw/cdcda24b5854ef380795ec11ab5321d0ec53c3fe/data/webspam_test.svm",
    )
    
    import urllib.request
    if not os.path.isfile(fnm_train):
        urllib.request.urlretrieve(urllink_train, fnm_train)
    if not os.path.isfile(fnm_test):
        urllib.request.urlretrieve(urllink_test, fnm_test)
    

    def _load_svmlight_data(path):
        X, y = skl_ds.load_svmlight_file(path)
        return X, y

    def load_data(path, file_type, max_data=0, max_dim=0,
                preprocess=True, include_offset=True):
        """Load data from a variety of file types.
        Parameters
        ----------
        path : string
            Data file path.
        file_type : string
            Supported file types are: 'svmlight', 'npy' (with the labels y in the
            rightmost col), 'npz', 'hdf5' (with datasets 'x' and 'y'), and 'csv'
            (with the labels y in the rightmost col)
        max_data : int
            If positive, maximum number of data points to use. If zero or negative,
            all data is used. Default is 0.
        max_dim : int
            If positive, maximum number of features to use. If zero or negative,
            all features are used. Default is 0.
        preprocess : boolean or Transformer, optional
            Flag indicating whether the data should be preprocessed. For sparse
            data, the features are scaled to [-1, 1]. For dense data, the features
            are scaled to have mean zero and variance one. Default is True.
        include_offset : boolean, optional
            Flag indicating that an offset feature should be added. Default is
            False.
        Returns
        -------
        X : array-like matrix, shape=(n_samples, n_features)
        y : int ndarray, shape=(n_samples,)
            Each entry indicates whether each example is negative (-1 value) or
            positive (+1 value)
        pp_obj : None or Transformer
            Transformer object used on data, or None if ``preprocess=False``
        """
        if not isinstance(path, str):
            raise ValueError("'path' must be a string")

        if file_type in ["svmlight", "svm"]:
            X, y = _load_svmlight_data(path)
        else:
            raise ValueError("unsupported file type, %s" % file_type)

        y_vals = set(y)
        if len(y_vals) != 2:
            raise ValueError('Only expected y to take on two values, but instead'
                            'takes on the values ' + ', '.join(y_vals))
        if 1.0 not in y_vals:
            raise ValueError('y does not take on 1.0 as one on of its values, but '
                            'instead takes on the values ' + ', '.join(y_vals))
        if -1.0 not in y_vals:
            y_vals.remove(1.0)
            print('converting y values of %s to -1.0' % y_vals.pop())
            y[y != 1.0] = -1.0

        if preprocess is False:
            pp_obj = None
        else:
            if preprocess is True:
                if sp.issparse(X):
                    pp_obj = preprocessing.MaxAbsScaler(copy=False)
                else:
                    pp_obj = preprocessing.StandardScaler(copy=False)
                pp_obj.fit(X)
            else:
                pp_obj = preprocess
            X = pp_obj.transform(X)

        if include_offset:
            X = preprocessing.add_dummy_feature(X)
            X = np.flip(X, -1) # move intercept to the last column of the array

        if sp.issparse(X) and (X.nnz > np.prod(X.shape) / 10 or X.shape[1] <= 20):
            print("X is either low-dimensional or not very sparse, so converting "
                "to a numpy array")
            X = X.toarray()
        if isinstance(max_data, int) and max_data > 0 and max_data < X.shape[0]:
            X = X[:max_data,:]
            y = y[:max_data]
        if isinstance(max_dim, int) and max_dim > 0 and max_dim < X.shape[1]:
            X = X[:,:max_dim]

        return X, y, pp_obj


    X, y, _ = load_data(fnm_train, 'svm')
    # load testing data if it exists
    Xt, yt, _ = load_data(fnm_test, 'svm')
    y[y==-1], yt[yt==-1] = 0, 0
    np.savez('webspam', X=X, y=y, Xt=Xt, yt=yt)
    return X, y, Xt, yt



def make_synthetic(num_datapoints=1000, D=2):
    r"""
    Generate D-dimensional synthetic dataset for logistic regression
    """
    mu = np.array([0]*D)
    cov = np.eye(D)
    th = np.array([5]*D)
    X = np.random.multivariate_normal(mu, cov, num_datapoints)
    ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
    y = (np.random.rand(num_datapoints) <= ps).astype(int)
    y[y==0] = -1
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def read_dataset(dnm, method_args):
    r"""
    Returns one of the supported benchmark or synthetic dataset for the experiments in logistic regression, classification or regression via Bayesian nns
    """
    # TBC: check if inference methods are compatible with the dataset and raise exceptions accordingly
    if dnm != "MNIST":  # UCI or synthetic datasets
        if dnm == "halfmoon":
            # Generate HalfMoon data
            (X, Y), num_classes = (
                make_moons(n_samples=1000, noise=0.1, random_state=42),
                2,
            )
            X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(
                Y.astype(np.float32)
            )
        elif dnm == "four_blobs":
            # Generate synthetic multiclass data
            (X, Y), num_classes = make_four_class_dataset(N_K=250), 4
        elif dnm == "phishing":
            (X, Y), num_classes = read_phishing(method_args["data_folder"]), 2
            X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(
                Y.astype(np.float32)
            )
        elif dnm == "adult":
            (x, y, xt, yt), num_classes = read_adult(method_args["data_folder"]), 2
            x, y, xt, yt = (
                torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)),
                torch.from_numpy(xt.astype(np.float32)),
                torch.from_numpy(yt.astype(np.float32)),
            )
        elif dnm == "webspam":
            (x, y, xt, yt), num_classes = read_webspam(method_args["data_folder"]), 2
            x, y, xt, yt = (
                torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)),
                torch.from_numpy(xt.astype(np.float32)),
                torch.from_numpy(yt.astype(np.float32)),
            )
        elif dnm.startswith("synth_lr"):
            (X, Y), num_classes = make_synthetic(D=int(dnm.split('_')[-1]), num_datapoints=1000), 2
        if dnm.startswith(("halfmoon", "four_blobs", "phishing", "synth_lr")):  # splite in train / test data
            Y[Y == -1] = 0
            test_size = int(method_args["test_ratio"] * X.shape[0])
            x, y, xt, yt = (
                X[:-test_size],
                Y[:-test_size],
                X[-test_size:],
                Y[-test_size:],
            )
        N, D = x.shape
        (train_dataset, test_dataset) = (
            (SynthDataset(x, y), SynthDataset(xt, yt))
            if dnm.startswith(("halfmoon", "four_blobs", "phishing", "synth_lr", "webspam", "adult"))
            else (None, None)
        )
    else:
        _, input_size, num_classes, normalization, _ = get_torchvision_info(dnm)
        real_size = dataset_stats[dnm].real_size
        N, D = 60000, input_size
        if input_size != real_size:
            transform_list = [
                torchvision.transforms.Resize([input_size, input_size], Image.BICUBIC)
            ]
        else:
            transform_list = []
        transform_list += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            train_dataset, test_dataset = torchvision.datasets.MNIST(
                root=method_args["data_folder"],
                download=True,
                train=True,
                transform=torchvision.transforms.Compose(transform_list),
            ), torchvision.datasets.MNIST(
                root=method_args["data_folder"],
                download=True,
                train=False,
                transform=torchvision.transforms.Compose(transform_list),
            )
        x, y, xt, yt = None, None, None, None
    return x, y, xt, yt, N, D, train_dataset, test_dataset, num_classes


from json.decoder import JSONDecodeError
def update_hyperparams_dict(dnm, best_tau, fnm='psvi/data/opt_regr_hyperparams.json'):
    pass 
    '''
    with open(fnm, "a+") as f:
        try:
            opt_taus = json.loads(f)
        except JSONDecodeError:
            opt_taus = {"init":0}
            json.dumps(opt_taus, f)
            opt_taus = json.load(f)
        opt_taus[dnm] = opt_taus.get(dnm, best_tau)
    '''
