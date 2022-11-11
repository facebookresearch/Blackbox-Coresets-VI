# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name="psvi",
    version="0.1.0",
    description="Setting up a python package for Bayesian inference using variational coresets",
    author="Dionysis Manousakas",
    author_email="dm754@cantab.ac.uk",
    license="LICENSE",
    packages=find_packages(include=["psvi", "psvi.*"]),
    install_requires=[
        "iopath==0.1.10",
        "matplotlib>=3.5.2",
        "numpy>=1.22.4",
        "pandas>=1.4.3",
        "Pillow==9.2.0",
        "requests==2.25.1",
        "scikit_learn>=1.1.1",
        "setuptools>=59.6.0",
        "torch>=1.12.0",
        "torchvision==0.13.0",
        "tqdm==4.64.0",
        "TyXe @ git+https://github.com/TyXe-BDL/TyXe",
        "arff==0.9",
	"pystan==3.5.0",
    ],
    keywords=[
        "bilevel optimization",
        "hypergradient",
	"sampling",
        "importance sampling",
        "variational inference",
        "Monte Carlo",
        "Bayesian",
        "neural networks",
        "pruning",
        "sparsity",
        "coresets",
        "distillation",
        "meta-learning",
        "inducing points",
        "pseudodata",
        "neural networks",
    ],
)
