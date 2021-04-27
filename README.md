# Federated Learning via Posterior Averaging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alshedivat/fedpa/blob/master/fedpa_playground.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![FedAvg vs. FedPA](/assets/fedavg-vs-fedpa.gif)

This repository contains a minimalistic (yet general and modular) [JAX](https://github.com/google/jax) implementation of federated posterior averaging (FedPA) algorithm along with a variety of simulation experiments on synthetically generated problems.

## Usage
The easiest way to reproduce our synthetic experiments and/or compare FedAvg and FedPA is by using the colab notebook provided in this repository (simply click the `Open in Colab` button at the top of README).
If you would like to use our JAX implementation of FedAvg or FedPA elsewhere, the `federated/` folder can be used a standalone Python package.

## Organization of the code

All the code is located under the `federated` folder and organized into multiple sub-modules:
- `objectives`: Contains implementations of synthetic objective functions. We assume that each client is represented by the corresponding objective. The objective is base on the client's data and can be (1) evaluated or (2) return a (stochastic) gradient at a given point (or parameter vector).
- `inference`: Contains functions for computing client updates, running SGD (and its variations), doing posterior sampling, estimating moments of a distribution from samples.
- `learning`: Contains implementations of the learning algorithms (FedAvg, FedPA, and their variations.)
- `utils`: Contains various utility functions, e.g., for timing code or plotting figures.

**A note on implementation.**
Our minimalistic library is implemented in a functional style.
The main learning function is [`fed_opt`](https://github.com/alshedivat/fedpa/blob/master/federated/learning/algorithms.py#L85-L158) (defined in `federated.learning.algorithms`), which implements the generalized federated optimization (corresponds to Algorithm 1 in [our paper](https://arxiv.org/abs/2010.05273), originally proposed by [Reddi*, Charles*, et al. (2020)](https://arxiv.org/abs/2003.00295)).
`fed_opt` takes `client_update_fn` and `server_update_fn` functions as arguments, which are used for computing client and server updates, respectively.
FedAvg and FedPA are implemented by providing the corresponding `client_update_fn` and `server_update_fn` arguments to `fed_opt`.

## Reproducing results on FL benchmark tasks

This mini-library does NOT support running experiments on FL benchmark tasks such as EMNIST, CIFAR100, etc.
If you would like to run FedPA on these benchmarks, please use our [TFF implementation](https://github.com/google-research/federated/tree/master/posterior_averaging).

## Citation
```bibtex
@inproceedings{alshedivat2021federated,
  title={Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms},
  author={Al-Shedivat, Maruan and Gillenwater, Jennifer and Xing, Eric and Rostamizadeh, Afshin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

## License

Apache 2.0
