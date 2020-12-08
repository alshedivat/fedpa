# Federated Learning via Posterior Averaging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alshedivat/fedpa/blob/master/fedpa_playground.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![FedAvg vs. FedPA](/assets/fedavg-vs-fedpa.gif)

This repository contains a minimalistic (yet general and modular) JAX implementation of federated posterior averaging (FedPA) algorithm along with a variety of simulation experiments on synthetically generated problems.

## Usage
The easiest way to reproduce our experiments and/or compare FedAvg and FedPA is by using the colab notebook provided in this repository (simply click the `Open in Colab` button at the top of README).
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
FedAvg and FedPA are implemented by providing the corresponding `client_update_fn` and `server_update_fn` arguments.

## FAQ

- **Q:** Is there a TensorFlow Federated (TFF) implementation of FedPA? <br>
  **A:** Currently, TFF implementation is available in the `fedpa` branch of a mirror of [google-research/federated](https://github.com/alshedivat/federated-research/blob/fedpa/optimization/shared/post_avg.py).
The code will be eventually released as a sub-project in the official [google-research/federated](https://github.com/google-research/federated).

For any other questions, please feel free to raise an issue.

## Citation
```bibtex
@article{alshedivat2020federated,
  title={Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms},
  author={Al-Shedivat, Maruan and Gillenwater, Jennifer and Xing, Eric and Rostamizadeh, Afshin},
  journal={arXiv preprint arXiv:2010.05273},
  year={2020}
}
```

## License

Apache 2.0
