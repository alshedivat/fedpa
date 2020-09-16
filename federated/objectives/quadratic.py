# coding=utf-8
# Copyright 2020 Maruan Al-Shedivat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quadratic objective functions."""

import functools
from typing import List, Optional

import attr
import jax.numpy as jnp
import numpy as np
from jax.api import jit
from sklearn import datasets

from .base import Dataset, Objective, StochasticObjective


@attr.s
class LeastSquares(StochasticObjective):
    """A quadratic that represents a least squares problem.

    Objective function: `0.5 * ||y - X^T w||^2`.
    """

    lam: float = attr.ib(default=0.0)

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    @functools.partial(jit, static_argnums=(0,))
    def eval(self, x: jnp.ndarray, data_batch: Dataset) -> jnp.ndarray:
        """Computes the mean squared error loss for `x` on the `data_batch`."""
        x_batch, y_batch = data_batch
        loss = 0.5 * jnp.mean(jnp.square(jnp.dot(x, x_batch.T) - y_batch))
        reg = jnp.dot(x, x)
        return loss + self.lam * reg

    @functools.partial(jit, static_argnums=(0,))
    def solve(self, eps: float = 0.0):
        """Returns the optimum by solving `(X^T X) x = X^T y`."""
        n = self.X.shape[0]
        A = jnp.dot(self.X.T, self.X) / n
        b = jnp.dot(self.X.T, self.y) / n
        return jnp.linalg.solve(A + eps * jnp.eye(self.dim), b)


@attr.s
class Quadratic(Objective):
    """Represents a quadratic objective `0.5 * x^T A x - b^T x + c`."""

    A: jnp.ndarray = attr.ib()
    b: jnp.ndarray = attr.ib()
    c: jnp.ndarray = attr.ib(default=0.0)

    @classmethod
    def from_least_squares(cls, obj: LeastSquares):
        n = obj.X.shape[0]
        A = jnp.dot(obj.X.T, obj.X) / n
        b = jnp.dot(obj.X.T, obj.y) / n
        c = jnp.dot(obj.y, obj.y) / (2 * n)
        return Quadratic(A=A, b=b, c=c)

    @property
    def dim(self):
        return self.b.shape[0]

    @functools.partial(jit, static_argnums=(0,))
    def eval(self, x: jnp.ndarray):
        """Computes the value of the quadratic at the specified point."""
        return (
            0.5 * jnp.einsum("ij,i,j->", self.A, x, x)
            - jnp.dot(self.b, x)
            + self.c
        )

    def solve(self, eps: float = 0.0) -> jnp.ndarray:
        """Returns the optimum by solving `A x = b`."""
        return jnp.linalg.solve(self.A + eps * jnp.eye(self.dim), self.b)


def create_random_quadratics(
    dim: int,
    num_objectives: int,
    min_eig: float = 1.0,
    max_eig: float = 10.0,
    b_scale: float = 10.0,
    diagonal: bool = False,
) -> List[Quadratic]:
    """Creates quadratics with constraints on the spectrum and offset scale.

    Args:
        dim: The dimensionality of each objective.
        num_objectives: The number of random quadratics to generate.
        min_eig: The lower bound on the eigenvalues of the quadratics.
        max_eig: The upper bound on the eigenvalues of the quadratics.
        b_scale: The scale of the random normal vector of biases.
        diagonal: An indicator of whether quadratics should be diagonal.

    Returns:
        A list of `Quadratic` objects.
    """
    quadratics = []
    for _ in range(num_objectives):
        if diagonal:
            A = np.diag(min_eig + np.random.rand(dim) * (max_eig - min_eig))
        else:
            A = np.random.randn(dim, dim)
            A = 0.5 * (A + A.T)
            w, V = np.linalg.eigh(A)
            w = (max_eig - min_eig) * (w - w.min()) / (
                w.max() - w.min()
            ) + min_eig
            A = V.T.dot(np.diag(w)).dot(V)
        b = b_scale * np.random.randn(dim)
        q = Quadratic(A=jnp.asarray(A), b=jnp.asarray(b))
        quadratics.append(q)
    return quadratics


def create_global_quadratic(
    objectives: List[Quadratic], weights: np.ndarray
) -> Quadratic:
    """Creates the global objective as a weighted average of local objectives.

    Args:
      objectives: A list of local objective functions.
      weights: An array of non-negative weights.

    Returns:
      A weighted sum of the provided quadratics with the specified weights.
    """
    if len(objectives) != len(weights):
        raise ValueError(
            "The number of quadratics their weights must be equal."
        )
    if not np.all(weights >= 0):
        raise ValueError("Weights must be non-negative.")
    weights = jnp.asarray(weights) / jnp.sum(weights)
    A = jnp.einsum("ijk,i->jk", jnp.stack([q.A for q in objectives]), weights)
    b = jnp.einsum("ij,i->j", jnp.stack([q.b for q in objectives]), weights)
    c = jnp.einsum("i,i->", jnp.stack([q.c for q in objectives]), weights)
    return Quadratic(A=A, b=b, c=c)


def create_random_least_squares(
    num_objectives: int,
    batch_size: int,
    n_features: int = 100,
    n_informative: int = 10,
    n_samples_min: int = 100,
    n_samples_max: int = 1000,
    effective_rank: Optional[int] = None,
    bias_scale: float = 0.0,
    noise: float = 0.0,
    lam: float = 0.0,
    seed: int = 0,
) -> List[LeastSquares]:
    """Creates random `LeastSquares` problems.

    Args:
        num_objectives: The number of random least squares to generate.
        batch_size: The batch size used by each objective.
        n_features: The number of features in the generated data.
        n_informative: The number of informative features.
            See `sklearn.datasets.make_regression` for details.
        n_samples_min: The minimal number of samples per objective.
        n_samples_max: The maximal number of samples per objective.
        effective_rank: Optional approximate number of singular vectors required
            to explain most of the input data by linear combinations.
            See `sklearn.datasets.make_regression` for details.
        bias_scale: The scale of the bias term in the underlying linear model.
        noise: The standard deviation of the noise applied to the output.
        lam: The L2 regularization coefficient.
        seed: The random seed.

    Returns:
        A list of `LeastSquares` stochastic objectives.
    """
    rng = np.random.RandomState(seed)
    objectives = []
    for _ in range(num_objectives):
        X, y = datasets.make_regression(
            n_samples=np.random.randint(n_samples_min, n_samples_max),
            n_features=n_features,
            n_informative=n_informative,
            effective_rank=effective_rank,
            bias=rng.normal(scale=bias_scale),
            noise=noise,
            random_state=rng.randint(np.iinfo(np.int32).max),
        )
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        objectives.append(
            LeastSquares(
                X=jnp.array(X), y=jnp.array(y), batch_size=batch_size, lam=lam
            )
        )
    return objectives


def create_global_least_squares(
    objectives: List[LeastSquares],
    batch_size: Optional[int] = None,
    lam: Optional[float] = None,
) -> LeastSquares:
    """Creates the global least squares by concatenating all the data.

    Args:
        objectives: A list of `LeastSquares` objectives.
        batch_size: Optional batch size for the global objective.
            If None, the maximum batch size of the `objectives` is used.
        lam: Optional L2 regularization coefficient for the global objective.
            If None, the maximum coefficient of the `objectives` is used.

    Returns:
        A global `LeastSquares` objective.
    """
    X = jnp.vstack([o.X for o in objectives])
    y = jnp.hstack([o.y for o in objectives])
    if batch_size is None:
        batch_size = np.max([o.batch_size for o in objectives])
    if lam is None:
        lam = np.max([o.lam for o in objectives])
    return LeastSquares(X=X, y=y, batch_size=batch_size, lam=lam)
