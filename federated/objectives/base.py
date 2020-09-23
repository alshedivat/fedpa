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
"""Base classes and functionality for objective functions."""

import abc
import functools
from typing import Tuple

import attr
import jax.numpy as jnp
from jax import grad, jit, lax, random, vmap

# Types.
Dataset = Tuple[jnp.ndarray, jnp.ndarray]


class Objective(abc.ABC):
    """Abstract base class for objective functions."""

    @property
    @abc.abstractmethod
    def dim(self):
        """Must return the dimensionality of the problem."""
        pass

    @abc.abstractmethod
    def eval(self, x: jnp.ndarray):
        """Must return the value of the objective at `x`."""
        pass

    @functools.partial(jit, static_argnums=(0,))
    def _veval(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(vmap(self.eval)(x))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the value of the objective at `x`."""
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        return self._veval(x)

    @functools.partial(jit, static_argnums=(0,))
    def _vgrad(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(vmap(grad(self.eval))(x))

    def grad(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the gradient of the objective at `x`."""
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        return self._vgrad(x)


@attr.s
class StochasticObjective(abc.ABC):
    """Abstract base class for stochastic objective functions.

    Stochastic objectives must be build around a dataset of input-output pairs.
    Whenever the objective (or its gradient) called on an input `x`, it computes
    the stochastic value (or stochastic gradient) based on a batch of data
    randomly sampled from the underlying dataset.
    """

    X: jnp.ndarray = attr.ib()
    y: jnp.ndarray = attr.ib()
    batch_size: int = attr.ib()

    @property
    def data(self):
        return self.X, self.y

    @property
    @abc.abstractmethod
    def dim(self):
        """Must return the dimensionality of the problem."""
        pass

    @property
    def kwargs(self):
        return {}

    @property
    def num_points(self) -> int:
        return self.X.shape[0]

    @staticmethod
    @functools.partial(jit, static_argnums=(0,))
    def _sample_batch(
        batch_size: int, data: Dataset, prng_key: jnp.ndarray
    ) -> Dataset:
        x, y = data
        num_points = x.shape[0]
        batch_indices = random.choice(
            prng_key, num_points, (batch_size,), replace=False
        )
        x_batch = jnp.take(x, batch_indices, axis=0)
        y_batch = jnp.take(y, batch_indices, axis=0)
        return x_batch, y_batch

    @classmethod
    @functools.partial(jit, static_argnums=(0, 1))
    def _eval(
        cls,
        batch_size: int,
        data: Dataset,
        x: jnp.ndarray,
        prng_key: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        data_batch = cls._sample_batch(batch_size, data, prng_key)
        return cls.eval(x, data_batch, **kwargs)

    @classmethod
    @functools.partial(jit, static_argnums=(0, 1))
    def _veval(
        cls,
        batch_size: int,
        data: Dataset,
        x: jnp.ndarray,
        prng_keys: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        _eval = functools.partial(cls._eval, batch_size, data, **kwargs)
        return jnp.squeeze(vmap(_eval)(x, prng_keys))

    @classmethod
    @functools.partial(jit, static_argnums=(0, 1))
    def _grad(
        cls,
        batch_size: int,
        data: Dataset,
        x: jnp.ndarray,
        prng_key: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        data_batch = cls._sample_batch(batch_size, data, prng_key)
        return grad(cls.eval, argnums=0)(x, data_batch, **kwargs)

    @classmethod
    @functools.partial(jit, static_argnums=(0, 1))
    def _vgrad(
        cls,
        batch_size: int,
        data: Dataset,
        x: jnp.ndarray,
        prng_keys: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        _grad = functools.partial(cls._grad, batch_size, data, **kwargs)
        return jnp.squeeze(vmap(_grad)(x, prng_keys))

    def __call__(
        self, x: jnp.ndarray, prng_key: jnp.ndarray, deterministic: bool = False
    ) -> jnp.ndarray:
        """Computes the (stochastic) value of the objective at `x`."""
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        subkeys = random.split(prng_key, x.shape[0])
        batch_size = self.num_points if deterministic else self.batch_size
        return self._veval(
            batch_size, self.data, x, jnp.stack(subkeys), **self.kwargs
        )

    def grad(
        self, x: jnp.ndarray, prng_key: jnp.ndarray, deterministic: bool = False
    ) -> jnp.ndarray:
        """Computes the (stochastic) gradient of the objective at `x`."""
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        subkeys = random.split(prng_key, x.shape[0])
        batch_size = self.num_points if deterministic else self.batch_size
        return self._vgrad(
            batch_size, self.data, x, jnp.stack(subkeys), **self.kwargs
        )

    @classmethod
    @abc.abstractmethod
    def eval(cls, x: jnp.ndarray, data_batch: Dataset, **kwargs) -> jnp.ndarray:
        """Must compute objective value at `x` given `data_batch`."""
        pass
