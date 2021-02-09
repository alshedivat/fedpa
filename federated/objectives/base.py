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
from typing import Any, Dict, Tuple

import attr
import jax.numpy as jnp
from jax import grad, jit, random, vmap

# Types.
Dataset = Tuple[jnp.ndarray, jnp.ndarray]
ObjectiveParams = Tuple[jnp.ndarray, ...]


class Objective(abc.ABC):
    """Abstract base class for objective functions."""

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {}

    @classmethod
    @functools.partial(jit, static_argnums=(0,))
    def _veval(
        cls, params: ObjectiveParams, x: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        _eval = functools.partial(cls.eval, params, **kwargs)
        return vmap(_eval)(x)

    @classmethod
    @functools.partial(jit, static_argnums=(0,))
    def _vgrad(
        cls, params: ObjectiveParams, x: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        _eval = functools.partial(cls.eval, params, **kwargs)
        return vmap(grad(_eval))(x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the value of the objective at `x`."""
        # Add batch dimension, if necessary.
        squeeze = False
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            squeeze = True
        # Run a vectorized version of grad.
        value = self._veval(self.params, x, **self.kwargs)
        if squeeze:
            value = jnp.squeeze(value)
        return value

    def grad(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the gradient of the objective at `x`."""
        # Add batch dimension, if necessary.
        squeeze = False
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            squeeze = True
        # Run a vectorized version of grad.
        value = self._vgrad(self.params, x, **self.kwargs)
        if squeeze:
            value = jnp.squeeze(value)
        return value

    @property
    @abc.abstractmethod
    def params(self) -> ObjectiveParams:
        """Must return a tuple of parameters of the objective"""
        pass

    @staticmethod
    @abc.abstractmethod
    def eval(params: ObjectiveParams, x: jnp.ndarray) -> jnp.ndarray:
        """Must return the value of the objective at `x`."""
        pass


@attr.s(eq=False)
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
    def kwargs(self) -> Dict[str, Any]:
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
        prng_key: jnp.ndarray,
        x: jnp.ndarray,
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
        prng_keys: jnp.ndarray,
        x: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        _eval = functools.partial(cls._eval, batch_size, data, **kwargs)
        return vmap(_eval)(prng_keys, x)

    @classmethod
    @functools.partial(jit, static_argnums=(0, 1))
    def _grad(
        cls,
        batch_size: int,
        data: Dataset,
        prng_key: jnp.ndarray,
        x: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        _eval = functools.partial(
            cls._eval, batch_size, data, prng_key, **kwargs
        )
        return grad(_eval)(x)

    @classmethod
    @functools.partial(jit, static_argnums=(0, 1))
    def _vgrad(
        cls,
        batch_size: int,
        data: Dataset,
        prng_keys: jnp.ndarray,
        x: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        _grad = functools.partial(cls._grad, batch_size, data, **kwargs)
        return vmap(_grad)(prng_keys, x)

    def __call__(
        self, x: jnp.ndarray, prng_key: jnp.ndarray, deterministic: bool = False
    ) -> jnp.ndarray:
        """Computes the (stochastic) value of the objective at `x`."""
        # Add batch dimension, if necessary.
        squeeze = False
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            squeeze = True
        # Run a vectorized version of eval.
        subkeys = random.split(prng_key, x.shape[0])
        batch_size = self.num_points if deterministic else self.batch_size
        args = batch_size, self.data, jnp.stack(subkeys)
        value = self._veval(*args, x, **self.kwargs)
        if squeeze:
            value = jnp.squeeze(value)
        return value

    def grad(
        self, x: jnp.ndarray, prng_key: jnp.ndarray, deterministic: bool = False
    ) -> jnp.ndarray:
        """Computes the (stochastic) gradient of the objective at `x`."""
        # Add batch dimension, if necessary.
        squeeze = False
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            squeeze = True
        # Run a vectorized version of grad.
        subkeys = random.split(prng_key, x.shape[0])
        batch_size = self.num_points if deterministic else self.batch_size
        args = batch_size, self.data, jnp.stack(subkeys)
        value = self._vgrad(*args, x, **self.kwargs)
        if squeeze:
            value = jnp.squeeze(value)
        return value

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Must return the dimensionality of the problem."""
        pass

    @staticmethod
    @abc.abstractmethod
    def eval(x: jnp.ndarray, data_batch: Dataset, **kwargs) -> jnp.ndarray:
        """Must compute objective value at `x` given `data_batch`."""
        pass

    @abc.abstractmethod
    def solve(self) -> jnp.ndarray:
        """Must return the minimizer of the objective."""
        pass
