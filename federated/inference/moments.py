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
"""Utils for estimation of the posterior moments."""

import abc
import functools
from typing import Callable

import attr
import jax.numpy as jnp
from jax import jit


class MomentEstimator(abc.ABC):
    """The base abstract class for moment estimators."""

    @staticmethod
    def estimate_mean(samples: jnp.ndarray):
        """Returns the sample mean estimator of the first moment."""
        return jnp.mean(samples, axis=0)

    @abc.abstractmethod
    def estimate_cov(self, samples: jnp.ndarray):
        """Must return an estimate of the covariance matrix."""
        pass


@attr.s
class ShrinkageMomentEstimator(MomentEstimator):
    """An estimator that uses shrinkage for estimation of the covariance."""

    rho_fn: Callable[[jnp.ndarray], float] = attr.ib()

    @functools.partial(jit, static_argnums=(0,))
    def estimate_cov(self, samples: jnp.ndarray):
        """Estimates the covariance matrix using shrinkage."""
        n, d = samples.shape
        rho = self.rho_fn(samples)
        shrinkage_rho = 1 / (1 + (n - 1) * rho)
        sample_cov = jnp.cov(samples, rowvar=False)
        return shrinkage_rho * jnp.eye(d) + (1 - shrinkage_rho) * sample_cov
