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
"""Utils for posterior sampling."""

import abc
import functools
from typing import List, Tuple, Union

import attr
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from ..objectives.base import Objective, StochasticObjective
from ..objectives.quadratic import Quadratic
from .optimization import solve_sgd


class PosteriorSampler(abc.ABC):
    """Abstract base class for posterior samplers."""

    @abc.abstractmethod
    def sample(
        self,
        objective: Union[Objective, StochasticObjective],
        num_samples: int,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Must return a list of samples from the (approximate) posterior."""
        pass


class ExactQuadraticSampler(PosteriorSampler):
    """A sampler that produces exact samples from a quadratic posterior."""

    def sample(
        self, objective: Quadratic, num_samples: int, prng_key: jnp.ndarray
    ) -> jnp.ndarray:
        """Generates exact samples from a quadratic posterior (Gaussian)."""
        state_mean = objective.solve()
        state_cov = jnp.linalg.pinv(objective.A)
        samples = random.multivariate_normal(
            prng_key, state_mean, state_cov, shape=(num_samples,)
        )
        return samples


@attr.s
class IterateAveragedStochasticGradientSampler(PosteriorSampler):
    """A sampler that produces approximate samples using IASG.

    Args:
        avg_steps: The number of SGD steps averaged to produce a sample.
        burnin_steps: The number of initial SGD steps used for burn-in.
        discard_steps: The number of SGD steps discarded between samples.

    References:
        1.  Stochastic gradient descent as approximate bayesian inference.
            S. Mandt, M. D. Hoffman, D. M. Blei. JMLR, 2017.
            https://www.jmlr.org/papers/volume18/17-214/17-214
    """

    avg_steps: int = attr.ib()
    burnin_steps: int = attr.ib()
    learning_rate: float = attr.ib()
    discard_steps: int = attr.ib(default=0)
    momentum: float = attr.ib(default=0.0)

    @functools.partial(jit, static_argnums=(0, 1, 2))
    def sample(
        self,
        objective: StochasticObjective,
        num_samples: int,
        prng_key: jnp.ndarray,
        init_state: jnp.ndarray,
        init_momentum: jnp.ndarray,
    ) -> jnp.ndarray:
        # Burn-in.
        x, v, _, prng_key = solve_sgd(
            objective=objective,
            prng_key=prng_key,
            steps=self.burnin_steps,
            init_state=init_state,
            init_momentum=jnp.zeros_like(init_state),
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
        # Sample.
        samples = []
        for _ in range(num_samples):
            x, v, x_avg, prng_key = solve_sgd(
                objective=objective,
                prng_key=prng_key,
                steps=self.avg_steps,
                init_state=x,
                init_momentum=v,
                learning_rate=self.learning_rate,
                momentum=self.momentum,
            )
            samples.append(x_avg)
            # Discard the specified number of steps, if necessary.
            if self.discard_steps > 0:
                x, v, _, prng_key = solve_sgd(
                    objective=objective,
                    prng_key=prng_key,
                    steps=self.discard_steps,
                    init_state=x,
                    init_momentum=v,
                    learning_rate=self.learning_rate,
                    momentum=self.momentum,
                )
        return jnp.stack(samples)


class HamiltonianMonteCarlo(PosteriorSampler):
    """A sampler that produces approximate samples using HMC."""

    # TODO: implement.


class StochasticGradientHamiltonianMonteCarlo(PosteriorSampler):
    """A sampler that produces approximate samples using SG-HMC."""

    # TODO: implement.
