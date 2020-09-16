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
import math
from typing import Optional, Union

import attr
import jax.numpy as jnp
from jax import random

from ..objectives.base import Objective, StochasticObjective
from ..objectives.quadratic import Quadratic
from .optimization import solve_sgd


class PosteriorSampler(abc.ABC):
    """Abstract base class for posterior samplers."""

    @abc.abstractmethod
    def sample(
        self,
        objective: Union[Objective, StochasticObjective],
        prng_key: jnp.ndarray,
        num_samples: int = 1,
    ) -> jnp.ndarray:
        """Must return a list of samples from the (approximate) posterior."""
        pass


class ExactQuadraticSampler(PosteriorSampler):
    """A sampler that produces exact samples from a quadratic posterior."""

    def sample(
        self, objective: Quadratic, prng_key: jnp.ndarray, num_samples: int = 1
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

    def sample(
        self,
        objective: StochasticObjective,
        prng_key: jnp.ndarray,
        num_samples: int = 1,
        parallel_chains: int = 1,
        init_state: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if init_state is None:
            init_states = jnp.zeros((parallel_chains, objective.dim))
        else:
            init_states = jnp.tile(
                jnp.expand_dims(init_state, axis=0), (parallel_chains, 1)
            )
        init_momenta = jnp.zeros_like(init_states)

        def _lr_schedule(_):
            return self.learning_rate

        # Burn-in.
        prng_key, subkey = random.split(prng_key)
        xs, vs, _ = solve_sgd(
            objective=objective,
            prng_key=subkey,
            init_states=init_states,
            init_momenta=init_momenta,
            steps=self.burnin_steps,
            learning_rate_schedule=_lr_schedule,
            momentum=self.momentum,
        )

        # Sample.
        samples = []
        for i in range(math.ceil(num_samples / parallel_chains)):
            batch_size = min(parallel_chains, num_samples - i * parallel_chains)
            prng_key, subkey = random.split(prng_key)
            xs, vs, x_avgs = solve_sgd(
                objective=objective,
                prng_key=subkey,
                init_states=xs[:batch_size],
                init_momenta=vs[:batch_size],
                steps=self.avg_steps,
                learning_rate_schedule=_lr_schedule,
                momentum=self.momentum,
            )
            samples.append(x_avgs)
            # Discard the specified number of steps, if necessary.
            if self.discard_steps > 0:
                prng_key, subkey = random.split(prng_key)
                xs, vs, _ = solve_sgd(
                    objective=objective,
                    prng_key=subkey,
                    init_states=xs,
                    init_momenta=vs,
                    steps=self.discard_steps,
                    learning_rate_schedule=_lr_schedule,
                    momentum=self.momentum,
                )
        return jnp.concatenate(samples, axis=0)


@attr.s
class StochasticGradientLangevinDynamics(PosteriorSampler):
    """A sampler that produces approximate samples using SGLD."""

    # TODO: implement.


@attr.s
class HamiltonianMonteCarlo(PosteriorSampler):
    """A sampler that produces approximate samples using HMC."""

    # TODO: implement.


@attr.s
class StochasticGradientHamiltonianMonteCarlo(PosteriorSampler):
    """A sampler that produces approximate samples using SG-HMC."""

    # TODO: implement.


# Aliases.
HMC = HamiltonianMonteCarlo
SGLD = StochasticGradientLangevinDynamics
SGHMC = StochasticGradientHamiltonianMonteCarlo
IASG = IterateAveragedStochasticGradientSampler
