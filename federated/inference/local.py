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
"""Functions for local posterior inference."""

from collections import defaultdict
from typing import Callable, Union

import jax.numpy as jnp

from ..objectives.base import StochasticObjective
from ..objectives.quadratic import LeastSquares, Quadratic
from .moments import MomentEstimator
from .optimization import solve_sgd
from .sampling import PosteriorSampler


def compute_exact_delta(
    objective: Union[Quadratic, LeastSquares], init_state: jnp.ndarray
):
    """Computes the client delta exactly for a quadratic objective.

    The delta is computed as `Sigma^{-1} (init_state - mu)`, where `Sigma` is
    the exact posterior variance and mu is the exact posterior mean.

    Args:
        objective: A quadratic objective function.
        init_state: The initial state for which the exact delta is computed.
    """
    # Convert objective to Quadratic, if necessary.
    if isinstance(objective, LeastSquares):
        objective = Quadratic.from_least_squares(objective)

    posterior_mean = objective.solve()
    posterior_cov_inv = objective.A

    return jnp.dot(posterior_cov_inv, init_state - posterior_mean)


def compute_fed_avg_delta(
    objective: StochasticObjective,
    init_state: jnp.ndarray,
    prng_key: jnp.ndarray,
    num_steps: int,
    *,
    learning_rate_schedule: Callable[[int], float],
    momentum: float = 0.0,
    noise_scale: float = 0.0,
) -> jnp.ndarray:
    """The standard SGD-based computation of client delta (used in FedAvg).

    The delta is computed by simply running SGD for the specified number of
    steps and returning `init_state - final_sgd_iterate`.

    Args:
        objective: The stochastic objective proportional to the posterior.
        init_state: The initial state with respect to which delta is computed.
        prng_key: A key for random number generation.
        num_steps: The number of local SGD steps.
        learning_rate_schedule: The learning rate schedule function for SGD.
        momentum: The momentum used by SGD.
        noise_scale: The scale of the injected Gaussian gradient noise.

    Returns:
        The computed client delta.
    """
    (x, _), _ = solve_sgd(
        objective=objective,
        prng_key=prng_key,
        init_states=init_state,
        learning_rate_schedule=learning_rate_schedule,
        steps=num_steps,
        momentum=momentum,
        noise_scale=noise_scale,
    )
    return init_state - x


def compute_mb_sgd_delta(
    objective: StochasticObjective,
    init_state: jnp.ndarray,
    prng_key: jnp.ndarray,
    num_grads: int,
    *,
    learning_rate_schedule: Callable[[int], float],
    momentum: float = 0.0,
    noise_scale: float = 0.0,
) -> jnp.ndarray:
    """Computes MB-SGD delta for the given objective and initial state.

    Computes delta by averaging `num_grads` stochastic gradients computed on the
    `objective` at `init_state`.

    Args:
        objective: The stochastic objective proportional to the posterior.
        init_state: The initial state with respect to which delta is computed.
        prng_key: A key for random number generation.
        num_grads: The number of local stochastic gradients to be averaged.
        learning_rate_schedule: The learning rate schedule function for SGD.
        momentum: The momentum used by SGD.
        noise_scale: The scale of the injected Gaussian gradient noise.

    Returns:
        The computed client delta.
    """
    init_states = jnp.tile(jnp.expand_dims(init_state, axis=0), (num_grads, 1))
    (xs, _), _ = solve_sgd(
        objective,
        prng_key=prng_key,
        init_states=init_states,
        learning_rate_schedule=learning_rate_schedule,
        steps=1,
        momentum=momentum,
        noise_scale=noise_scale,
    )
    return init_state - jnp.mean(xs, axis=0)


def compute_post_avg_delta(
    objective: StochasticObjective,
    init_state: jnp.ndarray,
    prng_key: jnp.ndarray,
    num_samples: int,
    *,
    sampler: PosteriorSampler,
    moment_estimator: MomentEstimator,
) -> jnp.ndarray:
    """Computes posterior delta for the given objective and initial state.

    Uses a `sampler` to produce posterior samples, then a `moment_estimator`
    to compute estimates of the posterior mean and covariance from the samples.
    Finally, computes client delta by solving `C x = init_state - m`, where
    `C` and `m` are the posterior covariance and mean estimates, respectively.

    Args:
        objective: The stochastic objective proportional to the posterior.
        init_state: The initial state with respect to which delta is computed.
        prng_key: A key for random number generation.
        num_samples: The number of samples to collect for estimation.
        sampler: A posterior sampler.
        moment_estimator: A moment estimator.

    Returns:
        The computed client delta.
    """
    # Produce (approximate) samples from the posterior.
    samples = sampler.sample(
        objective, prng_key, num_samples, init_state=init_state
    )

    # Estimate posterior moments.
    posterior_mean_est = moment_estimator.estimate_mean(samples)
    posterior_cov_est = moment_estimator.estimate_cov(samples)

    # Compute and return delta.
    return jnp.linalg.solve(posterior_cov_est, init_state - posterior_mean_est)


def compute_post_avg_delta_dp(
    objective: StochasticObjective,
    init_state: jnp.ndarray,
    prng_key: jnp.ndarray,
    num_samples: int,
    *,
    sampler: PosteriorSampler,
    rho_fn: Callable[[jnp.ndarray], float],
) -> jnp.ndarray:
    """Computes posterior delta for the given objective and initial state.

    Similar to `compute_post_avg_delta`, computes the client delta based on
    approximate posterior samples by solving `C x = init_state - m`. However,
    additionally exploits the linear algebraic structure of the shrinkage
    covariance and efficiently computes deltas using dynamic programming.

    Args:
        objective: The stochastic objective proportional to the posterior.
        init_state: The initial state with respect to which delta is computed.
        prng_key: A key for random number generation.
        num_samples: The number of samples to collect for estimation.
        sampler: A posterior sampler.
        rho_fn: An estimator for the shrinkage parameter.

    Returns:
        The computed client delta.
    """
    # Produce (approximate) samples from the posterior.
    samples = sampler.sample(
        objective, prng_key, num_samples, init_state=init_state
    )

    # Compute delta from the samples using dynamic programming.
    rho = rho_fn(samples)
    dp = defaultdict(list)
    samples_ra = samples[0]
    delta = init_state - samples_ra
    for t, s in enumerate(samples[1:], 2):
        u = v = s - samples_ra
        # Compute v_{t-1,t} (solution of `sigma_{t-1} x = u_t`).
        for k, (v_k, dot_uk_vk) in enumerate(zip(dp["v"], dp["dot_u_v"]), 2):
            gamma_k = rho * (k - 1) / k
            v = v - gamma_k * jnp.dot(v_k, u) / (1 + gamma_k * dot_uk_vk) * v_k
        # Compute `dot(u_t, v_t)` and `dot(u_t, delta_t)`.
        dot_u_v = jnp.dot(u, v)
        dot_u_d = jnp.dot(u, delta)
        # Compute delta.
        gamma = rho * (t - 1) / t
        c = gamma * (t * dot_u_d - dot_u_v) / (1 + gamma * dot_u_v)
        delta -= (1 + c) * v / t
        # Update the DP state.
        dp["v"].append(v)
        dp["dot_u_v"].append(dot_u_v)
        # Update running mean of the samples.
        samples_ra = ((t - 1) * samples_ra + s) / t

    return delta * (1 + (num_samples - 1) * rho)
