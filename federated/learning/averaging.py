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
"""Functions for computing various state averages."""

from typing import List, Tuple

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jax import jit

from ..objectives.quadratic import Quadratic


def compute_weighted_average(
    states: List[jnp.ndarray], weights: jnp.ndarray
) -> jnp.ndarray:
    """Computes a weighted average of multiple state vectors.

    Args:
      states: A list of arrays that represent states.
      weights: An array of weights.

    Returns:
      A tuple of weighted average states and weights.
    """
    return jnp.einsum("ij,i->j", jnp.stack(states), weights / jnp.sum(weights))


def compute_optimal_convex_average(
    states: List[np.ndarray], objective: Quadratic, abstol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes weighted average of the states that minimizes the objective.

    The optimal weights are computed by solving a QP.

    Args:
      states: A list of arrays that represent states.
      objective: The global quadratic objective.
      abstol: Absolute value of the QP solver's tolerance.

    Returns:
      A tuple of weighted average states and optimal weights.
    """
    X = np.stack(states)
    A, b = objective.A, objective.b

    # Define the QP for optimal weights.
    c = cp.Variable(len(states))
    x = X.T @ c

    # Solve QP.
    qp_objective = cp.Minimize(0.5 * cp.quad_form(x, A) - b @ x)
    constraints = [c >= 0, cp.sum(c) == 1]
    prob = cp.Problem(qp_objective, constraints)
    prob.solve()

    optimal_weights = c.value
    avg_state = x.value

    return avg_state, optimal_weights


def compute_optimal_hypercube_average(
    states: List[np.ndarray], objective: Quadratic, abstol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes an element-wise weighted average of the states that minimizes the objective.

    The optimal weights are computed by solving a QP.

    Args:
      states: A list of arrays that represent states.
      objective: The global quadratic objective.
      abstol: Absolute value of the QP solver's tolerance.

    Returns:
      A tuple of weighted average states and optimal weights.
    """
    X = np.stack(states)
    A, b, dim = objective.A, objective.b, objective.dim

    # Define the QP for optimal weights.
    c = cp.Variable((len(states), dim))
    x = cp.sum(cp.multiply(X, c), axis=0)

    # Solve QP.
    qp_objective = cp.Minimize(0.5 * cp.quad_form(x, A) - b @ x)
    constraints = [c >= 0, cp.sum(c, axis=0) == 1]
    prob = cp.Problem(qp_objective, constraints)
    prob.solve()

    optimal_weights = c.value
    avg_state = x.value

    return avg_state, optimal_weights


@jit
def compute_posterior_average(
    state_means: List[jnp.ndarray],
    state_covs: List[jnp.ndarray],
    weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the mean of the approximate global posterior of the state.

    Assuming that local posterior distributions are Gaussian, the global
    posterior is also Gaussian and is estimated multiplicatively:

        ```
        global_cov  = inv(sum(local_cov_inv))
        global_mean = global_cov * sum(local_cov_inv * local_mean)
        ```

    Args:
      state_means: A list of arrays that represent state means.
      state_covs: A list of arrays that represent state covariances.
      weights: An array of weights.

    Returns:
      A tuple of posterior mean and covariance arrays.
    """
    state_cov_invs = [
        w * jnp.linalg.pinv(c) for c, w in zip(state_covs, weights)
    ]
    posterior_means_sum = sum(
        [jnp.dot(c, m) for m, c in zip(state_means, state_cov_invs)]
    )
    posterior_cov = jnp.linalg.pinv(sum(state_cov_invs))
    posterior_mean = jnp.dot(posterior_cov, posterior_means_sum)
    return posterior_mean, posterior_cov
