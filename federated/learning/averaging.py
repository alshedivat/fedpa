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

import jax.numpy as jnp
from jax import jit


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
