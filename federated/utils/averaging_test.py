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
"""Tests for averaging utility functions."""

import jax.numpy as jnp
import numpy as np
from absl import flags
from absl.testing import absltest
from jax import random

from ..objectives.quadratic import (
    Quadratic,
    create_global_least_squares,
    create_global_quadratic,
    create_random_least_squares,
    create_random_quadratics,
)
from .averaging import (
    compute_optimal_convex_average,
    compute_optimal_hypercube_average,
    compute_posterior_average,
    compute_weighted_average,
)

FLAGS = flags.FLAGS


class ComputeAverageTests(absltest.TestCase):
    def test_optimal_convex_average(self):
        np.random.seed(0)
        weights = np.ones(5)
        local_objectives = create_random_quadratics(dim=10, num_objectives=5)
        global_objective = create_global_quadratic(local_objectives, weights)
        local_optima = [o.solve() for o in local_objectives]

        global_opt_state = global_objective.solve()
        global_opt_value = global_objective(global_opt_state)

        avg_state = compute_weighted_average(local_optima, weights)
        avg_value = global_objective(avg_state)

        opt_avg_state, opt_weights = compute_optimal_convex_average(
            local_optima, global_objective
        )
        np.testing.assert_almost_equal(np.sum(opt_weights), 1.0)

        opt_avg_value = global_objective(opt_avg_state)
        self.assertLess(opt_avg_value, avg_value)
        self.assertLess(global_opt_value, opt_avg_value)

    def test_optimal_hypercube_average(self):
        np.random.seed(0)
        weights = np.ones(5)
        local_objectives = create_random_quadratics(dim=10, num_objectives=5)
        global_objective = create_global_quadratic(local_objectives, weights)
        local_optima = [o.solve() for o in local_objectives]

        global_opt_state = global_objective.solve()
        global_opt_value = global_objective(global_opt_state)

        avg_state = compute_weighted_average(local_optima, weights)
        avg_value = global_objective(avg_state)

        opt_avg_state, opt_weights = compute_optimal_convex_average(
            local_optima, global_objective
        )
        np.testing.assert_almost_equal(np.sum(opt_weights), 1.0)
        opt_avg_value = global_objective(opt_avg_state)

        opt_ew_avg_state, opt_ew_weights = compute_optimal_hypercube_average(
            local_optima, global_objective
        )
        np.testing.assert_allclose(np.sum(opt_ew_weights, axis=0), 1.0)
        opt_ew_avg_value = global_objective(opt_ew_avg_state)

        self.assertLess(opt_avg_value, avg_value + 1e-5)
        self.assertLess(opt_ew_avg_value, opt_avg_value + 1e-5)
        self.assertLess(global_opt_value, opt_ew_avg_value + 1e-5)

    def test_posterior_average(self):
        np.random.seed(0)
        local_objectives = create_random_least_squares(
            num_objectives=5, batch_size=10, lam=1e-3
        )
        local_quadratics = [
            Quadratic.from_least_squares(o) for o in local_objectives
        ]
        weights = jnp.asarray([o.num_points for o in local_objectives])

        global_objective = create_global_least_squares(local_objectives)
        global_opt_state = global_objective.solve()

        local_optima = [o.solve() for o in local_objectives]
        state_covs = [jnp.linalg.pinv(q.A) for q in local_quadratics]
        posterior_mean, _ = compute_posterior_average(
            state_means=local_optima, state_covs=state_covs, weights=weights
        )
        np.testing.assert_allclose(posterior_mean, global_opt_state, rtol=1e-3)
