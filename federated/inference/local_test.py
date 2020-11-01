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
"""Tests for local inference."""

import numpy as np
from absl import flags
from absl.testing import absltest
from jax import random

from ..objectives.quadratic import (
    create_global_least_squares,
    create_random_least_squares,
)
from .local import (
    compute_exact_delta,
    compute_fed_avg_delta,
    compute_mb_sgd_delta,
    compute_post_avg_delta,
    compute_post_avg_delta_dp,
)
from .moments import ShrinkageMomentEstimator
from .sampling import ExactQuadraticSampler

FLAGS = flags.FLAGS


def _setup():
    d = 5
    batch_size = 10
    num_clients = 5

    def client_lr_schedule(_):
        return 0.03

    client_objectives = create_random_least_squares(
        num_objectives=num_clients,
        batch_size=batch_size,
        n_features=(d - 1),
        lam=1e-3,
    )
    global_objective = create_global_least_squares(client_objectives)

    return client_objectives, global_objective, client_lr_schedule


class ComputeDeltaTests(absltest.TestCase):
    def test_compute_fed_avg_delta(self):
        np.random.seed(0)
        num_local_steps = 1000
        client_objectives, global_objective, client_lr_schedule = _setup()

        prng_key = random.PRNGKey(np.random.randint(1000))
        init_state = random.normal(prng_key, (global_objective.dim,))
        for obj in client_objectives:
            opt = obj.solve()
            prng_key, subkey = random.split(prng_key)
            fed_avg_delta = compute_fed_avg_delta(
                objective=obj,
                init_state=init_state,
                prng_key=subkey,
                num_steps=num_local_steps,
                learning_rate_schedule=client_lr_schedule,
            )
            np.testing.assert_allclose(
                fed_avg_delta, (init_state - opt), rtol=1e-1
            )

    def test_compute_mb_sgd_delta(self):
        np.random.seed(0)
        num_grads = 10000
        client_objectives, global_objective, client_lr_schedule = _setup()

        prng_key = random.PRNGKey(np.random.randint(1000))
        init_state = random.normal(prng_key, (global_objective.dim,))
        for obj in client_objectives:
            prng_key, subkey = random.split(prng_key)
            mb_sgd_delta = compute_mb_sgd_delta(
                objective=obj,
                init_state=init_state,
                prng_key=subkey,
                num_grads=num_grads,
                learning_rate_schedule=client_lr_schedule,
            )
            lr = client_lr_schedule(0)
            true_grad = obj.grad(init_state, prng_key, deterministic=True)
            np.testing.assert_allclose(mb_sgd_delta, lr * true_grad, rtol=1e-1)

    def test_compute_post_avg_delta(self):
        np.random.seed(0)
        num_samples = 1000
        rho = 1.0
        client_objectives, global_objective, client_lr_schedule = _setup()

        sampler = ExactQuadraticSampler()
        moment_estimator = ShrinkageMomentEstimator(rho_fn=lambda _: rho)

        prng_key = random.PRNGKey(np.random.randint(1000))
        init_state = random.normal(prng_key, (global_objective.dim,))
        for obj in client_objectives:
            prng_key, subkey = random.split(prng_key)
            post_avg_delta = compute_post_avg_delta(
                objective=obj,
                init_state=init_state,
                prng_key=subkey,
                num_samples=num_samples,
                sampler=sampler,
                moment_estimator=moment_estimator,
            )
            exact_delta = compute_exact_delta(
                objective=obj, init_state=init_state
            )
            np.testing.assert_allclose(post_avg_delta, exact_delta, rtol=1e-0)

    def test_compute_post_avg_delta_dp(self):
        np.random.seed(0)
        num_samples = 10
        rho = 1.0
        client_objectives, global_objective, client_lr_schedule = _setup()

        sampler = ExactQuadraticSampler()
        moment_estimator = ShrinkageMomentEstimator(rho_fn=lambda _: rho)

        prng_key = random.PRNGKey(np.random.randint(1000))
        init_state = random.normal(prng_key, (global_objective.dim,))
        for obj in client_objectives:
            prng_key, subkey = random.split(prng_key)
            post_avg_delta = compute_post_avg_delta(
                objective=obj,
                init_state=init_state,
                prng_key=subkey,
                num_samples=num_samples,
                sampler=sampler,
                moment_estimator=moment_estimator,
            )
            post_avg_delta_dp = compute_post_avg_delta_dp(
                objective=obj,
                init_state=init_state,
                prng_key=subkey,
                num_samples=num_samples,
                sampler=sampler,
                rho_fn=lambda _: rho,
            )
            np.testing.assert_allclose(
                post_avg_delta, post_avg_delta_dp, rtol=1e-3
            )
