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
"""Tests for federated learning utility functions."""

import numpy as np
from absl import flags
from absl.testing import absltest
from jax import random

from ..objectives.quadratic import (
    Quadratic,
    create_global_least_squares,
    create_random_least_squares,
)
from .algorithms import (
    create_fed_avg,
    create_mb_sgd,
    create_post_avg_exact,
    create_post_avg_iasg,
)

FLAGS = flags.FLAGS


def _setup():
    d = 5
    batch_size = 10
    num_clients = 100
    num_rounds = 5
    num_clients_per_round = 2
    num_client_steps_per_round = 5
    client_momentum = 0.0
    server_momentum = 0.0

    def client_lr_schedule(_):
        return 0.1

    def server_lr_schedule(_):
        return 1.0

    client_objectives = create_random_least_squares(
        num_objectives=num_clients,
        batch_size=batch_size,
        n_features=(d - 1),
        lam=1e-3,
    )
    global_objective = create_global_least_squares(client_objectives)
    global_quadratic = Quadratic.from_least_squares(global_objective)

    return (
        num_rounds,
        num_clients_per_round,
        num_client_steps_per_round,
        client_lr_schedule,
        client_momentum,
        server_lr_schedule,
        server_momentum,
        client_objectives,
        global_objective,
        global_quadratic,
    )


class MBSGDTests(absltest.TestCase):
    def test_mb_sgd_end_to_end(self):
        np.random.seed(0)

        (
            num_rounds,
            num_clients_per_round,
            num_client_grads_per_round,
            client_lr_schedule,
            client_momentum,
            server_lr_schedule,
            server_momentum,
            client_objectives,
            global_objective,
            global_quadratic,
        ) = _setup()

        fed_learn = create_mb_sgd(
            client_grads_per_round=num_client_grads_per_round,
            client_learning_rate_schedule=client_lr_schedule,
            server_learning_rate_schedule=server_lr_schedule,
            client_momentum=client_momentum,
            server_momentum=server_momentum,
        )

        prng_key = random.PRNGKey(np.random.randint(1000))
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, (client_objectives[0].dim,))
        trajectory, _ = fed_learn(
            client_objectives,
            init_state,
            prng_key,
            num_rounds,
            num_clients_per_round,
        )
        final_state = trajectory[-1].x

        init_global_value = global_quadratic(init_state)
        final_global_value = global_quadratic(final_state)
        self.assertLess(final_global_value, init_global_value)


class FedAvgTests(absltest.TestCase):
    def test_fed_avg_end_to_end(self):
        np.random.seed(0)

        (
            num_rounds,
            num_clients_per_round,
            num_client_steps_per_round,
            client_lr_schedule,
            client_momentum,
            server_lr_schedule,
            server_momentum,
            client_objectives,
            global_objective,
            global_quadratic,
        ) = _setup()

        fed_learn = create_fed_avg(
            client_steps_per_round=num_client_steps_per_round,
            client_learning_rate_schedule=client_lr_schedule,
            server_learning_rate_schedule=server_lr_schedule,
            client_momentum=client_momentum,
            server_momentum=server_momentum,
        )

        prng_key = random.PRNGKey(np.random.randint(1000))
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, (client_objectives[0].dim,))
        trajectory, _ = fed_learn(
            client_objectives,
            init_state,
            prng_key,
            num_rounds,
            num_clients_per_round,
        )
        final_state = trajectory[-1].x

        init_global_value = global_quadratic(init_state)
        final_global_value = global_quadratic(final_state)
        self.assertLess(final_global_value, init_global_value)


class PostAvgTests(absltest.TestCase):
    def test_post_avg_exact_end_to_end(self):
        np.random.seed(0)
        num_client_samples_per_round = 10

        (
            num_rounds,
            num_clients_per_round,
            _,
            client_lr_schedule,
            client_momentum,
            server_lr_schedule,
            server_momentum,
            client_objectives,
            global_objective,
            global_quadratic,
        ) = _setup()

        fed_learn = create_post_avg_exact(
            client_samples_per_round=num_client_samples_per_round,
            server_learning_rate_schedule=server_lr_schedule,
            server_momentum=server_momentum,
        )

        prng_key = random.PRNGKey(np.random.randint(1000))
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, (client_objectives[0].dim,))
        trajectory, _ = fed_learn(
            client_objectives,
            init_state,
            prng_key,
            num_rounds,
            num_clients_per_round,
        )
        final_state = trajectory[-1].x

        init_global_value = global_quadratic(init_state)
        final_global_value = global_quadratic(final_state)
        self.assertLess(final_global_value, init_global_value)

    def test_post_avg_iasg_end_to_end(self):
        np.random.seed(0)

        (
            num_rounds,
            num_clients_per_round,
            _,
            client_lr_schedule,
            client_momentum,
            server_lr_schedule,
            server_momentum,
            client_objectives,
            global_objective,
            global_quadratic,
        ) = _setup()

        fed_learn = create_post_avg_iasg(
            client_samples_per_round=10,
            client_iasg_avg_steps=100,
            client_iasg_burnin_steps=100,
            client_iasg_learning_rate=0.1,
            client_iasg_discard_steps=100,
            client_iasg_momentum=0.0,
            server_learning_rate_schedule=server_lr_schedule,
            server_momentum=server_momentum,
            shrinkage_rho=0.01,
        )

        prng_key = random.PRNGKey(np.random.randint(1000))
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, (client_objectives[0].dim,))
        trajectory, _ = fed_learn(
            client_objectives,
            init_state,
            prng_key,
            num_rounds,
            num_clients_per_round,
        )
        final_state = trajectory[-1].x

        init_global_value = global_quadratic(init_state)
        final_global_value = global_quadratic(final_state)
        self.assertLess(final_global_value, init_global_value)
