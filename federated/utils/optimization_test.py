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
"""Tests for optimization utility functions."""

import jax.numpy as jnp
import numpy as np
from absl import flags
from absl.testing import absltest
from jax import random

from ..objectives.quadratic import LeastSquares
from .optimization import solve_sgd

FLAGS = flags.FLAGS


class SolveSGDTests(absltest.TestCase):
    def test_solve_gd_on_least_squares(self):
        rng = np.random.RandomState(seed=0)
        prng_seed = 0
        n, d = 100, 10
        batch_size = n  # to use GD instead of SGD

        X = rng.randn(n, d)
        y = X.dot(rng.randn(d)) + rng.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size)
        opt = obj.solve()

        prng_key = random.PRNGKey(prng_seed)
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, opt.shape, dtype=opt.dtype)

        def _lr_schedule(_):
            return 0.1

        (x, v), x_avg = solve_sgd(
            objective=obj,
            prng_key=prng_key,
            steps=1000,
            init_states=init_state,
            learning_rate_schedule=_lr_schedule,
            momentum=0.9,
            noise_scale=0.0,
        )

        self.assertSequenceEqual(x.shape, opt.shape)
        self.assertSequenceEqual(v.shape, opt.shape)
        self.assertSequenceEqual(x_avg.shape, opt.shape)
        np.testing.assert_allclose(x, opt, rtol=1e-6)

    def test_solve_gd_on_least_squares_vectorized(self):
        rng = np.random.RandomState(seed=0)
        prng_seed = 0
        n, d = 100, 10
        batch_size = n  # to use GD instead of SGD
        num_samples = 100

        X = rng.randn(n, d)
        y = X.dot(rng.randn(d)) + rng.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size)
        opt = jnp.tile(jnp.expand_dims(obj.solve(), axis=0), (num_samples, 1))

        prng_key = random.PRNGKey(prng_seed)
        prng_key, subkey = random.split(prng_key)
        init_states = random.normal(subkey, opt.shape, dtype=opt.dtype)

        def _lr_schedule(_):
            return 0.1

        (xs, vs), x_avgs = solve_sgd(
            objective=obj,
            prng_key=prng_key,
            steps=1000,
            init_states=init_states,
            learning_rate_schedule=_lr_schedule,
            momentum=0.9,
            noise_scale=0.0,
        )

        self.assertSequenceEqual(xs.shape, opt.shape)
        self.assertSequenceEqual(vs.shape, opt.shape)
        self.assertSequenceEqual(x_avgs.shape, opt.shape)
        np.testing.assert_allclose(xs, opt, rtol=1e-6)

    def test_solve_sgd_on_least_squares(self):
        rng = np.random.RandomState(seed=0)
        prng_seed = 0
        n, d = 100, 10
        batch_size = 10

        X = rng.randn(n, d)
        y = X.dot(rng.randn(d)) + rng.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size)
        opt = obj.solve()

        def _lr_schedule(i):
            return 1.0 / jnp.sqrt(1.0 + i * 0.01)

        prng_key = random.PRNGKey(prng_seed)
        prng_key, subkey = random.split(prng_key)
        x = random.normal(subkey, opt.shape, dtype=opt.dtype)

        (x, v), x_avg = solve_sgd(
            objective=obj,
            prng_key=prng_key,
            steps=1000,
            init_states=x,
            learning_rate_schedule=_lr_schedule,
        )

        self.assertSequenceEqual(x.shape, opt.shape)
        self.assertSequenceEqual(v.shape, opt.shape)
        self.assertSequenceEqual(x_avg.shape, opt.shape)
        np.testing.assert_allclose(x_avg, opt, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
