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
        init_momentum = jnp.zeros_like(opt)
        opt_sgd, _, opt_sgd_avg, prng_key = solve_sgd(
            obj,
            prng_key=prng_key,
            steps=1000,
            init_state=init_state,
            init_momentum=init_momentum,
            learning_rate=0.1,
            momentum=0.9,
        )

        np.testing.assert_allclose(opt_sgd, opt, rtol=1e-6)

    def test_solve_sgd_on_least_squares(self):
        rng = np.random.RandomState(seed=0)
        prng_seed = 0
        n, d = 100, 10
        batch_size = 10

        X = rng.randn(n, d)
        y = X.dot(rng.randn(d)) + rng.randn(n)
        obj = LeastSquares(X=X, y=y, batch_size=batch_size)
        opt = obj.solve()

        prng_key = random.PRNGKey(prng_seed)
        prng_key, subkey = random.split(prng_key)
        x = random.normal(subkey, opt.shape, dtype=opt.dtype)
        v = jnp.zeros_like(opt)
        lr = lr_init = 0.0002
        momentum = 0.9
        for i in range(10):
            x, v, x_avg, prng_key = solve_sgd(
                obj,
                prng_key=prng_key,
                steps=5000,
                init_state=x,
                init_momentum=v,
                learning_rate=lr,
                momentum=momentum,
            )
            lr = lr_init / (1.0 + i * 5.0)

        np.testing.assert_allclose(x, opt, rtol=1e-2)
        np.testing.assert_allclose(x_avg, opt, rtol=1e-2)


if __name__ == "__main__":
    absltest.main()
