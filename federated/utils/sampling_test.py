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
"""Tests for posterior sampling utility functions."""

import jax.numpy as jnp
import numpy as np
from absl import flags
from absl.testing import absltest
from jax import random, vmap

from ..objectives.quadratic import (
    LeastSquares,
    Quadratic,
    create_global_least_squares,
    create_global_quadratic,
    create_random_least_squares,
    create_random_quadratics,
)
from .sampling import ExactQuadraticSampler

FLAGS = flags.FLAGS


class ExactQuadraticSamplerTests(absltest.TestCase):
    def test_sample(self):
        np.random.seed(0)

        obj = Quadratic(
            A=jnp.asarray([[2.0, 1.0], [1.0, 5.0]]), b=jnp.asarray([1.0, 2.0])
        )
        obj_opt_state = obj.solve()
        posterior_cov = jnp.linalg.pinv(obj.A)

        num_samples = 10000
        prng_key = random.PRNGKey(0)
        sampler = ExactQuadraticSampler()
        samples = sampler.sample(obj, num_samples, prng_key)

        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples, rowvar=False)

        np.testing.assert_allclose(sample_mean, obj_opt_state, rtol=1e-2)
        np.testing.assert_allclose(sample_cov, posterior_cov, rtol=1e-1)
