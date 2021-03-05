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
"""Tests for posterior sampling functions."""

import jax.numpy as jnp
import numpy as np
import numpyro.infer as infer
from absl import flags
from absl.testing import absltest
from jax import random

from ..objectives.quadratic import Quadratic, create_random_least_squares
from .sampling import EQS, HMC, IASG

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
        sampler = EQS()
        samples = sampler.sample(
            objective=obj, prng_key=prng_key, num_samples=num_samples
        )
        self.assertEqual(samples.shape[0], num_samples)

        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples, rowvar=False)

        np.testing.assert_allclose(sample_mean, obj_opt_state, rtol=1e-2)
        np.testing.assert_allclose(sample_cov, posterior_cov, rtol=1e-1)


class IASGSampler(absltest.TestCase):
    def test_sample(self):
        np.random.seed(0)
        n, d = 1000, 5
        batch_size = 10
        num_samples = 200
        parallel_chains = 1

        obj = create_random_least_squares(
            num_objectives=1,
            batch_size=batch_size,
            n_features=(d - 1),
            n_samples=(n,),
            lam=1e-3,
        )[0]
        opt = obj.solve()
        q_obj = Quadratic.from_least_squares(obj)
        posterior_cov = jnp.linalg.pinv(q_obj.A)
        posterior_cov /= jnp.trace(posterior_cov)

        # Approximate sampling from the posterior.
        prng_key = random.PRNGKey(0)
        sampler = IASG(
            avg_steps=100,
            burnin_steps=100,
            learning_rate=1.0,
            discard_steps=100,
        )
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, shape=(d,))
        samples = sampler.sample(
            objective=obj,
            prng_key=prng_key,
            init_state=init_state,
            num_samples=num_samples,
            parallel_chains=parallel_chains,
        )
        self.assertEqual(samples.shape[0], num_samples)

        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples, rowvar=False)
        sample_cov /= jnp.trace(sample_cov)

        sample_cov_fro_err = jnp.linalg.norm(sample_cov - posterior_cov, "fro")
        np.testing.assert_allclose(sample_mean, opt, rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(
            sample_cov_fro_err, 0.0, rtol=1e-1, atol=1e-1
        )


class HMCSampler(absltest.TestCase):
    def test_sample(self):
        np.random.seed(0)
        n, d = 1000, 5
        batch_size = 10
        num_samples = 1000
        parallel_chains = 1

        obj = create_random_least_squares(
            num_objectives=1,
            batch_size=batch_size,
            n_features=(d - 1),
            n_samples=(n,),
            lam=1e-3,
        )[0]
        opt = obj.solve()
        q_obj = Quadratic.from_least_squares(obj)
        posterior_cov = jnp.linalg.pinv(q_obj.A)
        posterior_cov /= jnp.trace(posterior_cov)

        # Approximate sampling from the posterior.
        prng_key = random.PRNGKey(0)
        sampler = HMC(kernel=infer.NUTS, num_warmup=1000)
        prng_key, subkey = random.split(prng_key)
        init_state = random.normal(subkey, shape=(d,))
        samples = sampler.sample(
            objective=obj,
            prng_key=prng_key,
            init_state=init_state,
            num_samples=num_samples,
            parallel_chains=parallel_chains,
        )
        self.assertEqual(samples.shape[0], num_samples)

        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples, rowvar=False)
        sample_cov /= jnp.trace(sample_cov)

        sample_cov_fro_err = jnp.linalg.norm(sample_cov - posterior_cov, "fro")
        np.testing.assert_allclose(sample_mean, opt, rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(
            sample_cov_fro_err, 0.0, rtol=1e-1, atol=1e-1
        )


if __name__ == "__main__":
    absltest.main()
