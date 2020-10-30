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
"""Federated learning algorithms."""

from typing import Any, Callable, Dict, List, Tuple

import attr
import jax.numpy as jnp
from jax import random

from ..inference.local import (
    compute_fed_avg_delta,
    compute_mb_sgd_delta,
    compute_post_avg_delta,
    compute_post_avg_delta_dp,
)
from ..inference.moments import ShrinkageMomentEstimator
from ..inference.sampling import (
    ExactQuadraticSampler,
    IterateAveragedStochasticGradientSampler,
)
from ..objectives.base import StochasticObjective
from ..utils.timing import Timer
from .averaging import compute_weighted_average


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
    """Represents the server state."""

    r: int = attr.ib()
    x: jnp.ndarray = attr.ib()
    v: jnp.ndarray = attr.ib()


# Type aliases.
ClientUpdateFn = Callable[
    [
        StochasticObjective,  # client objective function
        jnp.ndarray,  # initial state
    ],
    jnp.ndarray,  # client delta
]
ServerUpdateFn = Callable[
    [
        List[jnp.ndarray],  # client deltas
        jnp.ndarray,  # client weights
        ServerState,  # previous server state
    ],
    ServerState,  # updated server state
]
SampleClientsFn = Callable[
    [
        jnp.ndarray,  # prng key
        int,  # the total number of clients
        int,  # the number of clients to sample
    ],
    jnp.ndarray,  # sampled client ids
]
RoundInfo = Dict[str, Any]
FederatedLearningFn = Callable[
    [
        List[StochasticObjective],  # a list of client objectives
        jnp.ndarray,  # initial state
        jnp.ndarray,  # prng key
        int,  # number of round
        int,  # number of clients per round
    ],
    Tuple[List[ServerState], List[RoundInfo]],  # trajectory
]


def fed_opt(
    client_objectives: List[StochasticObjective],
    client_update_fn: ClientUpdateFn,
    server_update_fn: ServerUpdateFn,
    sample_clients_fn: SampleClientsFn,
    prng_key: jnp.ndarray,
    init_state: jnp.ndarray,
    num_rounds: int,
    num_clients_per_round: int,
) -> Tuple[List[ServerState], List[RoundInfo]]:
    """Runs generalized federated averaging for the specified number of rounds.

    At each round, the algorithm does the following:
        1.  Samples a batch of clients using `sample_clients_fn`.
        2.  Runs `client_update_fn` on each sampled client objective that
            returns a `client_delta`.
        3.  Aggregates `client_deltas` using `server_update_fn`.

    Args:
        client_update_fn: A function for computing local client updates.
        server_update_fn: A function for computing server updates.
        sample_clients_fn: A function for sampling indices of the clients.
        client_objectives: A list of client objective functions.
        prng_key: A key for random number generation.
        init_state: The initial server state.
        num_rounds: The number of training rounds to run.
        num_clients_per_round: The number of clients used at each round.

    Returns:
        A list of tuples `(round: int, state: ServerState)` that represents the
        trajectory of the server state over the course of training.
    """
    num_clients = len(client_objectives)
    server_state = ServerState(r=0, x=init_state, v=jnp.zeros_like(init_state))

    trajectory = [server_state]
    info = [None]
    for _ in range(num_rounds):
        round_info = {}

        # Select clients.
        prng_key, subkey = random.split(prng_key)
        with Timer("select_clients_time") as t:
            client_ids = sample_clients_fn(
                subkey, num_clients, num_clients_per_round
            )
            client_objectives_round = [client_objectives[i] for i in client_ids]
            client_weights_round = jnp.asarray(
                [float(o.num_points) for o in client_objectives_round]
            )
        round_info[t.description] = t.elapsed

        # Compute client updates.
        client_deltas_round = []
        # TODO: parallelize this loop.
        with Timer("client_updates_time") as t:
            for client_objective in client_objectives_round:
                prng_key, subkey = random.split(prng_key)
                client_delta = client_update_fn(
                    client_objective, server_state.x, subkey
                )
                client_deltas_round.append(client_delta)
        round_info[t.description] = t.elapsed

        # Update server state.
        with Timer("server_update_time") as t:
            server_state = server_update_fn(
                client_deltas_round, client_weights_round, server_state
            )
        round_info[t.description] = t.elapsed
        trajectory.append(server_state)
        info.append(round_info)

    return trajectory, info


def compute_server_update(
    client_deltas: List[jnp.ndarray],
    client_weights: jnp.ndarray,
    init_state: ServerState,
    *,
    learning_rate_schedule: Callable[[int], float],
    momentum: float = 0.0,
) -> ServerState:
    """Computes the server update by averaging deltas and taking a step.

    Args:
        client_deltas: A list of client deltas.
        client_weights: An array of client weights.
        init_state: The initial server state.
        learning_rate_schedule: The server learning rate schedule.
        momentum: The server momentum coefficient.

    Returns:
        An updated `ServerState`.
    """
    # Compute the weighted average of client deltas.
    client_deltas_avg = compute_weighted_average(client_deltas, client_weights)
    # Take an SGD step.
    v = momentum * init_state.v + client_deltas_avg
    x = init_state.x - learning_rate_schedule(init_state.r) * v
    return ServerState(r=(init_state.r + 1), x=x, v=v)


def sample_clients_uniformly(
    prng_key: jnp.ndarray,
    num_clients_total: int,
    num_clients_to_sample: int,
    replace: bool = False,
) -> jnp.ndarray:
    """Samples clients uniformly at random.

    Args:
        prng_key: A key for random number generation.
        num_clients_total: The total number of clients.
        num_clients_to_sample: The number of clients to sample.
        replace: Whether to sample clients with replacement.

    Returns:
        An array of client indices.
    """
    return random.choice(
        prng_key, num_clients_total, (num_clients_to_sample,), replace=replace
    )


# Utility functions that create specific FL algorithms.


def create_mb_sgd(
    *,
    client_grads_per_round: int,
    client_learning_rate_schedule: Callable[[int], float],
    server_learning_rate_schedule: Callable[[int], float],
    client_momentum: float = 0.0,
    server_momentum: float = 0.0,
) -> FederatedLearningFn:
    """Creates a MB-SGD federated learner.

    MB-SGD is essentially a distributed SGD. At each round, the selected clients
    compute `client_steps_per_round` stochastic gradients at the provided state,
    then average them and send to the server. The server further averages client
    gradients and makes and SGD step.

    Args:
        client_grads_per_round: The number of gradients computed by each client.
        client_learning_rate_schedule: The schedule for client learning rate.
        server_learning_rate_schedule: The schedule for server learning rate.
        client_momentum: The momentum used by client optimizers.
        server_momentum: The momentum used by the server optimizer.

    Returns:
        A federated learning function.
    """

    def _client_update_fn(
        objective: StochasticObjective,
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_mb_sgd_delta(
            objective=objective,
            init_state=init_state,
            prng_key=prng_key,
            num_grads=client_grads_per_round,
            learning_rate_schedule=client_learning_rate_schedule,
            momentum=client_momentum,
        )

    def _server_update_fn(
        client_deltas: List[jnp.ndarray],
        client_weights: jnp.ndarray,
        init_state: ServerState,
    ) -> ServerState:
        return compute_server_update(
            client_deltas=client_deltas,
            client_weights=client_weights,
            init_state=init_state,
            learning_rate_schedule=server_learning_rate_schedule,
            momentum=server_momentum,
        )

    def _fed_learn(
        client_objectives: List[StochasticObjective],
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
        num_rounds: int,
        num_clients_per_round: int,
    ) -> Tuple[List[ServerState], List[RoundInfo]]:
        return fed_opt(
            client_objectives=client_objectives,
            client_update_fn=_client_update_fn,
            server_update_fn=_server_update_fn,
            sample_clients_fn=sample_clients_uniformly,
            prng_key=prng_key,
            init_state=init_state,
            num_rounds=num_rounds,
            num_clients_per_round=num_clients_per_round,
        )

    return _fed_learn


def create_fed_avg(
    *,
    client_steps_per_round: int,
    client_learning_rate_schedule: Callable[[int], float],
    server_learning_rate_schedule: Callable[[int], float],
    client_momentum: float = 0.0,
    server_momentum: float = 0.0,
) -> FederatedLearningFn:
    """Creates a generalized FedAvg.

    Args:
        client_steps_per_round: The number of local SGD steps done by clients.
        client_learning_rate_schedule: The schedule for client learning rate.
        server_learning_rate_schedule: The schedule for server learning rate.
        client_momentum: The momentum used by client optimizers.
        server_momentum: The momentum used by the server optimizer.

    Returns:
        A federated learning function.
    """

    def _client_update_fn(
        objective: StochasticObjective,
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_fed_avg_delta(
            objective=objective,
            init_state=init_state,
            prng_key=prng_key,
            num_steps=client_steps_per_round,
            learning_rate_schedule=client_learning_rate_schedule,
            momentum=client_momentum,
        )

    def _server_update_fn(
        client_deltas: List[jnp.ndarray],
        client_weights: jnp.ndarray,
        init_state: ServerState,
    ) -> ServerState:
        return compute_server_update(
            client_deltas=client_deltas,
            client_weights=client_weights,
            init_state=init_state,
            learning_rate_schedule=server_learning_rate_schedule,
            momentum=server_momentum,
        )

    def _fed_learn(
        client_objectives: List[StochasticObjective],
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
        num_rounds: int,
        num_clients_per_round: int,
    ) -> Tuple[List[ServerState], List[RoundInfo]]:
        return fed_opt(
            client_objectives=client_objectives,
            client_update_fn=_client_update_fn,
            server_update_fn=_server_update_fn,
            sample_clients_fn=sample_clients_uniformly,
            prng_key=prng_key,
            init_state=init_state,
            num_rounds=num_rounds,
            num_clients_per_round=num_clients_per_round,
        )

    return _fed_learn


def create_post_avg_exact(
    *,
    client_num_samples_per_round: int,
    server_learning_rate_schedule: Callable[[int], float],
    server_momentum: float = 0.0,
    shrinkage_rho: float = 0.0,
    use_dp: bool = False,
) -> FederatedLearningFn:
    """Creates a FedPostAvg with exact local posterior sampling.

    Args:
        client_num_samples_per_round: The number of posterior samples used by
            clients at each round to compute model deltas.
        server_learning_rate_schedule: The schedule for server learning rate.
        server_momentum: The momentum used by the server optimizer.
        shrinkage_rho: The shrinkage parameter for covariance estimation.
        use_dp: Whether to use dynamic programming for computing client deltas.
    Returns:
        A federated learning function.
    """
    posterior_sampler = ExactQuadraticSampler()
    moment_estimator = ShrinkageMomentEstimator(rho=shrinkage_rho)

    def _client_update_fn(
        objective: StochasticObjective,
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_post_avg_delta(
            objective=objective,
            init_state=init_state,
            prng_key=prng_key,
            num_samples=client_num_samples_per_round,
            sampler=posterior_sampler,
            moment_estimator=moment_estimator,
        )

    def _server_update_fn(
        client_deltas: List[jnp.ndarray],
        client_weights: jnp.ndarray,
        init_state: ServerState,
    ) -> ServerState:
        return compute_server_update(
            client_deltas=client_deltas,
            client_weights=client_weights,
            init_state=init_state,
            learning_rate_schedule=server_learning_rate_schedule,
            momentum=server_momentum,
        )

    def _fed_learn(
        client_objectives: List[StochasticObjective],
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
        num_rounds: int,
        num_clients_per_round: int,
    ) -> Tuple[List[ServerState], List[RoundInfo]]:
        return fed_opt(
            client_objectives=client_objectives,
            client_update_fn=_client_update_fn,
            server_update_fn=_server_update_fn,
            sample_clients_fn=sample_clients_uniformly,
            prng_key=prng_key,
            init_state=init_state,
            num_rounds=num_rounds,
            num_clients_per_round=num_clients_per_round,
        )

    return _fed_learn


def create_post_avg_iasg(
    *,
    client_samples_per_round: int,
    client_iasg_avg_steps: int,
    client_iasg_burnin_steps: int,
    client_iasg_learning_rate: float,
    client_iasg_discard_steps: int = 0,
    client_iasg_momentum: float = 0.0,
    server_learning_rate_schedule: Callable[[int], float],
    server_momentum: float = 0.0,
    shrinkage_rho: float = 0.0,
    use_dp: bool = False,
) -> FederatedLearningFn:
    """Creates a FedPostAvg with IASG-based local posterior sampling.

    Args:
        client_samples_per_round: The number of local SGD steps done by clients.
        client_iasg_avg_steps: The number of averaged SGD steps per IASG sample.
        client_iasg_burnin_steps: The number of IASG burn-in SGD steps.
        client_iasg_learning_rate: The learning rate used by the local SGD.
        client_iasg_discard_steps: The number of discarded SGD per IASG sample.
        client_iasg_momentum: The momentum used by the local SGD.
        server_learning_rate_schedule: The schedule for server learning rate.
        server_momentum: The momentum used by the server optimizer.
        shrinkage_rho: The shrinkage parameter for covariance estimation.
        use_dp: Whether to use dynamic programming for computing client deltas.

    Returns:
        A federated learning function.
    """
    posterior_sampler = IterateAveragedStochasticGradientSampler(
        avg_steps=client_iasg_avg_steps,
        burnin_steps=client_iasg_burnin_steps,
        learning_rate=client_iasg_learning_rate,
        discard_steps=client_iasg_discard_steps,
        momentum=client_iasg_momentum,
    )
    moment_estimator = ShrinkageMomentEstimator(rho=shrinkage_rho)

    def _client_update_fn(
        objective: StochasticObjective,
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_post_avg_delta(
            objective=objective,
            init_state=init_state,
            prng_key=prng_key,
            num_samples=client_samples_per_round,
            sampler=posterior_sampler,
            moment_estimator=moment_estimator,
        )

    def _server_update_fn(
        client_deltas: List[jnp.ndarray],
        client_weights: jnp.ndarray,
        init_state: ServerState,
    ) -> ServerState:
        return compute_server_update(
            client_deltas=client_deltas,
            client_weights=client_weights,
            init_state=init_state,
            learning_rate_schedule=server_learning_rate_schedule,
            momentum=server_momentum,
        )

    def _fed_learn(
        client_objectives: List[StochasticObjective],
        init_state: jnp.ndarray,
        prng_key: jnp.ndarray,
        num_rounds: int,
        num_clients_per_round: int,
    ) -> Tuple[List[ServerState], List[RoundInfo]]:
        return fed_opt(
            client_objectives=client_objectives,
            client_update_fn=_client_update_fn,
            server_update_fn=_server_update_fn,
            sample_clients_fn=sample_clients_uniformly,
            prng_key=prng_key,
            init_state=init_state,
            num_rounds=num_rounds,
            num_clients_per_round=num_clients_per_round,
        )

    return _fed_learn
