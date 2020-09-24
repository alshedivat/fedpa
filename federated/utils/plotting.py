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
"""Plotting utils."""

import itertools
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes

from ..objectives.base import Objective


def plot_objective_contours(
    ax: Axes,
    objectives: List[Objective],
    global_objective: Optional[Objective] = None,
    contour_alpha: float = 0.7,
    contour_linewidth: float = 0.8,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    mesh_margins: Tuple[float, float] = (10.0, 10.0),
    mesh_steps: Tuple[float, float] = (0.1, 0.1),
    num_levels: int = 10,
    levels_log_base: float = 2.0,
    max_level_coeff: float = 0.25,
):
    for o in objectives:
        assert o.dim == 2, "Can plot contours only for 2D objectives!"

    # Find optima of each objective.
    optima = [o.solve() for o in objectives]
    if global_objective is not None:
        optima.append(global_objective.solve())

    # Determine coordinate limits, if not provided.
    if xlim is None:
        optima_x = [o[0] for o in optima]
        xlim = (
            min(optima_x) - mesh_margins[0],
            max(optima_x) + mesh_margins[0],
        )

    if ylim is None:
        optima_y = [o[1] for o in optima]
        ylim = (
            min(optima_y) - mesh_margins[1],
            max(optima_y) + mesh_margins[1],
        )

    # Create coordinate meshes.
    xs = np.arange(*xlim, mesh_steps[0], dtype=np.float32)
    ys = np.arange(*ylim, mesh_steps[1], dtype=np.float32)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    x_mesh_flat, y_mesh_flat = x_mesh.flatten(), y_mesh.flatten()
    params = np.stack([x_mesh_flat, y_mesh_flat]).T
    z_meshes = [np.asarray(o(params)).reshape(x_mesh.shape) for o in objectives]

    # Determine contour levels.
    corners = np.asarray(itertools.product(xlim, ylim))
    level_min = np.min([obj(opt) for obj, opt in zip(objectives, optima)])
    level_max = np.max([obj(corners) for obj in objectives]) * max_level_coeff
    levels = np.logspace(
        np.log(level_min), np.log(level_max), num_levels, base=levels_log_base
    )

    # Plot and return contours.
    contour_kwargs = {"alpha": contour_alpha, "linewidths": contour_linewidth}

    return [
        ax.contour(x_mesh, y_mesh, z_mesh, levels, **contour_kwargs)
        for z_mesh in z_meshes
    ]
