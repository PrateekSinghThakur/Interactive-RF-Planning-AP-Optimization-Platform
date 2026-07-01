"""Particle Swarm Optimization for AP placement.

Phase 6 scope:
- Generic PSO implementation.
- AP placement wrapper using the FMM solver as a fitness function.

PSO math summary:
Each particle has position x and velocity v. It remembers its personal best p,
and the swarm remembers global best g. Updates are:

    v <- w*v + c1*r1*(p - x) + c2*r2*(g - x)
    x <- x + v

where w is inertia, c1 cognitive attraction, c2 social attraction, and r1/r2 are
uniform random values. PSO is a heuristic: it often works well on non-convex
search spaces, but it has no guarantee of global optimality and can prematurely
converge if parameters are poor.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from backend.modules.rf.fmm_solver import RadioParams, solve_fmm_rssi
from backend.modules.rf.speed_map import SpeedMapResult

FitnessFn = Callable[[np.ndarray], float]
ConstraintFn = Callable[[np.ndarray], bool]


@dataclass(frozen=True)
class PSOParams:
    particles: int = 24
    iterations: int = 30
    inertia: float = 0.72
    cognitive: float = 1.45
    social: float = 1.45
    seed: int = 7
    plateau_iterations: int = 8
    n_jobs: int = 1


@dataclass(frozen=True)
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: list[float]
    iterations_run: int


def pso_optimize(
    fitness_fn: FitnessFn,
    bounds: list[tuple[float, float]],
    params: PSOParams = PSOParams(),
    constraint_fn: ConstraintFn | None = None,
    penalty_value: float = -1e6,
) -> PSOResult:
    rng = np.random.default_rng(params.seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo

    positions = rng.uniform(lo, hi, size=(params.particles, dim))
    velocities = rng.uniform(-span, span, size=(params.particles, dim)) * 0.08
    personal_best = positions.copy()

    def eval_one(x: np.ndarray) -> float:
        if constraint_fn is not None and not constraint_fn(x):
            return penalty_value
        return float(fitness_fn(x))

    def eval_many(pop: np.ndarray) -> np.ndarray:
        if params.n_jobs and params.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=params.n_jobs) as ex:
                return np.array(list(ex.map(eval_one, [p.copy() for p in pop])), dtype=float)
        return np.array([eval_one(p.copy()) for p in pop], dtype=float)

    personal_scores = eval_many(positions)
    best_idx = int(np.argmax(personal_scores))
    global_best = personal_best[best_idx].copy()
    global_score = float(personal_scores[best_idx])
    history = [global_score]
    plateau = 0

    for iteration in range(params.iterations):
        r1 = rng.random(size=(params.particles, dim))
        r2 = rng.random(size=(params.particles, dim))
        velocities = (
            params.inertia * velocities
            + params.cognitive * r1 * (personal_best - positions)
            + params.social * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lo, hi)

        scores = eval_many(positions)
        improved = scores > personal_scores
        personal_best[improved] = positions[improved]
        personal_scores[improved] = scores[improved]

        idx = int(np.argmax(personal_scores))
        if personal_scores[idx] > global_score + 1e-9:
            global_score = float(personal_scores[idx])
            global_best = personal_best[idx].copy()
            plateau = 0
        else:
            plateau += 1
        history.append(global_score)
        if plateau >= params.plateau_iterations:
            return PSOResult(global_best, global_score, history, iteration + 1)

    return PSOResult(global_best, global_score, history, params.iterations)


def _valid_ap_vector(vector: np.ndarray, speed_result: SpeedMapResult) -> bool:
    # Valid AP: inside bbox, not on attenuating segment cell.
    for i in range(0, len(vector), 2):
        x, y = float(vector[i]), float(vector[i + 1])
        if not (speed_result.bbox_m["min_x"] <= x <= speed_result.bbox_m["max_x"]):
            return False
        if not (speed_result.bbox_m["min_y"] <= y <= speed_result.bbox_m["max_y"]):
            return False
        row, col = speed_result.xy_m_to_row_col(x, y)
        if speed_result.attenuation_map_db[row, col] > 0:
            return False
    return True


def pso_place_aps_fmm(
    speed_result: SpeedMapResult,
    n_aps: int = 2,
    target_rssi_dbm: float = -68.0,
    radio: RadioParams | None = None,
    params: PSOParams = PSOParams(particles=12, iterations=12, n_jobs=1),
) -> dict[str, object]:
    """Optimize AP positions with PSO using FMM coverage as fitness.

    Fitness = fraction of non-wall grid cells with RSSI >= target threshold.
    Invalid particles receive a hard penalty via constraint_fn.
    """
    radio = radio or RadioParams()
    free_cells = speed_result.attenuation_map_db <= 0

    bounds: list[tuple[float, float]] = []
    for _ in range(n_aps):
        bounds.extend([
            (speed_result.bbox_m["min_x"], speed_result.bbox_m["max_x"]),
            (speed_result.bbox_m["min_y"], speed_result.bbox_m["max_y"]),
        ])

    def fitness(vector: np.ndarray) -> float:
        best = np.full(speed_result.speed_map.shape, -120.0, dtype=float)
        for i in range(0, len(vector), 2):
            fmm = solve_fmm_rssi(
                speed_result.speed_map,
                (float(vector[i]), float(vector[i + 1])),
                speed_result.resolution_cells_per_m,
                radio=radio,
                smooth_sigma_cells=None,
            )
            best = np.maximum(best, fmm["rssi_dbm"])
        return float(np.mean(best[free_cells] >= target_rssi_dbm))

    result = pso_optimize(
        fitness,
        bounds,
        params=params,
        constraint_fn=lambda v: _valid_ap_vector(v, speed_result),
        penalty_value=-10.0,
    )
    aps = [(float(result.best_position[i]), float(result.best_position[i + 1])) for i in range(0, len(result.best_position), 2)]
    return {"aps": aps, "fitness": result.best_fitness, "history": result.history, "iterations_run": result.iterations_run}
