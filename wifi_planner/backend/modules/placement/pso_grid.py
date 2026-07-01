"""PSO placement over the existing Building Model grid.

This is the UI-facing PSO wrapper for the current raster/grid app. It uses the
existing backend coverage scorer as the fitness function, while enforcing that
AP coordinates stay inside placeable grid cells and the active ROI when present.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable

import numpy as np

from backend.modules.optimization.pso import PSOParams, pso_optimize

CoverageScorer = Callable[[dict[str, Any], list[dict[str, Any]], bool], dict[str, Any]]


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _active_region(model: dict[str, Any]) -> list[list[float]] | None:
    for region in model.get("analysis_regions", []) or []:
        if region.get("active"):
            return region.get("boundary_polygon_m")
    return None


def _cell_allowed(model: dict[str, Any], idx: int) -> bool:
    cell = model["grid"]["cells"][idx]
    if not cell.get("placeable") or cell.get("type") == "outside":
        return False
    polygon = _active_region(model)
    if polygon:
        x, y = cell["center_m"]
        return _point_in_polygon(float(x), float(y), polygon)
    return True


def _nearest_cell_index(model: dict[str, Any], x: float, y: float) -> int | None:
    grid = model["grid"]
    col = int((x - grid["origin_m"][0]) / grid["resolution_m"])
    row = int((y - grid["origin_m"][1]) / grid["resolution_m"])
    if row < 0 or col < 0 or row >= grid["rows"] or col >= grid["cols"]:
        return None
    return row * grid["cols"] + col


def recommend_access_points_pso(
    model: dict[str, Any],
    scorer: CoverageScorer,
    min_signal_dbm: float = -67,
    max_ap_count: int = 3,
    particles: int = 12,
    iterations: int = 12,
) -> dict[str, Any]:
    """Recommend APs using PSO over the model grid.

    Fitness = fraction of allowed cells with RSSI >= min_signal_dbm.
    Invalid APs receive a hard penalty through PSO's constraint function.
    """
    allowed = [i for i in range(len(model["grid"]["cells"])) if _cell_allowed(model, i)]
    if not allowed:
        updated = deepcopy(model)
        updated["access_points"] = []
        return updated

    xs = [model["grid"]["cells"][i]["center_m"][0] for i in allowed]
    ys = [model["grid"]["cells"][i]["center_m"][1] for i in allowed]
    bounds: list[tuple[float, float]] = []
    for _ in range(max_ap_count):
        bounds.extend([(min(xs), max(xs)), (min(ys), max(ys))])

    def vector_to_aps(vec: np.ndarray) -> list[dict[str, Any]]:
        aps = []
        for ap_i in range(max_ap_count):
            x = float(vec[2 * ap_i])
            y = float(vec[2 * ap_i + 1])
            aps.append({"id": f"ap_{ap_i+1}", "position_m": [x, y], "tx_power_dbm": 18, "freq_ghz": 5.0, "source": "algorithm"})
        return aps

    def valid(vec: np.ndarray) -> bool:
        for ap_i in range(max_ap_count):
            x = float(vec[2 * ap_i])
            y = float(vec[2 * ap_i + 1])
            idx = _nearest_cell_index(model, x, y)
            if idx is None or not _cell_allowed(model, idx):
                return False
        return True

    def fitness(vec: np.ndarray) -> float:
        aps = vector_to_aps(vec)
        coverage = scorer(model, aps, True)["coverage_dbm"]
        covered = sum(1 for idx in allowed if coverage[idx] >= min_signal_dbm)
        return covered / max(1, len(allowed))

    result = pso_optimize(
        fitness,
        bounds,
        params=PSOParams(particles=particles, iterations=iterations, seed=11, plateau_iterations=6, n_jobs=1),
        constraint_fn=valid,
        penalty_value=-10.0,
    )

    updated = deepcopy(model)
    updated["access_points"] = vector_to_aps(result.best_position)
    updated.setdefault("metadata", {})["pso_last_fitness"] = result.best_fitness
    return updated
