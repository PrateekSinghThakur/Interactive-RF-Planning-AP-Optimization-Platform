"""Access point placement algorithm.

The module owns candidate filtering, greedy selection, and local refinement. It does
not import the propagation engine; callers inject a scoring function with the public
shape scorer(model, access_points, preview) -> {coverage_dbm: [...]}.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

__all__ = ["recommend_access_points"]

CoverageScorer = Callable[[dict[str, Any], list[dict[str, Any]], bool], dict[str, Any]]


def _occupancy_weights(model: dict[str, Any]) -> dict[str, float]:
    mapping = {"low": 0.7, "medium": 1.0, "high": 1.35}
    return {room["id"]: mapping.get(room.get("occupancy_level", "medium"), 1.0) for room in model.get("rooms", [])}


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _active_region_polygon(model: dict[str, Any]) -> list[list[float]] | None:
    for region in model.get("analysis_regions", []) or []:
        if region.get("active"):
            return region.get("boundary_polygon_m")
    return None


def _cell_in_active_region(model: dict[str, Any], idx: int) -> bool:
    polygon = _active_region_polygon(model)
    if not polygon:
        return True
    x, y = model["grid"]["cells"][idx]["center_m"]
    return _point_in_polygon(float(x), float(y), polygon)


def _candidate_cells(model: dict[str, Any], stride: int, max_candidates: int = 240) -> list[int]:
    cells = model["grid"]["cells"]
    candidates = [i for i, c in enumerate(cells) if c.get("placeable") and c.get("type") != "outside" and _cell_in_active_region(model, i)]
    if not candidates:
        return []
    effective_stride = max(1, stride, int(len(candidates) / max(1, max_candidates)))
    sampled = candidates[::effective_stride]
    return sampled[:max_candidates]


def _score_coverage(model: dict[str, Any], coverage: list[float], threshold_dbm: float, already: set[int]) -> float:
    weights = _occupancy_weights(model)
    score = 0.0
    for idx, value in enumerate(coverage):
        if idx in already or value < threshold_dbm:
            continue
        room_id = model["grid"]["cells"][idx].get("room_id")
        score += weights.get(room_id, 1.0)
    return score


def _ap_at(model: dict[str, Any], idx: int, ap_id: str, tx_power_dbm: float, freq_ghz: float) -> dict[str, Any]:
    return {
        "id": ap_id,
        "position_m": model["grid"]["cells"][idx]["center_m"],
        "tx_power_dbm": tx_power_dbm,
        "freq_ghz": freq_ghz,
        "source": "algorithm",
    }


def recommend_access_points(
    model: dict[str, Any],
    scorer: CoverageScorer,
    coverage_target: float = 0.9,
    max_ap_count: int = 4,
    min_signal_dbm: float = -67,
    tx_power_dbm: float = 18,
    freq_ghz: float = 5.0,
    candidate_stride: int = 3,
) -> dict[str, Any]:
    if max_ap_count <= 0:
        raise ValueError("max_ap_count must be positive")

    updated = deepcopy(model)
    candidates = _candidate_cells(model, candidate_stride)
    target_cells = [i for i, c in enumerate(model["grid"]["cells"]) if c.get("placeable") and _cell_in_active_region(model, i)]
    target_count = max(1, len(target_cells))
    covered: set[int] = set()
    placed: list[dict[str, Any]] = []

    for ap_num in range(1, max_ap_count + 1):
        best_idx = None
        best_score = -1.0
        for idx in candidates:
            ap = _ap_at(model, idx, f"ap_{ap_num}", tx_power_dbm, freq_ghz)
            coverage = scorer(model, placed + [ap], True)["coverage_dbm"]
            score = _score_coverage(model, coverage, min_signal_dbm, covered)
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx is None or best_score <= 0:
            break
        new_ap = _ap_at(model, best_idx, f"ap_{ap_num}", tx_power_dbm, freq_ghz)
        placed.append(new_ap)
        coverage = scorer(model, placed, True)["coverage_dbm"]
        covered = {i for i in target_cells if coverage[i] >= min_signal_dbm}
        if len(covered) / target_count >= coverage_target:
            break

    # Local refinement around selected cells using neighbor offsets.
    cols = model["grid"]["cols"]
    candidate_set = set(_candidate_cells(model, 1))
    for ap_i, ap in enumerate(list(placed)):
        cell_index = min(range(len(model["grid"]["cells"])), key=lambda idx: sum((model["grid"]["cells"][idx]["center_m"][j] - ap["position_m"][j]) ** 2 for j in (0, 1)))
        neighbor_indices = [cell_index + d for d in (-cols - 1, -cols, -cols + 1, -1, 1, cols - 1, cols, cols + 1)]
        best_ap = ap
        best_score = _score_coverage(model, scorer(model, placed, True)["coverage_dbm"], min_signal_dbm, set())
        for ni in neighbor_indices:
            if ni not in candidate_set:
                continue
            trial = deepcopy(placed)
            trial[ap_i] = _ap_at(model, ni, ap["id"], tx_power_dbm, freq_ghz)
            score = _score_coverage(model, scorer(model, trial, True)["coverage_dbm"], min_signal_dbm, set())
            if score > best_score:
                best_score = score
                best_ap = trial[ap_i]
        placed[ap_i] = best_ap

    updated["access_points"] = placed
    return updated
