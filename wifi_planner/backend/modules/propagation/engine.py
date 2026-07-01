"""Wi-Fi signal propagation over a generic raster grid.

Owns grid + access_points -> coverage array. It does not modify structural JSON.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # Numba is optional at import-time but used when available.
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

__all__ = ["compute_coverage"]


def _fspl_db(distance_m: np.ndarray, freq_ghz: float) -> np.ndarray:
    # FSPL: 32.44 + 20log10(d_km) + 20log10(f_MHz)
    d = np.maximum(distance_m, 1.0)
    return 32.44 + 20 * np.log10(d / 1000.0) + 20 * np.log10(freq_ghz * 1000.0)


if njit:
    @njit(cache=True)
    def _ray_attenuation(att_grid: np.ndarray, rows: int, cols: int, ar: int, ac: int, tr: int, tc: int) -> float:
        dr = abs(tr - ar)
        dc = abs(tc - ac)
        sr = 1 if ar < tr else -1
        sc = 1 if ac < tc else -1
        r = ar
        c = ac
        total = 0.0
        if dc > dr:
            err = dc / 2
            while c != tc:
                if 0 <= r < rows and 0 <= c < cols:
                    total += att_grid[r, c]
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr / 2
            while r != tr:
                if 0 <= r < rows and 0 <= c < cols:
                    total += att_grid[r, c]
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        if 0 <= tr < rows and 0 <= tc < cols:
            total += att_grid[tr, tc]
        return total

    @njit(cache=True)
    def _coverage_kernel(att_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, rows: int, cols: int, res: float, apx: float, apy: float, tx: float, freq: float) -> np.ndarray:
        out = np.empty(rows * cols, dtype=np.float64)
        ac = int(apx / res)
        ar = int(apy / res)
        for idx in range(rows * cols):
            tr = idx // cols
            tc = idx - tr * cols
            dx = xs[idx] - apx
            dy = ys[idx] - apy
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1.0:
                d = 1.0
            fspl = 32.44 + 20.0 * math.log10(d / 1000.0) + 20.0 * math.log10(freq * 1000.0)
            wall_loss = _ray_attenuation(att_grid, rows, cols, ar, ac, tr, tc)
            out[idx] = tx - fspl - wall_loss
        return out
else:  # pragma: no cover
    _coverage_kernel = None


def _grid_arrays(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = grid["rows"]
    cols = grid["cols"]
    cells = grid["cells"]
    xs = np.array([cell["center_m"][0] for cell in cells], dtype=np.float64)
    ys = np.array([cell["center_m"][1] for cell in cells], dtype=np.float64)
    attenuation = np.array([cell["attenuation_db"] for cell in cells], dtype=np.float64).reshape(rows, cols)
    return xs, ys, attenuation


def _coverage_numpy(grid: dict[str, Any], ap: dict[str, Any]) -> np.ndarray:
    """Very fast preview coverage.

    This intentionally avoids ray-casting. It is used during dragging and greedy
    candidate search, where responsiveness matters more than final RF precision.
    Full mode still uses the Numba Bresenham ray kernel when available.
    """
    xs, ys, _att = _grid_arrays(grid)
    apx, apy = ap["position_m"]
    tx = float(ap["tx_power_dbm"])
    freq = float(ap["freq_ghz"])
    distances = np.sqrt((xs - apx) ** 2 + (ys - apy) ** 2)
    values = tx - _fspl_db(distances, freq)

    # Cheap attenuation hint: penalize the target cell's own material plus a small
    # distance-related indoor clutter term. This keeps AP dragging and placement
    # search fast, then final coverage can be recalculated in full mode.
    cell_att = np.array([cell["attenuation_db"] for cell in grid["cells"]], dtype=np.float64)
    values -= cell_att * 1.15
    values -= np.minimum(distances * 0.18, 10.0)
    return values


def compute_coverage(model: dict[str, Any], access_points: list[dict[str, Any]] | None = None, preview: bool = False) -> dict[str, Any]:
    grid = model["grid"]
    aps = access_points if access_points is not None else model.get("access_points", [])
    total_cells = grid["rows"] * grid["cols"]
    if not aps:
        return {"coverage_dbm": [-120.0] * total_cells, "rows": grid["rows"], "cols": grid["cols"], "resolution_m": grid["resolution_m"], "mode": "preview" if preview else "full"}

    rows, cols, res = grid["rows"], grid["cols"], float(grid["resolution_m"])
    xs, ys, att = _grid_arrays(grid)
    strongest = np.full(total_cells, -120.0, dtype=np.float64)

    for ap in aps:
        if preview or _coverage_kernel is None:
            values = _coverage_numpy(grid, ap)
        else:
            values = _coverage_kernel(att, xs, ys, rows, cols, res, float(ap["position_m"][0]), float(ap["position_m"][1]), float(ap["tx_power_dbm"]), float(ap["freq_ghz"]))
        strongest = np.maximum(strongest, values)

    # Outside cells are not meaningful coverage targets; keep them transparent-ish.
    for i, cell in enumerate(grid["cells"]):
        if cell["type"] == "outside":
            strongest[i] = -120.0
    return {
        "coverage_dbm": [round(float(v), 2) for v in strongest],
        "rows": rows,
        "cols": cols,
        "resolution_m": res,
        "mode": "preview" if preview else "full",
    }
