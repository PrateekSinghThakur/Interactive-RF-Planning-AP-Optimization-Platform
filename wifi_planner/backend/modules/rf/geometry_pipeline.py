"""Unified geometry-first RF pipeline.

All input sources must produce GeometrySegment[]. This module is the shared path
for FastAPI, CLI, and future notebooks:

    GeometrySegment[] -> speed map -> FMM -> optional PSO -> visualization
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.modules.geometry.segments import GeometrySegment, coerce_geometry_segment
from backend.modules.optimization.pso import PSOParams, pso_place_aps_fmm
from backend.modules.rf.fmm_solver import RadioParams, solve_fmm_rssi
from backend.modules.rf.speed_map import rasterize_segments_to_speed_map


def _fig_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _solve_multi_ap(speed_result, aps: list[tuple[float, float]], radio: RadioParams, resolution: float) -> np.ndarray:
    master = np.full(speed_result.speed_map.shape, -120.0, dtype=float)
    for ap in aps:
        fmm = solve_fmm_rssi(speed_result.speed_map, ap, resolution, radio=radio, smooth_sigma_cells=None)
        master = np.maximum(master, fmm["rssi_dbm"])
    try:
        from scipy.ndimage import gaussian_filter
        master = gaussian_filter(master, sigma=2.0)
    except Exception:
        pass
    return np.clip(master, -90, -40)


def _plot_heatmap(speed_result, heatmap: np.ndarray, aps: list[tuple[float, float]], title: str, optimized: bool = False) -> str:
    fig = plt.figure(figsize=(15 if optimized else 13, 10 if optimized else 9), dpi=140 if optimized else 130)
    plt.imshow(speed_result.speed_map == 1.0, cmap="gray", origin="upper", alpha=0.25 if optimized else 0.20)
    overlay = plt.imshow(heatmap, cmap="jet", alpha=0.55 if optimized else 0.60, origin="upper", vmin=-90, vmax=-40)
    for idx, ap in enumerate(aps, 1):
        row, col = speed_result.xy_m_to_row_col(ap[0], ap[1])
        plt.scatter(col, row, color="magenta", edgecolor="black", s=350 if optimized else 300, marker="*", linewidth=2.5 if optimized else 2, zorder=10, label=f"{'Optimized ' if optimized else ''}AP {idx} ({ap[0]:.1f}m, {ap[1]:.1f}m)")
    plt.title(title, fontsize=13, pad=15)
    plt.xlabel("Local Building Width (Meters)", fontsize=10)
    plt.ylabel("Local Building Height (Meters)", fontsize=10)
    res = speed_result.resolution_cells_per_m
    h_pixels, w_pixels = speed_result.speed_map.shape
    x_ticks = np.arange(0, w_pixels, max(1, int(res * 10)))
    y_ticks = np.arange(0, h_pixels, max(1, int(res * 5)))
    plt.xticks(x_ticks, [f"{t/res:.0f}m" for t in x_ticks])
    plt.yticks(y_ticks, [f"{(h_pixels-t)/res:.0f}m" for t in y_ticks])
    plt.legend(loc="upper right", frameon=True, facecolor="white", shadow=True)
    cbar = plt.colorbar(overlay, orientation="horizontal", pad=0.08, shrink=0.6)
    cbar.set_label("Receiver Signal Strength Indicator Scale (dBm)   [ Red = Strong | Blue = Weak ]", fontsize=11)
    plt.grid(False)
    return _fig_to_data_url(fig)


def run_geometry_rf_pipeline(
    segments: list[GeometrySegment | dict[str, Any] | Any],
    bbox_m: dict[str, float],
    resolution: float = 10.0,
    default_aps: list[tuple[float, float]] | None = None,
    optimize: bool = True,
    num_aps: int = 2,
) -> dict[str, Any]:
    geometry = [coerce_geometry_segment(seg) for seg in segments]
    active = [seg for seg in geometry if seg.is_obstacle]
    speed = rasterize_segments_to_speed_map(active, bbox_m, resolution_cells_per_m=resolution)
    radio = RadioParams(tx_power_dbm=20, tx_antenna_gain_dbi=4, freq_ghz=5.0, path_loss_exponent=3.0)

    if default_aps is None:
        default_aps = [(bbox_m["width"] * 0.32, bbox_m["height"] * 0.50), (bbox_m["width"] * 0.68, bbox_m["height"] * 0.55)][:num_aps]

    direct_heatmap = _solve_multi_ap(speed, default_aps, radio, resolution)
    direct_img = _plot_heatmap(speed, direct_heatmap, default_aps, "Direct Vector-Parsed (CAD DXF) Wi-Fi Signal Propagation Heatmap", optimized=False)

    opt_img = None
    opt_heatmap = None
    opt_aps = None
    opt_fitness = None
    if optimize:
        pso = pso_place_aps_fmm(speed, n_aps=num_aps, target_rssi_dbm=-68, radio=radio, params=PSOParams(particles=10, iterations=8, seed=13, plateau_iterations=5, n_jobs=1))
        opt_aps = pso["aps"]
        opt_fitness = pso["fitness"]
        opt_heatmap = _solve_multi_ap(speed, opt_aps, radio, resolution)
        opt_img = _plot_heatmap(speed, opt_heatmap, opt_aps, "Particle Swarm Optimized (PSO) Multi-AP Coverage Simulation\n[Mathematics & Computing Engineering Pipeline]", optimized=True)

    final = opt_heatmap if opt_heatmap is not None else direct_heatmap
    return {
        "speed_map_shape": list(speed.speed_map.shape),
        "segment_count": len(geometry),
        "active_obstacle_count": len(active),
        "direct_heatmap_image_data_url": direct_img,
        "optimized_heatmap_image_data_url": opt_img,
        "heatmap_image_data_url": opt_img or direct_img,
        "default_aps": default_aps,
        "optimized_aps": opt_aps,
        "pso_fitness": None if opt_fitness is None else float(opt_fitness),
        "rssi_min_dbm": float(np.nanmin(final)),
        "rssi_max_dbm": float(np.nanmax(final)),
    }
