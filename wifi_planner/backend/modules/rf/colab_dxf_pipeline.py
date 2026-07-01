"""Colab-style DXF FMM pipeline wrapped for the web app.

This intentionally mirrors the notebook logic the user validated visually:
- scan DXF layers for wall/glass keywords
- fallback to all LINE/polyline segments as walls if no keywords match
- localize coordinates to building bbox
- convert mm->m when raw dimensions are large
- rasterize speed_map with wall=0.004 and glass=0.45
- solve skfmm.travel_time
- convert to RSSI with the same notebook formula
- generate engineering Matplotlib plots as PNG data URLs
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import cv2
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skfmm
from scipy.ndimage import gaussian_filter

WALL_KEYWORDS = ["WALL", "CONCRETE", "BRICK", "EXTERIOR", "INTERIOR", "STR-"]
GLASS_KEYWORDS = ["WINDOW", "GLASS", "PARTITION", "GLAZ", "DOOR"]


def _fig_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _iter_dxf_segments(msp) -> list[dict[str, Any]]:
    """Extract LINE plus polyline edges as simple segment dictionaries."""
    segments: list[dict[str, Any]] = []
    for line in msp.query("LINE"):
        segments.append({
            "start": (float(line.dxf.start[0]), float(line.dxf.start[1])),
            "end": (float(line.dxf.end[0]), float(line.dxf.end[1])),
            "layer": str(line.dxf.layer),
            "entity": "LINE",
        })
    for pl in msp.query("LWPOLYLINE"):
        pts = [(float(x), float(y)) for x, y, *_ in pl.get_points()]
        for a, b in zip(pts[:-1], pts[1:]):
            segments.append({"start": a, "end": b, "layer": str(pl.dxf.layer), "entity": "LWPOLYLINE"})
        if getattr(pl, "closed", False) and len(pts) > 2:
            segments.append({"start": pts[-1], "end": pts[0], "layer": str(pl.dxf.layer), "entity": "LWPOLYLINE"})
    for pl in msp.query("POLYLINE"):
        pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in pl.vertices]
        for a, b in zip(pts[:-1], pts[1:]):
            segments.append({"start": a, "end": b, "layer": str(pl.dxf.layer), "entity": "POLYLINE"})
        if getattr(pl, "is_closed", False) and len(pts) > 2:
            segments.append({"start": pts[-1], "end": pts[0], "layer": str(pl.dxf.layer), "entity": "POLYLINE"})
    return segments


def _classify_segments(segments: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], str]], int, int, bool]:
    structural: list[tuple[dict[str, Any], str]] = []
    walls = 0
    glass = 0
    for seg in segments:
        layer_name = seg["layer"].upper()
        if any(k in layer_name for k in WALL_KEYWORDS):
            structural.append((seg, "wall"))
            walls += 1
        elif any(k in layer_name for k in GLASS_KEYWORDS):
            structural.append((seg, "glass"))
            glass += 1
    fallback = False
    if len(structural) == 0:
        fallback = True
        structural = [(seg, "wall") for seg in segments]
        walls = len(structural)
        glass = 0
    return structural, walls, glass, fallback


def _build_speed_map(structural_lines, resolution_scale: int):
    x_coords = [s["start"][0] for s, _ in structural_lines] + [s["end"][0] for s, _ in structural_lines]
    y_coords = [s["start"][1] for s, _ in structural_lines] + [s["end"][1] for s, _ in structural_lines]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    raw_width = x_max - x_min
    raw_height = y_max - y_min
    unit_conversion = 0.001 if raw_width > 500 else 1.0
    width_m = raw_width * unit_conversion
    height_m = raw_height * unit_conversion
    h_pixels = max(10, int(height_m * resolution_scale))
    w_pixels = max(10, int(width_m * resolution_scale))
    speed_map = np.ones((h_pixels, w_pixels), dtype=np.float64)

    for seg, mat_type in structural_lines:
        x1, y1 = seg["start"]
        x2, y2 = seg["end"]
        lx1 = (x1 - x_min) * unit_conversion
        ly1 = (y1 - y_min) * unit_conversion
        lx2 = (x2 - x_min) * unit_conversion
        ly2 = (y2 - y_min) * unit_conversion
        px1 = int(lx1 * resolution_scale)
        py1 = int(ly1 * resolution_scale)
        px2 = int(lx2 * resolution_scale)
        py2 = int(ly2 * resolution_scale)
        px1, px2 = np.clip(px1, 0, w_pixels - 1), np.clip(px2, 0, w_pixels - 1)
        py1, py2 = np.clip(py1, 0, h_pixels - 1), np.clip(py2, 0, h_pixels - 1)
        if mat_type == "wall":
            cv2.line(speed_map, (px1, py1), (px2, py2), 0.004, thickness=3)
        elif mat_type == "glass":
            cv2.line(speed_map, (px1, py1), (px2, py2), 0.45, thickness=2)
    meta = {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "raw_width": raw_width,
        "raw_height": raw_height,
        "unit_conversion": unit_conversion,
        "width_meters": width_m,
        "height_meters": height_m,
        "h_pixels": h_pixels,
        "w_pixels": w_pixels,
    }
    return speed_map, meta


def _solve_heatmap(speed_map: np.ndarray, aps_m: list[dict[str, Any]], resolution_scale: int, path_loss_exponent: float):
    h_pixels, w_pixels = speed_map.shape
    master = np.full((h_pixels, w_pixels), -100.0, dtype=np.float64)
    for ap in aps_m:
        ap_px = int(ap["x_meters"] * resolution_scale)
        ap_py = int(ap["y_meters"] * resolution_scale)
        ap_px = int(np.clip(ap_px, 0, w_pixels - 1))
        ap_py = int(np.clip(ap_py, 0, h_pixels - 1))
        phi = np.ones((h_pixels, w_pixels), dtype=np.float64)
        phi[ap_py, ap_px] = 0.0
        travel_times = skfmm.travel_time(phi, speed_map, dx=1.0 / resolution_scale)
        p_tx, g_tx, l_0 = 20, 4, -40
        effective_dist = travel_times + 0.1
        individual_rssi = p_tx + g_tx + (l_0 - (10 * path_loss_exponent * np.log10(effective_dist)))
        master = np.maximum(master, individual_rssi)
    heatmap = gaussian_filter(master, sigma=2.0)
    return np.clip(heatmap, -90, -40)


def _plot_direct(speed_map, heatmap, aps_m, resolution_scale):
    fig = plt.figure(figsize=(13, 9), dpi=130)
    plt.imshow(speed_map == 1.0, cmap="gray", origin="lower", alpha=0.20)
    overlay = plt.imshow(heatmap, cmap="jet", alpha=0.60, origin="lower", vmin=-90, vmax=-40)
    for ap in aps_m:
        ap_px = int(ap["x_meters"] * resolution_scale)
        ap_py = int(ap["y_meters"] * resolution_scale)
        plt.scatter(ap_px, ap_py, color="magenta", edgecolor="black", s=300, marker="*", label=f"{ap['label']} ({ap['x_meters']}m, {ap['y_meters']}m)", linewidth=2, zorder=10)
    plt.title("Direct Vector-Parsed (CAD DXF) Wi-Fi Signal Propagation Heatmap", fontsize=13, pad=15)
    plt.xlabel("X Dimensions (Meters)", fontsize=10)
    plt.ylabel("Y Dimensions (Meters)", fontsize=10)
    plt.legend(loc="upper right", frameon=True, facecolor="white", shadow=True)
    cbar = plt.colorbar(overlay, orientation="horizontal", pad=0.08, shrink=0.6)
    cbar.set_label("Receiver Signal Strength Intensity Profile Scale (dBm)", fontsize=11)
    return _fig_to_data_url(fig)


def _evaluate_network_coverage(test_aps_px, speed_map, dx_scale, path_loss_exponent):
    h_g, w_g = speed_map.shape
    combined = np.full((h_g, w_g), -100.0, dtype=np.float64)
    for ax_px, ay_px in test_aps_px:
        if speed_map[ay_px, ax_px] <= 0.01:
            return -99999
        phi = np.ones((h_g, w_g), dtype=np.float64)
        phi[ay_px, ax_px] = 0.0
        t_times = skfmm.travel_time(phi, speed_map, dx=dx_scale)
        rssi = 20 + 4 + (-40 - (10 * path_loss_exponent * np.log10(t_times + 0.1)))
        combined = np.maximum(combined, rssi)
    return int(np.sum(combined >= -68))


def _simple_pso_positions(speed_map, resolution_scale, num_aps=2, particles=15, iterations=10, path_loss_exponent=3.0):
    h_pixels, w_pixels = speed_map.shape
    rng = np.random.default_rng(7)
    particles_pos = []
    open_positions = np.argwhere(speed_map > 0.5)
    if len(open_positions) == 0:
        open_positions = np.argwhere(np.ones_like(speed_map, dtype=bool))
    for _ in range(particles):
        p_aps = []
        chosen = rng.choice(len(open_positions), size=num_aps, replace=len(open_positions) < num_aps)
        for idx in chosen:
            ry, rx = open_positions[idx]
            p_aps.append((int(rx), int(ry)))
        particles_pos.append(p_aps)

    best_score = -float("inf")
    best_positions = particles_pos[0]
    scores_cache = {}
    for _it in range(iterations):
        for current in particles_pos:
            key = tuple(current)
            if key not in scores_cache:
                scores_cache[key] = _evaluate_network_coverage(current, speed_map, 1.0 / resolution_scale, path_loss_exponent)
            score = scores_cache[key]
            if score > best_score:
                best_score = score
                best_positions = current
    return best_positions, best_score


def _plot_optimized(speed_map, heatmap, best_positions, resolution_scale):
    fig = plt.figure(figsize=(15, 10), dpi=140)
    plt.imshow(speed_map == 1.0, cmap="gray", origin="lower", alpha=0.25)
    overlay = plt.imshow(heatmap, cmap="jet", alpha=0.55, origin="lower", vmin=-90, vmax=-40)
    for idx, (bx, by) in enumerate(best_positions):
        local_x = bx / resolution_scale
        local_y = by / resolution_scale
        plt.scatter(bx, by, color="magenta", edgecolor="black", s=350, marker="*", label=f"Optimized AP {idx+1} ({local_x:.1f}m, {local_y:.1f}m)", linewidth=2.5, zorder=10)
    plt.title("Particle Swarm Optimized (PSO) Multi-AP Coverage Simulation\n[Mathematics & Computing Engineering Pipeline]", fontsize=13, pad=15)
    plt.xlabel("Local Building Width (Meters)", fontsize=10)
    plt.ylabel("Local Building Height (Meters)", fontsize=10)
    h_pixels, w_pixels = speed_map.shape
    x_ticks = np.arange(0, w_pixels, resolution_scale * 10)
    y_ticks = np.arange(0, h_pixels, resolution_scale * 5)
    plt.xticks(x_ticks, [f"{t/resolution_scale:.0f}m" for t in x_ticks])
    plt.yticks(y_ticks, [f"{t/resolution_scale:.0f}m" for t in y_ticks])
    plt.legend(loc="upper right", frameon=True, facecolor="white", shadow=True)
    cbar = plt.colorbar(overlay, orientation="horizontal", pad=0.08, shrink=0.6)
    cbar.set_label("Receiver Signal Strength Indicator Scale (dBm)   [ Red = Maximum Strength | Blue = Weak Coverage ]", fontsize=11)
    plt.grid(False)
    return _fig_to_data_url(fig)


def run_colab_dxf_pipeline(
    dxf_path: str | Path,
    access_points: list[dict[str, Any]] | None = None,
    path_loss_exponent: float = 3.0,
    resolution_scale: int = 10,
    optimize: bool = True,
    num_aps_to_deploy: int = 2,
) -> dict[str, Any]:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    all_layers = [layer.dxf.name for layer in doc.layers]
    all_segments = _iter_dxf_segments(msp)
    structural_lines, assigned_wall, assigned_glass, fallback = _classify_segments(all_segments)
    if not structural_lines:
        raise ValueError("No line/polyline geometry found in DXF")

    speed_map, meta = _build_speed_map(structural_lines, resolution_scale)
    if access_points is None:
        access_points = [
            {"x_meters": 15.0, "y_meters": 12.0, "label": "AP 1 (Core)"},
            {"x_meters": 35.0, "y_meters": 20.0, "label": "AP 2 (Wing)"},
        ]
    # Clip configured APs to local map dimensions.
    aps = []
    for ap in access_points:
        aps.append({
            "x_meters": float(np.clip(ap["x_meters"], 0, meta["width_meters"])),
            "y_meters": float(np.clip(ap["y_meters"], 0, meta["height_meters"])),
            "label": ap.get("label", "AP"),
        })
    direct_heatmap = _solve_heatmap(speed_map, aps, resolution_scale, path_loss_exponent)
    direct_image = _plot_direct(speed_map, direct_heatmap, aps, resolution_scale)

    best_positions = None
    optimized_image = None
    optimized_heatmap = None
    best_score = None
    if optimize:
        best_positions, best_score = _simple_pso_positions(speed_map, resolution_scale, num_aps_to_deploy, path_loss_exponent=path_loss_exponent)
        opt_aps = [{"x_meters": bx / resolution_scale, "y_meters": by / resolution_scale, "label": f"Optimized AP {i+1}"} for i, (bx, by) in enumerate(best_positions)]
        optimized_heatmap = _solve_heatmap(speed_map, opt_aps, resolution_scale, path_loss_exponent)
        optimized_image = _plot_optimized(speed_map, optimized_heatmap, best_positions, resolution_scale)

    final_heatmap = optimized_heatmap if optimized_heatmap is not None else direct_heatmap
    return {
        "layers": sorted(all_layers),
        "assigned_wall": assigned_wall,
        "assigned_glass": assigned_glass,
        "fallback_used": fallback,
        "segment_count": len(all_segments),
        "structural_segment_count": len(structural_lines),
        "bbox": meta,
        "speed_map_shape": list(speed_map.shape),
        "direct_heatmap_image_data_url": direct_image,
        "optimized_heatmap_image_data_url": optimized_image,
        "heatmap_image_data_url": optimized_image or direct_image,
        "best_global_positions_px": best_positions,
        "best_global_score_pixels": best_score,
        "rssi_min_dbm": float(np.nanmin(final_heatmap)),
        "rssi_max_dbm": float(np.nanmax(final_heatmap)),
    }
