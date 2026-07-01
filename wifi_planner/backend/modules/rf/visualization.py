"""Visualization helpers for speed maps, FMM heatmaps, and APs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.modules.rf.speed_map import SpeedMapResult


def plot_speed_and_heatmap(
    speed_result: SpeedMapResult,
    rssi_dbm: np.ndarray,
    ap_xy: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    output_path: str | Path,
    vmin: float = -90,
    vmax: float = -40,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    im0 = axes[0].imshow(speed_result.speed_map, cmap="viridis", origin="upper")
    axes[0].set_title("Speed map / structural obstacles")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rssi_dbm, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)
    for ap in ap_xy:
        row, col = speed_result.xy_m_to_row_col(ap[0], ap[1])
        axes[1].scatter([col], [row], c="white", edgecolors="black", s=70)
    axes[1].set_title("RSSI heatmap")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="dBm")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
