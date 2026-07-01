#!/usr/bin/env python3
"""Phase 3 demo: DXF -> speed map -> FMM RSSI heatmap."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from backend.modules.cad.dxf_parser import parse_dxf, print_layer_report
from backend.modules.rf.fmm_solver import RadioParams, solve_fmm_rssi
from backend.modules.rf.speed_map import rasterize_segments_to_speed_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/dxf_fmm"))
    parser.add_argument("--res", type=float, default=10.0, help="cells per meter")
    parser.add_argument("--ap-x", type=float, default=None)
    parser.add_argument("--ap-y", type=float, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    parsed = parse_dxf(args.dxf)
    print_layer_report(parsed)

    classified_segments = [s for s in parsed.segments if not s.requires_classification]
    if not classified_segments:
        raise SystemExit("No classified structural segments found. Provide layer_material_map or classify layers.")

    speed_result = rasterize_segments_to_speed_map(classified_segments, parsed.bbox_m, resolution_cells_per_m=args.res)

    ap_x = args.ap_x if args.ap_x is not None else parsed.bbox_m["width"] * 0.30
    ap_y = args.ap_y if args.ap_y is not None else parsed.bbox_m["height"] * 0.50
    fmm = solve_fmm_rssi(
        speed_result.speed_map,
        (ap_x, ap_y),
        resolution_cells_per_m=args.res,
        radio=RadioParams(tx_power_dbm=20, freq_ghz=5.0, path_loss_exponent=2.2),
        smooth_sigma_cells=1.0,
    )

    ap_row, ap_col = speed_result.xy_m_to_row_col(ap_x, ap_y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = axes[0].imshow(speed_result.speed_map, cmap="viridis", origin="upper")
    axes[0].scatter([ap_col], [ap_row], c="red", s=45)
    axes[0].set_title("DXF speed map")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(fmm["travel_time_m"], cmap="magma", origin="upper")
    axes[1].scatter([ap_col], [ap_row], c="cyan", s=45)
    axes[1].set_title("FMM travel time")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(fmm["rssi_dbm_smoothed"], cmap="jet", origin="upper", vmin=-90, vmax=-35)
    axes[2].scatter([ap_col], [ap_row], c="white", edgecolors="black", s=60)
    axes[2].set_title("RSSI heatmap")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="dBm")

    fig.savefig(args.out / "dxf_to_fmm_heatmap.png", dpi=180)
    plt.close(fig)

    np.save(args.out / "speed_map.npy", speed_result.speed_map)
    np.save(args.out / "rssi_dbm.npy", fmm["rssi_dbm"])
    print(f"Saved {args.out / 'dxf_to_fmm_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
