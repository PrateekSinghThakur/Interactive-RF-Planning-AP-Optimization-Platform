"""Fast Marching Method Eikonal solver for Wi-Fi RSSI heatmaps.

Phase 1 scope only: standalone Eikonal/FMM propagation on a hand-built speed map.
No DXF parsing, no PSO, no UI integration, no floorplan detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional import so the module gives a clear error if dependency is absent.
    import skfmm  # type: ignore
except Exception:  # pragma: no cover - handled by _require_skfmm
    skfmm = None


@dataclass(frozen=True)
class RadioParams:
    """Parameters for the log-distance path-loss model.

    The received signal model used here is:

        RSSI(dBm) = P_tx + G_tx + G_rx - PL(d_eff)

    with

        PL(d_eff) = PL(d0) + 10*n*log10(d_eff / d0)

    where:
      - d0 is a reference distance, normally 1 meter.
      - n is the path-loss exponent.
      - PL(d0) defaults to free-space path loss at d0 for freq_ghz.
      - d_eff is an "effective distance" produced by the Eikonal solver.

    Mathematical caveat:
      The FMM travel_time T solves |grad T| = 1 / f(x), where f(x) is a
      dimensionless speed map normalized so free space has f=1. With dx in
      meters, T has units of meters when f is dimensionless. In free space,
      T equals Euclidean distance. In slower regions, T = integral(ds/f),
      which behaves like a refractive/attenuation-weighted path length.

      Therefore this implementation uses:

          d_eff = T * reference_speed

      with reference_speed = 1.0 by default. This is an approximation: it treats
      attenuation obstacles as increasing effective propagation distance rather
      than separately adding material loss in dB. This is mathematically
      consistent as a cost-distance model, but it is not a full Maxwell/RF model.
      Confirm this modeling choice before using it as the final physics model.
    """

    tx_power_dbm: float = 20.0
    tx_antenna_gain_dbi: float = 0.0
    rx_antenna_gain_dbi: float = 0.0
    freq_ghz: float = 5.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.2
    reference_path_loss_db: float | None = None
    reference_speed: float = 1.0


def _require_skfmm() -> Any:
    if skfmm is None:
        raise ImportError(
            "scikit-fmm is required for the Eikonal solver. Install with: "
            "python -m pip install scikit-fmm"
        )
    return skfmm


def free_space_path_loss_db(distance_m: float, freq_ghz: float) -> float:
    """Free-space path loss in dB for distance in meters and frequency in GHz.

    Standard form:
        FSPL = 32.44 + 20 log10(d_km) + 20 log10(f_MHz)
    """
    d_km = max(distance_m, 1e-9) / 1000.0
    f_mhz = freq_ghz * 1000.0
    return 32.44 + 20.0 * np.log10(d_km) + 20.0 * np.log10(f_mhz)


def attenuation_db_to_speed(
    attenuation_db: float,
    alpha: float = 0.09,
    min_speed: float = 0.04,
    max_speed: float = 1.0,
) -> float:
    """Map material attenuation to dimensionless Eikonal speed.

    This helper is used only for the Phase 1 synthetic test map. A later
    rasterizer module should own the production mapping.

    Mapping:
        speed = exp(-alpha * attenuation_db), clipped to [min_speed, max_speed]

    Interpretation:
      Higher attenuation means a slower wavefront speed. Since FMM travel time is
      integral(ds / speed), slower cells increase effective propagation distance.

    Approximation warning:
      This is a tunable phenomenological mapping, not a first-principles RF law.
      It must be calibrated against measurements or accepted as a planning
      approximation.
    """
    return float(np.clip(np.exp(-alpha * attenuation_db), min_speed, max_speed))


def solve_fmm_rssi(
    speed_map: np.ndarray,
    ap_xy_m: tuple[float, float],
    resolution_cells_per_m: float,
    radio: RadioParams | None = None,
    smooth_sigma_cells: float | None = 1.0,
) -> dict[str, np.ndarray]:
    """Solve Eikonal travel time and convert to RSSI heatmap.

    Parameters
    ----------
    speed_map:
        2D array of positive dimensionless speeds. Free space should be 1.0.
        Obstacles/walls should be lower but nonzero, e.g. 0.05-0.5.

    ap_xy_m:
        Access point location in meters as (x, y). x maps to columns, y to rows.

    resolution_cells_per_m:
        Grid resolution. For example, 10 means each cell is 0.1 m.

    radio:
        Log-distance RF model parameters.

    smooth_sigma_cells:
        Optional Gaussian sigma for visualization only. The returned raw RSSI is
        not smoothed. If scipy is unavailable or sigma is None/0, smoothing is
        skipped.

    Returns
    -------
    dict with:
        travel_time_m:
            FMM cost distance. In free space this approximates Euclidean meters.
        effective_distance_m:
            travel_time_m * reference_speed, clipped to reference distance.
        rssi_dbm:
            Raw RSSI heatmap from log-distance path loss.
        rssi_dbm_smoothed:
            Smoothed heatmap for visualization only.

    Mathematical model
    ------------------
    FMM solves the Eikonal equation:

        |∇T(x,y)| = 1 / f(x,y)

    with T=0 at the AP. f(x,y) is the speed map. Since free-space f=1 and dx is
    meters, T is measured in meter-equivalent cost distance. Low-speed wall cells
    increase T and produce smooth diffraction-like bending around obstacles.

    RSSI conversion uses a log-distance path-loss model with d_eff=T. This is an
    approximation that treats wall interaction as increased effective distance,
    not as explicit wall-crossing dB subtraction.
    """
    _skfmm = _require_skfmm()
    radio = radio or RadioParams()

    if speed_map.ndim != 2:
        raise ValueError("speed_map must be a 2D array")
    if resolution_cells_per_m <= 0:
        raise ValueError("resolution_cells_per_m must be positive")

    speed = np.asarray(speed_map, dtype=np.float64)
    if not np.all(np.isfinite(speed)):
        raise ValueError("speed_map contains NaN or infinite values")
    if np.any(speed <= 0):
        raise ValueError("speed_map must be strictly positive; use a small epsilon for walls")

    rows, cols = speed.shape
    ap_col = int(round(ap_xy_m[0] * resolution_cells_per_m))
    ap_row = int(round(ap_xy_m[1] * resolution_cells_per_m))
    if not (0 <= ap_row < rows and 0 <= ap_col < cols):
        raise ValueError(f"AP {ap_xy_m} maps outside speed_map shape {speed.shape}")

    # skfmm.travel_time computes first-arrival travel time from the zero level set.
    # Setting phi=1 everywhere and phi=0 at the AP creates a point source at the AP.
    phi = np.ones_like(speed, dtype=np.float64)
    phi[ap_row, ap_col] = 0.0

    dx = 1.0 / resolution_cells_per_m  # meters per grid cell
    travel_time = _skfmm.travel_time(phi, speed, dx=dx)
    travel_time = np.asarray(travel_time, dtype=np.float64)

    # Since speed is normalized to free-space speed=1, travel_time has units of
    # meter-equivalent cost distance. Multiplying by reference_speed keeps the
    # derivation explicit if a later model changes speed units.
    effective_distance = travel_time * radio.reference_speed
    effective_distance = np.maximum(effective_distance, radio.reference_distance_m)

    pl0 = radio.reference_path_loss_db
    if pl0 is None:
        pl0 = free_space_path_loss_db(radio.reference_distance_m, radio.freq_ghz)

    path_loss = pl0 + 10.0 * radio.path_loss_exponent * np.log10(
        effective_distance / radio.reference_distance_m
    )
    rssi = radio.tx_power_dbm + radio.tx_antenna_gain_dbi + radio.rx_antenna_gain_dbi - path_loss

    rssi_smooth = rssi.copy()
    if smooth_sigma_cells and smooth_sigma_cells > 0:
        try:
            from scipy.ndimage import gaussian_filter

            rssi_smooth = gaussian_filter(rssi, sigma=smooth_sigma_cells)
        except Exception:
            # Visualization smoothing is optional; physics result is rssi.
            rssi_smooth = rssi.copy()

    return {
        "travel_time_m": travel_time,
        "effective_distance_m": effective_distance,
        "rssi_dbm": rssi,
        "rssi_dbm_smoothed": rssi_smooth,
    }


def make_rect_room_with_wall(
    width_m: float = 20.0,
    height_m: float = 12.0,
    resolution_cells_per_m: float = 10.0,
    wall_attenuation_db: float = 18.0,
) -> tuple[np.ndarray, tuple[float, float], dict[str, Any]]:
    """Create a synthetic rectangular room with one internal wall.

    The wall is a slow-speed vertical stripe with openings at top and bottom so
    the Eikonal wavefront can diffract around it. This should produce smooth
    shadowing rather than a hard ray-casting shadow.
    """
    rows = int(round(height_m * resolution_cells_per_m))
    cols = int(round(width_m * resolution_cells_per_m))
    speed = np.ones((rows, cols), dtype=np.float64)

    wall_speed = attenuation_db_to_speed(wall_attenuation_db)
    wall_col = int(round((width_m * 0.52) * resolution_cells_per_m))
    y0 = int(round((height_m * 0.18) * resolution_cells_per_m))
    y1 = int(round((height_m * 0.82) * resolution_cells_per_m))
    thickness = max(2, int(round(0.25 * resolution_cells_per_m)))
    speed[y0:y1, wall_col : wall_col + thickness] = wall_speed

    ap_xy_m = (width_m * 0.25, height_m * 0.50)
    meta = {
        "width_m": width_m,
        "height_m": height_m,
        "wall_col": wall_col,
        "wall_y0": y0,
        "wall_y1": y1,
        "wall_speed": wall_speed,
        "resolution_cells_per_m": resolution_cells_per_m,
    }
    return speed, ap_xy_m, meta


def run_phase1_demo(output_dir: str | Path = "outputs/fmm_phase1") -> dict[str, np.ndarray]:
    """Run the Phase 1 synthetic demo and save visualizations."""
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    speed, ap_xy_m, meta = make_rect_room_with_wall()
    result = solve_fmm_rssi(
        speed,
        ap_xy_m,
        resolution_cells_per_m=meta["resolution_cells_per_m"],
        radio=RadioParams(tx_power_dbm=20.0, freq_ghz=5.0, path_loss_exponent=2.2),
        smooth_sigma_cells=1.2,
    )

    ap_col = int(round(ap_xy_m[0] * meta["resolution_cells_per_m"]))
    ap_row = int(round(ap_xy_m[1] * meta["resolution_cells_per_m"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(speed, cmap="viridis", origin="upper")
    axes[0].scatter([ap_col], [ap_row], c="red", s=45, marker="o", label="AP")
    axes[0].set_title("Speed map: wall = slower medium")
    axes[0].axis("off")
    axes[0].legend(loc="lower right")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="dimensionless speed")

    im1 = axes[1].imshow(result["travel_time_m"], cmap="magma", origin="upper")
    axes[1].scatter([ap_col], [ap_row], c="cyan", s=45, marker="o")
    axes[1].set_title("FMM travel time / effective distance")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="m-equivalent")

    im2 = axes[2].imshow(result["rssi_dbm_smoothed"], cmap="jet", origin="upper", vmin=-90, vmax=-35)
    axes[2].scatter([ap_col], [ap_row], c="white", edgecolors="black", s=60, marker="o")
    axes[2].set_title("RSSI heatmap (smoothed for display)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="dBm")

    fig.suptitle("Phase 1: Eikonal/FMM Wi-Fi propagation demo", fontsize=14)
    fig.savefig(output_path / "fmm_phase1_demo.png", dpi=180)
    plt.close(fig)

    np.save(output_path / "speed_map.npy", speed)
    np.save(output_path / "travel_time_m.npy", result["travel_time_m"])
    np.save(output_path / "rssi_dbm.npy", result["rssi_dbm"])

    print(f"Saved demo visualization to: {output_path / 'fmm_phase1_demo.png'}")
    print(f"AP location (m): {ap_xy_m}")
    print(f"Wall speed: {meta['wall_speed']:.4f}")
    print(f"RSSI range raw: {np.nanmin(result['rssi_dbm']):.2f} to {np.nanmax(result['rssi_dbm']):.2f} dBm")
    return result


if __name__ == "__main__":
    run_phase1_demo()
