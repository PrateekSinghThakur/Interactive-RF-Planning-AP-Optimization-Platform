import importlib.util

import numpy as np
import pytest

skfmm_available = importlib.util.find_spec("skfmm") is not None


@pytest.mark.skipif(not skfmm_available, reason="scikit-fmm not installed")
def test_fmm_solver_shapes_and_ap_is_strongest():
    from backend.modules.rf.fmm_solver import make_rect_room_with_wall, solve_fmm_rssi

    speed, ap, meta = make_rect_room_with_wall(width_m=8, height_m=5, resolution_cells_per_m=5)
    result = solve_fmm_rssi(speed, ap, meta["resolution_cells_per_m"], smooth_sigma_cells=None)

    assert result["travel_time_m"].shape == speed.shape
    assert result["rssi_dbm"].shape == speed.shape
    assert np.isfinite(result["rssi_dbm"]).all()

    ap_col = int(round(ap[0] * meta["resolution_cells_per_m"]))
    ap_row = int(round(ap[1] * meta["resolution_cells_per_m"]))
    assert result["travel_time_m"][ap_row, ap_col] == pytest.approx(0.0)
    assert result["rssi_dbm"][ap_row, ap_col] == pytest.approx(np.max(result["rssi_dbm"]), rel=1e-6)


@pytest.mark.skipif(not skfmm_available, reason="scikit-fmm not installed")
def test_wall_slowdown_increases_travel_time():
    from backend.modules.rf.fmm_solver import make_rect_room_with_wall, solve_fmm_rssi

    speed_wall, ap, meta = make_rect_room_with_wall(width_m=12, height_m=6, resolution_cells_per_m=6)
    speed_free = np.ones_like(speed_wall)

    wall_result = solve_fmm_rssi(speed_wall, ap, meta["resolution_cells_per_m"], smooth_sigma_cells=None)
    free_result = solve_fmm_rssi(speed_free, ap, meta["resolution_cells_per_m"], smooth_sigma_cells=None)

    # Point behind the wall relative to AP.
    row = int(round(3.0 * meta["resolution_cells_per_m"]))
    col = int(round(10.0 * meta["resolution_cells_per_m"]))

    assert wall_result["travel_time_m"][row, col] > free_result["travel_time_m"][row, col]
    assert wall_result["rssi_dbm"][row, col] < free_result["rssi_dbm"][row, col]
