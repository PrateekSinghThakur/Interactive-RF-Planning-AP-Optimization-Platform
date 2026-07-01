"""Speed-map rasterizer for DXF/manual structural segments.

Phase 3 scope:
- Take structural segments from the DXF parser (or any same-shaped segment dict).
- Create a rectangular free-space speed map over the DXF bounding box.
- Rasterize wall/window/door segments as lower-speed cells.
- Output arrays and coordinate transforms ready for the FMM solver.

This module does not parse DXF and does not run PSO. It is the bridge between
vector geometry and the Eikonal/FMM solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from backend.modules.rf.fmm_solver import attenuation_db_to_speed


@dataclass(frozen=True)
class SpeedMapResult:
    """Rasterized speed map and coordinate metadata.

    Coordinate convention:
    - Input geometry is in meters with x right, y up/plan coordinates.
    - Grid arrays are indexed as [row, col].
    - col = round((x - min_x) * cells_per_m)
    - row = round((max_y - y) * cells_per_m)

    We flip y because image/array rows increase downward while CAD y usually
    increases upward. The inverse transform recovers plan coordinates.
    """

    speed_map: np.ndarray
    attenuation_map_db: np.ndarray
    bbox_m: dict[str, float]
    resolution_cells_per_m: float
    segment_count: int

    def xy_m_to_row_col(self, x: float, y: float) -> tuple[int, int]:
        col = int(round((x - self.bbox_m["min_x"]) * self.resolution_cells_per_m))
        row = int(round((self.bbox_m["max_y"] - y) * self.resolution_cells_per_m))
        row = int(np.clip(row, 0, self.speed_map.shape[0] - 1))
        col = int(np.clip(col, 0, self.speed_map.shape[1] - 1))
        return row, col

    def row_col_to_xy_m(self, row: int, col: int) -> tuple[float, float]:
        x = self.bbox_m["min_x"] + col / self.resolution_cells_per_m
        y = self.bbox_m["max_y"] - row / self.resolution_cells_per_m
        return (float(x), float(y))


def _segment_get(segment: Any, key: str) -> Any:
    if isinstance(segment, dict):
        return segment[key]
    return getattr(segment, key)


def _normalize_bbox(bbox_m: dict[str, float]) -> dict[str, float]:
    min_x = float(bbox_m.get("min_x", 0.0))
    min_y = float(bbox_m.get("min_y", 0.0))
    if "max_x" in bbox_m and "max_y" in bbox_m:
        max_x = float(bbox_m["max_x"])
        max_y = float(bbox_m["max_y"])
    else:
        max_x = min_x + float(bbox_m["width"])
        max_y = min_y + float(bbox_m["height"])
    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def rasterize_segments_to_speed_map(
    segments: Sequence[Any],
    bbox_m: dict[str, float],
    resolution_cells_per_m: float = 10.0,
    wall_thickness_m: float = 0.15,
    free_space_speed: float = 1.0,
    attenuation_alpha: float = 0.09,
    min_speed: float = 0.04,
) -> SpeedMapResult:
    """Rasterize structural vector segments into an FMM speed map.

    Parameters
    ----------
    segments:
        Sequence of objects/dicts with start_xy, end_xy, attenuation_db.
        This matches the output of Module 1 DXF parser.

    bbox_m:
        Building/drawing bounding box in meters.

    resolution_cells_per_m:
        Grid resolution. 10 means 10 cells per meter, dx=0.1 m.

    wall_thickness_m:
        Rasterized line thickness for vector segments. This is a geometric
        modeling parameter; real wall thickness may later come from DXF metadata.

    free_space_speed:
        Dimensionless speed of unobstructed space. The FMM solver assumes this is
        1.0 so travel_time has meter-equivalent units.

    attenuation_alpha, min_speed:
        Parameters for attenuation_db_to_speed. Higher attenuation creates slower
        cells. This is a tunable approximation, not a first-principles RF law.

    Mathematical note
    -----------------
    The Eikonal solver uses |grad T| = 1/f(x,y). By lowering f along walls, the
    shortest-time path may bend around obstacles, giving smoother shadowing than
    straight-line ray casting. In this Phase 3 rasterizer, material attenuation is
    encoded as speed reduction:

        speed = exp(-alpha * attenuation_db), clipped to [min_speed, 1]

    Later calibration can change this mapping without changing the FMM solver.
    """
    if resolution_cells_per_m <= 0:
        raise ValueError("resolution_cells_per_m must be positive")

    bbox = _normalize_bbox(bbox_m)
    cols = max(2, int(np.ceil(bbox["width"] * resolution_cells_per_m)) + 1)
    rows = max(2, int(np.ceil(bbox["height"] * resolution_cells_per_m)) + 1)

    speed_map = np.full((rows, cols), float(free_space_speed), dtype=np.float64)
    attenuation_map = np.zeros((rows, cols), dtype=np.float64)
    thickness_px = max(1, int(round(wall_thickness_m * resolution_cells_per_m)))

    def xy_to_cv_point(x: float, y: float) -> tuple[int, int]:
        col = int(round((x - bbox["min_x"]) * resolution_cells_per_m))
        row = int(round((bbox["max_y"] - y) * resolution_cells_per_m))
        col = int(np.clip(col, 0, cols - 1))
        row = int(np.clip(row, 0, rows - 1))
        return (col, row)

    for segment in segments:
        start = _segment_get(segment, "start_xy")
        end = _segment_get(segment, "end_xy")
        attenuation_db = float(_segment_get(segment, "attenuation_db"))
        if attenuation_db <= 0:
            # Unclassified layers are intentionally non-obstacles until the user
            # classifies them. This prevents furniture/annotation layers from
            # silently becoming walls.
            continue

        p1 = xy_to_cv_point(float(start[0]), float(start[1]))
        p2 = xy_to_cv_point(float(end[0]), float(end[1]))
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.line(mask, p1, p2, 255, thickness=thickness_px)

        segment_speed = attenuation_db_to_speed(
            attenuation_db,
            alpha=attenuation_alpha,
            min_speed=min_speed,
            max_speed=free_space_speed,
        )
        # If multiple segments overlap, keep the most attenuating one: lowest speed,
        # highest attenuation.
        hit = mask > 0
        speed_map[hit] = np.minimum(speed_map[hit], segment_speed)
        attenuation_map[hit] = np.maximum(attenuation_map[hit], attenuation_db)

    return SpeedMapResult(
        speed_map=speed_map,
        attenuation_map_db=attenuation_map,
        bbox_m=bbox,
        resolution_cells_per_m=float(resolution_cells_per_m),
        segment_count=len(segments),
    )
