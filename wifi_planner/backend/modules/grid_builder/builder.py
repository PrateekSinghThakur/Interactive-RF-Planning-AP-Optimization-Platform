"""Generic grid rasterizer: confirmed Building Model -> Building Model with grid.cells.

No Wi-Fi or signal physics lives here.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

__all__ = ["build_grid"]

DEFAULT_ATTENUATION_DB = {
    "open": 0,
    "drywall": 4,
    "wood": 5,
    "glass": 3,
    "brick": 9,
    "concrete": 12,
    "reinforced_concrete": 18,
    "metal": 25,
    "door": 4,
    "window": 3,
}


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _distance_point_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _nearest_room_id(x: float, y: float, rooms: list[dict[str, Any]]) -> str | None:
    for room in rooms:
        if _point_in_polygon(x, y, room["boundary_polygon_m"]):
            return room["id"]
    return None


def _inside_building(x: float, y: float, model: dict[str, Any], room_id: str | None) -> bool:
    footprints = model.get("building_footprints") or []
    if footprints:
        return any(_point_in_polygon(x, y, footprint["boundary_polygon_m"]) for footprint in footprints)
    # Backward-compatible fallback for old fixtures: if no footprint exists, rooms
    # define interior. If even rooms are absent, the floor rectangle is the only
    # usable mask and user ROI/editor tools constrain planning.
    if model.get("rooms"):
        return room_id is not None
    dims = model["metadata"]["floor_dimensions_m"]
    return 0 <= x <= float(dims["width"]) and 0 <= y <= float(dims["height"])


def _on_linear_element(x: float, y: float, element: dict[str, Any], walls_by_id: dict[str, dict[str, Any]], radius: float) -> bool:
    wall = walls_by_id.get(element.get("wall_id"))
    if not wall:
        return False
    wx, wy = element["position_m"]
    if math.hypot(x - wx, y - wy) > max(radius, element.get("width_m", 0.8) / 2):
        return False
    ax, ay = wall["start_m"]
    bx, by = wall["end_m"]
    return _distance_point_segment(x, y, ax, ay, bx, by) <= max(radius, wall.get("thickness_m", 0.2) / 2 + radius)


def build_grid(model: dict[str, Any], resolution_m: float = 0.5, floor_id: str = "floor_1", max_cells: int = 25_000) -> dict[str, Any]:
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    updated = deepcopy(model)
    width = float(model["metadata"]["floor_dimensions_m"]["width"])
    height = float(model["metadata"]["floor_dimensions_m"]["height"])

    # Safety guard: uploaded images may initially have provisional scale. Never
    # create a multi-million-cell grid; automatically coarsen instead.
    estimated_cells = max(1.0, (width / resolution_m) * (height / resolution_m))
    if estimated_cells > max_cells:
        resolution_m = math.sqrt((width * height) / max_cells)

    cols = max(1, int(math.ceil(width / resolution_m)))
    rows = max(1, int(math.ceil(height / resolution_m)))
    walls = model.get("walls", [])
    rooms = model.get("rooms", [])
    walls_by_id = {w["id"]: w for w in walls}
    doors = model.get("doors", [])
    windows = model.get("windows", [])
    cells: list[dict[str, Any]] = []

    for r in range(rows):
        for c in range(cols):
            x = round((c + 0.5) * resolution_m, 4)
            y = round((r + 0.5) * resolution_m, 4)
            room_id = _nearest_room_id(x, y, rooms)
            inside_building = _inside_building(x, y, model, room_id)
            cell_type = "open" if inside_building else "outside"
            attenuation = 0.0
            placeable = bool(inside_building)

            # Critical V2 rule: the building footprint is the simulation domain.
            # Walls outside the footprint must never turn outside cells into
            # coverage/obstacle cells.
            if not inside_building:
                cells.append(
                    {
                        "center_m": [x, y],
                        "type": cell_type,
                        "attenuation_db": attenuation,
                        "room_id": room_id,
                        "placeable": placeable,
                    }
                )
                continue

            for wall in walls:
                ax, ay = wall["start_m"]
                bx, by = wall["end_m"]
                if _distance_point_segment(x, y, ax, ay, bx, by) <= max(resolution_m * 0.55, wall.get("thickness_m", 0.2) / 2):
                    cell_type = "wall"
                    attenuation = float(wall.get("attenuation_db", DEFAULT_ATTENUATION_DB.get(wall.get("material"), 12)))
                    # V2: walls are attenuation obstacles, not the source of the
                    # simulation/AP-candidate domain. Keep interior cells placeable
                    # so changing wall threshold/count does not shrink coverage.
                    placeable = bool(inside_building)
                    break

            for door in doors:
                if _on_linear_element(x, y, door, walls_by_id, resolution_m * 0.75):
                    cell_type = "door"
                    attenuation = float(door.get("attenuation_db", DEFAULT_ATTENUATION_DB["door"]))
                    placeable = bool(inside_building)
                    break

            for window in windows:
                if _on_linear_element(x, y, window, walls_by_id, resolution_m * 0.75):
                    cell_type = "window"
                    attenuation = float(window.get("attenuation_db", DEFAULT_ATTENUATION_DB["window"]))
                    placeable = bool(inside_building)
                    break

            cells.append(
                {
                    "center_m": [x, y],
                    "type": cell_type,
                    "attenuation_db": attenuation,
                    "room_id": room_id,
                    "placeable": placeable,
                }
            )

    updated["grid"] = {
        "resolution_m": resolution_m,
        "origin_m": [0, 0],
        "cols": cols,
        "rows": rows,
        "floor_id": floor_id,
        "cells": cells,
    }
    return updated
