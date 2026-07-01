"""Manual spline/freehand wall tracing utilities.

Phase 4 scope:
- Take a user freehand stroke as points.
- Simplify it with Ramer-Douglas-Peucker (RDP).
- Snap nearly horizontal/vertical segments.
- Snap endpoints to existing wall endpoints.
- Emit the same structural segment shape as DXF parsing.

No automatic raster wall detection is performed here. The user provides the wall
stroke; this module only cleans it into segment geometry.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from backend.modules.geometry.segments import MATERIAL_ATTENUATION_DB, make_geometry_segment

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def ramer_douglas_peucker(points: Sequence[Point], epsilon: float) -> list[Point]:
    """Simplify a freehand polyline with RDP.

    RDP recursively keeps points that deviate by more than epsilon from the line
    between endpoints. It is a geometric simplification; it does not infer walls.
    """
    pts = list(points)
    if len(pts) < 3:
        return pts

    start = pts[0]
    end = pts[-1]
    max_dist = -1.0
    index = 0
    for i in range(1, len(pts) - 1):
        dist = _point_line_distance(pts[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    if max_dist > epsilon:
        left = ramer_douglas_peucker(pts[: index + 1], epsilon)
        right = ramer_douglas_peucker(pts[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def snap_axis_aligned_segments(points: Sequence[Point], angle_tolerance_deg: float = 5.0) -> list[Point]:
    """Snap near-horizontal/vertical segment endpoints.

    If a segment angle is within tolerance of 0/180 degrees, make y equal.
    If within tolerance of 90 degrees, make x equal.
    """
    if len(points) < 2:
        return list(points)
    snapped = [tuple(points[0])]
    for end in points[1:]:
        sx, sy = snapped[-1]
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        horizontal_delta = min(abs(angle), abs(angle - 180))
        vertical_delta = abs(angle - 90)
        if horizontal_delta <= angle_tolerance_deg:
            snapped.append((float(ex), float(sy)))
        elif vertical_delta <= angle_tolerance_deg:
            snapped.append((float(sx), float(ey)))
        else:
            snapped.append((float(ex), float(ey)))
    return snapped


def _existing_endpoints(existing_segments: Sequence[dict[str, Any] | Any]) -> list[Point]:
    endpoints: list[Point] = []
    for seg in existing_segments:
        if isinstance(seg, dict):
            start = seg.get("start_xy") or seg.get("start_m")
            end = seg.get("end_xy") or seg.get("end_m")
        else:
            start = getattr(seg, "start_xy", None)
            end = getattr(seg, "end_xy", None)
        if start is not None:
            endpoints.append((float(start[0]), float(start[1])))
        if end is not None:
            endpoints.append((float(end[0]), float(end[1])))
    return endpoints


def snap_points_to_existing_endpoints(
    points: Sequence[Point],
    existing_segments: Sequence[dict[str, Any] | Any] = (),
    tolerance: float = 8.0,
) -> list[Point]:
    endpoints = _existing_endpoints(existing_segments)
    snapped: list[Point] = []
    for point in points:
        if not endpoints:
            snapped.append(tuple(point))
            continue
        nearest = min(endpoints, key=lambda ep: math.hypot(ep[0] - point[0], ep[1] - point[1]))
        if math.hypot(nearest[0] - point[0], nearest[1] - point[1]) <= tolerance:
            snapped.append(nearest)
        else:
            snapped.append(tuple(point))
    return snapped


def points_to_segments(points: Sequence[Point], min_length: float = 1.0) -> list[Segment]:
    segments: list[Segment] = []
    for a, b in zip(points[:-1], points[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_length:
            segments.append((tuple(a), tuple(b)))
    return segments


def trace_stroke_to_structural_segments(
    stroke_points: Sequence[Point],
    material_type: str = "drywall",
    group_id: str = "manual:default",
    existing_segments: Sequence[dict[str, Any] | Any] = (),
    rdp_epsilon: float = 6.0,
    angle_tolerance_deg: float = 5.0,
    endpoint_snap_tolerance: float = 8.0,
    pixel_to_m: float = 1.0,
) -> list[dict[str, Any]]:
    """Clean a freehand stroke and emit structural segment dicts.

    Coordinates are output as meters using pixel_to_m. If your frontend canvas is
    already in model meters, use pixel_to_m=1.
    """
    if material_type not in MATERIAL_ATTENUATION_DB:
        raise ValueError(f"Unknown material_type: {material_type}")
    if len(stroke_points) < 2:
        return []

    simplified = ramer_douglas_peucker([(float(x), float(y)) for x, y in stroke_points], rdp_epsilon)
    axis_snapped = snap_axis_aligned_segments(simplified, angle_tolerance_deg)
    endpoint_snapped = snap_points_to_existing_endpoints(axis_snapped, existing_segments, endpoint_snap_tolerance)
    segments = points_to_segments(endpoint_snapped, min_length=max(1.0, rdp_epsilon * 0.5))

    output: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(segments, 1):
        seg = make_geometry_segment(
            start_xy=(round(start[0] * pixel_to_m, 6), round(start[1] * pixel_to_m, 6)),
            end_xy=(round(end[0] * pixel_to_m, 6), round(end[1] * pixel_to_m, 6)),
            layer_name="manual_trace",
            group_id=group_id,
            material_type=material_type,
            source="manual_trace",
            source_entity="MANUAL_TRACE",
            requires_classification=False,
        ).to_dict()
        seg["trace_index"] = idx
        output.append(seg)
    return output
