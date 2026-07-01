"""Scale extraction module.

Owns only metadata.scale_m_per_px, metadata.scale_confidence, metadata.scale_method.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

__all__ = ["apply_manual_scale", "estimate_from_reference_doors"]


def _touch_metadata(model: dict[str, Any], scale: float, confidence: float, method: str) -> dict[str, Any]:
    updated = deepcopy(model)
    updated["metadata"]["scale_m_per_px"] = float(scale)
    updated["metadata"]["scale_confidence"] = max(0.0, min(1.0, float(confidence)))
    updated["metadata"]["scale_method"] = method
    return updated


def apply_manual_scale(
    model: dict[str, Any], point_a_px: list[float], point_b_px: list[float], real_distance_m: float
) -> dict[str, Any]:
    px = math.dist(point_a_px, point_b_px)
    if px <= 0:
        raise ValueError("Manual scale points must be distinct")
    if real_distance_m <= 0:
        raise ValueError("Manual real-world distance must be positive")
    return _touch_metadata(model, real_distance_m / px, 1.0, "manual")


def estimate_from_reference_doors(model: dict[str, Any], nominal_door_width_m: float = 0.85) -> dict[str, Any]:
    doors = model.get("doors", [])
    widths = [float(d.get("width_m", 0)) for d in doors if float(d.get("width_m", 0) or 0) > 0]
    if not widths:
        return _touch_metadata(model, model["metadata"].get("scale_m_per_px", 1.0), 0.0, "reference_object")
    median_width_px = sorted(widths)[len(widths) // 2]
    scale = nominal_door_width_m / median_width_px
    confidence = 0.55 if len(widths) == 1 else 0.75
    return _touch_metadata(model, scale, confidence, "reference_object")
