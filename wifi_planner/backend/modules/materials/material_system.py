"""Grouped material and attenuation editing.

Phase 5 scope: every structural segment has a group_id. Updating a group changes
material_type and attenuation_db for every segment in that group in one action.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Sequence

from backend.modules.cad.dxf_parser import MATERIAL_ATTENUATION_DB


def group_segments_by_group_id(segments: Sequence[dict[str, Any] | Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, segment in enumerate(segments):
        group_id = segment.get("group_id") if isinstance(segment, dict) else getattr(segment, "group_id")
        groups[str(group_id)].append(idx)
    return dict(groups)


def update_group_material(
    segments: Sequence[dict[str, Any]],
    group_id: str,
    new_material_type: str,
) -> list[dict[str, Any]]:
    """Return a deep-copied segment list with one group's material updated."""
    if new_material_type not in MATERIAL_ATTENUATION_DB:
        raise ValueError(f"Unknown material type: {new_material_type}")
    updated = deepcopy(list(segments))
    attenuation = float(MATERIAL_ATTENUATION_DB[new_material_type])
    for segment in updated:
        if segment.get("group_id") == group_id:
            segment["material_type"] = new_material_type
            segment["attenuation_db"] = attenuation
            segment["requires_classification"] = False
    return updated


def material_group_summary(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = group_segments_by_group_id(segments)
    summary: list[dict[str, Any]] = []
    for group_id, indices in sorted(groups.items()):
        mats = sorted({segments[i].get("material_type", "unknown") for i in indices})
        summary.append({"group_id": group_id, "count": len(indices), "materials": mats})
    return summary
