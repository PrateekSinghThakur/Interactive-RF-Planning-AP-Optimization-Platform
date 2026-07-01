"""Tiny helpers for app-layer composition of full Building Model JSON artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_building_model(source_image_ref: str = "", width_m: float = 1.0, height_m: float = 1.0) -> dict[str, Any]:
    ts = now_iso()
    return {
        "schema_version": "0.1.0",
        "metadata": {
            "source_image_ref": source_image_ref,
            "scale_m_per_px": 1.0,
            "scale_confidence": 0.0,
            "scale_method": "manual",
            "floor_dimensions_m": {"width": width_m, "height": height_m},
            "created_at": ts,
            "last_modified_at": ts,
        },
        "walls": [],
        "rooms": [],
        "doors": [],
        "windows": [],
        "grid": {
            "resolution_m": 1.0,
            "origin_m": [0, 0],
            "cols": max(1, int(round(width_m))),
            "rows": max(1, int(round(height_m))),
            "floor_id": "floor_1",
            "cells": [],
        },
        "access_points": [],
        "building_footprints": [],
        "analysis_regions": [],
    }
