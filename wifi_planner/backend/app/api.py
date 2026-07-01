"""FastAPI app layer for the Wi-Fi v1 application.

The routes orchestrate modules through Building Model JSON artifacts. Domain modules
remain independently replaceable behind their public functions.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import base64
import io

import numpy as np

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.common.schema import load_schema, validate_building_model
from backend.common.model_factory import empty_building_model
from backend.modules.detection.detector import detect
from backend.modules.grid_builder.builder import build_grid
from backend.modules.placement.recommender import recommend_access_points
from backend.modules.placement.pso_grid import recommend_access_points_pso
from backend.modules.propagation.engine import compute_coverage
from backend.app.storage import load_project, save_project

app = FastAPI(title="Intelligent Floorplan Understanding and Wi-Fi Planner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


DEMO_LONG_SIDE_M = 42.0


def _scale_coord(coord: list[float], factor: float) -> list[float]:
    return [round(float(coord[0]) * factor, 4), round(float(coord[1]) * factor, 4)]


def _normalize_detection_units(partial: dict[str, Any], width_px: int, height_px: int) -> tuple[dict[str, Any], float, float, float]:
    """Convert detector pixel-space output into manageable provisional meters.

    This keeps grid building and AP recommendation fast even before a user enters
    a precise real-world scale. The scale module can later overwrite metadata.
    """
    normalized = deepcopy(partial)
    long_side = max(width_px, height_px, 1)
    factor = DEMO_LONG_SIDE_M / long_side
    width_m = round(width_px * factor, 3)
    height_m = round(height_px * factor, 3)

    for wall in normalized.get("walls", []):
        wall["start_m"] = _scale_coord(wall["start_m"], factor)
        wall["end_m"] = _scale_coord(wall["end_m"], factor)
        wall["thickness_m"] = max(0.08, round(float(wall.get("thickness_m", 0.2)) * factor, 4))
    for room in normalized.get("rooms", []):
        room["boundary_polygon_m"] = [_scale_coord(point, factor) for point in room["boundary_polygon_m"]]
        room["centroid_m"] = _scale_coord(room["centroid_m"], factor)
        room["area_m2"] = round(float(room.get("area_m2", 0)) * factor * factor, 3)
    for door in normalized.get("doors", []):
        door["position_m"] = _scale_coord(door["position_m"], factor)
        door["width_m"] = max(0.7, round(float(door.get("width_m", 0.9)) * factor, 3))
    for window in normalized.get("windows", []):
        window["position_m"] = _scale_coord(window["position_m"], factor)
        window["width_m"] = max(0.7, round(float(window.get("width_m", 1.2)) * factor, 3))
    for footprint in normalized.get("building_footprints", []):
        footprint["boundary_polygon_m"] = [_scale_coord(point, factor) for point in footprint["boundary_polygon_m"]]
    normalized["detection"]["provisional_units"] = {
        "long_side_m": DEMO_LONG_SIDE_M,
        "px_to_model_m": factor,
        "note": "Coordinates are normalized for fast demo planning until manual scale calibration.",
    }
    return normalized, width_m, height_m, factor


@app.get("/schema")
def schema() -> dict[str, Any]:
    return load_schema()


@app.post("/validate")
def validate(model: dict[str, Any]) -> dict[str, Any]:
    validate_building_model(model)
    return {"valid": True}


@app.post("/detect")
async def detect_floorplan(file: UploadFile = File(...)) -> dict[str, Any]:
    result_px = detect(await file.read())
    width_px = result_px["detection"]["image_size_px"]["width"]
    height_px = result_px["detection"]["image_size_px"]["height"]
    result, width_m, height_m, px_to_m = _normalize_detection_units(result_px, width_px, height_px)

    # App-layer composition: detection stays structural-only; model uses provisional
    # normalized meters until the scale module/user calibration confirms scale.
    model = empty_building_model(file.filename or "upload", width_m, height_m)
    model["metadata"]["scale_m_per_px"] = px_to_m
    model["metadata"]["scale_confidence"] = 0.2
    model["metadata"]["scale_method"] = "manual"
    model["grid"]["resolution_m"] = 0.5
    model["grid"]["cols"] = max(1, int(round(width_m / 0.5)))
    model["grid"]["rows"] = max(1, int(round(height_m / 0.5)))
    model["walls"] = result["walls"]
    model["rooms"] = result["rooms"] or [
        {
            "id": "r_uploaded_area",
            "boundary_polygon_m": [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
            "area_m2": round(width_m * height_m, 3),
            "centroid_m": [round(width_m / 2, 3), round(height_m / 2, 3)],
            "label": "unknown",
            "occupancy_level": "medium",
            "adjacent_room_ids": [],
            "confidence": 0.25,
            "user_edited": False,
        }
    ]
    model["doors"] = result["doors"]
    model["windows"] = result["windows"]
    model["building_footprints"] = result.get("building_footprints") or [
        {
            "id": "fp_uploaded_area",
            "boundary_polygon_m": [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
            "confidence": 0.2,
            "user_edited": False,
        }
    ]

    # Important V2 integration point: the detector module still owns only
    # structural extraction, but this app endpoint returns a model with an initial
    # footprint-based grid so the frontend/heatmap cannot accidentally behave as
    # if walls or room polygons define the simulation domain.
    model = build_grid(model, resolution_m=0.5)
    return {"partial": result, "model": model, "grid_stats": _grid_stats(model)}


def _grid_stats(model: dict[str, Any]) -> dict[str, Any]:
    cells = model.get("grid", {}).get("cells", [])
    by_type: dict[str, int] = {}
    for cell in cells:
        by_type[cell.get("type", "unknown")] = by_type.get(cell.get("type", "unknown"), 0) + 1
    return {
        "total_cells": len(cells),
        "interior_domain_cells": sum(1 for cell in cells if cell.get("type") != "outside"),
        "placeable_cells": sum(1 for cell in cells if cell.get("placeable")),
        "outside_cells": by_type.get("outside", 0),
        "wall_cells": by_type.get("wall", 0),
        "door_cells": by_type.get("door", 0),
        "window_cells": by_type.get("window", 0),
        "open_cells": by_type.get("open", 0),
        "footprint_count": len(model.get("building_footprints") or []),
    }


@app.post("/grid")
def grid(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload["model"]
    resolution = float(payload.get("resolution_m", 0.5))
    gridded = build_grid(model, resolution)
    return {"model": gridded, "grid_stats": _grid_stats(gridded)}


@app.post("/coverage")
def coverage(payload: dict[str, Any]) -> dict[str, Any]:
    return compute_coverage(payload["model"], payload.get("access_points"), bool(payload.get("preview", False)))


@app.post("/placement")
def placement(payload: dict[str, Any]) -> dict[str, Any]:
    def scorer(model: dict[str, Any], aps: list[dict[str, Any]], preview: bool) -> dict[str, Any]:
        return compute_coverage(model, aps, preview)

    model = recommend_access_points(
        payload["model"],
        scorer,
        coverage_target=float(payload.get("coverage_target", 0.9)),
        max_ap_count=int(payload.get("max_ap_count", 4)),
        min_signal_dbm=float(payload.get("min_signal_dbm", -67)),
    )
    return {"model": model, "coverage": compute_coverage(model, preview=True), "grid_stats": _grid_stats(model)}


@app.post("/placement-pso")
def placement_pso(payload: dict[str, Any]) -> dict[str, Any]:
    """Backend PSO AP optimization for the current raster/grid web app."""
    def scorer(model: dict[str, Any], aps: list[dict[str, Any]], preview: bool) -> dict[str, Any]:
        return compute_coverage(model, aps, preview)

    model = recommend_access_points_pso(
        payload["model"],
        scorer,
        min_signal_dbm=float(payload.get("min_signal_dbm", -67)),
        max_ap_count=int(payload.get("max_ap_count", 3)),
        particles=int(payload.get("particles", 12)),
        iterations=int(payload.get("iterations", 12)),
    )
    return {"model": model, "coverage": compute_coverage(model, preview=True), "grid_stats": _grid_stats(model)}


@app.post("/rf/dxf-heatmap")
async def dxf_heatmap(
    file: UploadFile = File(...),
    resolution: float = Form(10.0),
    optimize: bool = Form(True),
    num_aps: int = Form(2),
    ap1_x: float = Form(15.0),
    ap1_y: float = Form(12.0),
    ap2_x: float = Form(35.0),
    ap2_y: float = Form(20.0),
) -> dict[str, Any]:
    """DXF -> GeometrySegment[] -> FMM heatmap endpoint."""
    from backend.modules.cad.dxf_parser import parse_dxf
    from backend.modules.geometry.segments import make_geometry_segment
    from backend.modules.rf.geometry_pipeline import run_geometry_rf_pipeline

    content = await file.read()
    with NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        parsed = parse_dxf(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    segments = [seg for seg in parsed.segments if not seg.requires_classification]
    fallback_used = False
    if not segments:
        fallback_used = True
        segments = [
            make_geometry_segment(
                start_xy=seg.start_xy,
                end_xy=seg.end_xy,
                material_type="concrete",
                layer_name=seg.layer_name,
                group_id=seg.group_id,
                source="dxf",
                source_entity=seg.source_entity,
                requires_classification=False,
            )
            for seg in parsed.segments
        ]

    if not segments:
        return {"ok": False, "error": "No line/polyline geometry found in DXF", "layers": parsed.layers, "unclassified_layers": parsed.unclassified_layers}

    default_aps = [(float(ap1_x), float(ap1_y)), (float(ap2_x), float(ap2_y))][: max(1, min(4, int(num_aps)))]
    result = run_geometry_rf_pipeline(segments, parsed.bbox_m, resolution=float(resolution), default_aps=default_aps, optimize=optimize, num_aps=max(1, min(4, int(num_aps))))

    return {
        "ok": True,
        "layers": parsed.layers,
        "layer_materials": parsed.layer_materials,
        "unclassified_layers": parsed.unclassified_layers,
        "fallback_used": fallback_used,
        "bbox_m": parsed.bbox_m,
        "unit_reason": parsed.unit_reason,
        "segment_count": result["segment_count"],
        "classified_segment_count": result["active_obstacle_count"],
        "speed_map_shape": result["speed_map_shape"],
        "assigned_wall": sum(1 for s in segments if s.segment_type == "wall"),
        "assigned_glass": sum(1 for s in segments if s.segment_type in {"glass", "window"}),
        "best_global_positions_px": None,
        "best_global_score_pixels": result["pso_fitness"],
        "rssi_min_dbm": result["rssi_min_dbm"],
        "rssi_max_dbm": result["rssi_max_dbm"],
        "heatmap_image_data_url": result["heatmap_image_data_url"],
        "direct_heatmap_image_data_url": result["direct_heatmap_image_data_url"],
        "optimized_heatmap_image_data_url": result["optimized_heatmap_image_data_url"],
    }


@app.post("/projects")
def save_project_route(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload["model"]
    validate_building_model(model)
    project_id = save_project(model, payload.get("name", "Untitled project"), payload.get("project_id"))
    return {"project_id": project_id}


@app.get("/projects/{project_id}")
def load_project_route(project_id: str) -> dict[str, Any]:
    return {"model": load_project(project_id)}


@app.post("/report", response_class=PlainTextResponse)
def report(payload: dict[str, Any]) -> str:
    model = payload["model"]
    coverage = payload.get("coverage") or compute_coverage(model)
    min_signal_dbm = float(payload.get("min_signal_dbm", -67))
    target = [i for i, c in enumerate(model["grid"]["cells"]) if c.get("placeable")]
    strong = sum(1 for i in target if coverage["coverage_dbm"][i] >= min_signal_dbm)
    pct = 100 * strong / max(1, len(target))
    lines = [
        "Wi-Fi Design Report",
        "====================",
        f"Source image: {model['metadata']['source_image_ref']}",
        f"Floor size: {model['metadata']['floor_dimensions_m']['width']}m x {model['metadata']['floor_dimensions_m']['height']}m",
        f"Grid: {model['grid']['cols']} x {model['grid']['rows']} @ {model['grid']['resolution_m']}m",
        f"Access points: {len(model.get('access_points', []))}",
        f"Cells >= {min_signal_dbm:g} dBm: {pct:.1f}%",
        "",
        "AP placements:",
    ]
    for ap in model.get("access_points", []):
        lines.append(f"- {ap['id']}: {ap['position_m']} m, {ap['tx_power_dbm']} dBm @ {ap['freq_ghz']} GHz ({ap['source']})")
    return "\n".join(lines) + "\n"
