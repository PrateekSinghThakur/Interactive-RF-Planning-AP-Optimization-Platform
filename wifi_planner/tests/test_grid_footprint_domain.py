from backend.modules.grid_builder.builder import build_grid


def base_model():
    return {
        "schema_version": "0.1.0",
        "metadata": {
            "source_image_ref": "unit-test",
            "scale_m_per_px": 1.0,
            "scale_confidence": 1.0,
            "scale_method": "manual",
            "floor_dimensions_m": {"width": 10, "height": 10},
            "created_at": "2026-06-23T00:00:00Z",
            "last_modified_at": "2026-06-23T00:00:00Z",
        },
        "walls": [],
        "rooms": [],
        "doors": [],
        "windows": [],
        "grid": {"resolution_m": 1, "origin_m": [0, 0], "cols": 10, "rows": 10, "floor_id": "floor_1", "cells": []},
        "access_points": [],
        "building_footprints": [
            {
                "id": "fp_1",
                "boundary_polygon_m": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "confidence": 1.0,
                "user_edited": False,
            }
        ],
        "analysis_regions": [],
    }


def test_grid_domain_comes_from_building_footprint_not_walls():
    model = base_model()
    gridded = build_grid(model, resolution_m=1)
    cells = gridded["grid"]["cells"]

    assert len(cells) == 100
    assert sum(1 for c in cells if c["type"] == "open") == 100
    assert sum(1 for c in cells if c["placeable"]) == 100
    assert sum(1 for c in cells if c["type"] == "outside") == 0


def test_walls_are_obstacles_not_grid_domain():
    model = base_model()
    model["walls"] = [
        {
            "id": "w_1",
            "start_m": [5, 0],
            "end_m": [5, 10],
            "thickness_m": 0.2,
            "material": "concrete",
            "attenuation_db": 12,
            "confidence": 1.0,
            "user_edited": False,
        }
    ]

    gridded = build_grid(model, resolution_m=1)
    cells = gridded["grid"]["cells"]

    assert len(cells) == 100
    assert sum(1 for c in cells if c["type"] == "outside") == 0
    assert sum(1 for c in cells if c["type"] == "wall") > 0
    assert sum(1 for c in cells if c["type"] in {"open", "wall"}) == 100
    # V2 rule: wall detections should not shrink the AP/simulation domain.
    assert sum(1 for c in cells if c["placeable"]) == 100
