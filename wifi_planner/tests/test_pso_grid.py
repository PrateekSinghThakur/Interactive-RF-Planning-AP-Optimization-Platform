from backend.modules.placement.pso_grid import recommend_access_points_pso


def tiny_model():
    cells = []
    for r in range(4):
        for c in range(4):
            cells.append({"center_m": [c + 0.5, r + 0.5], "type": "open", "attenuation_db": 0, "room_id": None, "placeable": True})
    return {
        "schema_version": "0.1.0",
        "metadata": {},
        "walls": [], "rooms": [], "doors": [], "windows": [],
        "grid": {"resolution_m": 1, "origin_m": [0, 0], "cols": 4, "rows": 4, "floor_id": "floor_1", "cells": cells},
        "access_points": [],
        "analysis_regions": [{"id": "roi", "label": "ROI", "boundary_polygon_m": [[0,0],[2,0],[2,2],[0,2]], "active": True, "user_edited": True}],
    }


def fake_scorer(model, aps, preview):
    vals = []
    ap = aps[0]
    for cell in model["grid"]["cells"]:
        dx = cell["center_m"][0] - ap["position_m"][0]
        dy = cell["center_m"][1] - ap["position_m"][1]
        vals.append(-50 - (dx*dx + dy*dy))
    return {"coverage_dbm": vals}


def test_pso_grid_respects_roi():
    result = recommend_access_points_pso(tiny_model(), fake_scorer, max_ap_count=1, particles=6, iterations=4)
    ap = result["access_points"][0]
    assert 0 <= ap["position_m"][0] <= 2
    assert 0 <= ap["position_m"][1] <= 2
