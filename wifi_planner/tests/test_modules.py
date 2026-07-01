import json
from pathlib import Path

from backend.modules.grid_builder.builder import build_grid
from backend.modules.propagation.engine import compute_coverage
from backend.modules.placement.recommender import recommend_access_points
from backend.common.schema import validate_building_model, assert_grid_invariants

ROOT = Path(__file__).resolve().parents[1]


def load_wifi_model():
    return json.loads((ROOT / "validation_models" / "wifi_validation_model.json").read_text())


def test_grid_builder_outputs_valid_building_model():
    model = load_wifi_model()
    gridded = build_grid(model, resolution_m=1.0)
    validate_building_model(gridded)
    assert_grid_invariants(gridded)
    assert len(gridded["grid"]["cells"]) == gridded["grid"]["rows"] * gridded["grid"]["cols"]


def test_propagation_returns_one_value_per_cell():
    model = load_wifi_model()
    coverage = compute_coverage(model, preview=True)
    assert len(coverage["coverage_dbm"]) == model["grid"]["rows"] * model["grid"]["cols"]
    assert max(coverage["coverage_dbm"]) > -90


def test_placement_updates_only_access_points_shape():
    model = load_wifi_model()
    model["access_points"] = []

    def scorer(m, aps, preview):
        return compute_coverage(m, aps, preview=True)

    recommended = recommend_access_points(model, scorer, max_ap_count=2, candidate_stride=8)
    assert 1 <= len(recommended["access_points"]) <= 2
    validate_building_model(recommended)
