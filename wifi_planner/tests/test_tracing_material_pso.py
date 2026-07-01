import numpy as np

from backend.modules.materials.material_system import group_segments_by_group_id, update_group_material
from backend.modules.optimization.pso import PSOParams, pso_optimize
from backend.modules.tracing.spline_trace import trace_stroke_to_structural_segments


def test_manual_trace_simplifies_and_snaps_axis_aligned():
    stroke = [(0, 0), (5, 0.2), (10, -0.1), (20, 0.3)]
    segments = trace_stroke_to_structural_segments(stroke, rdp_epsilon=1.0, angle_tolerance_deg=5.0)
    assert len(segments) == 1
    assert segments[0]["start_xy"] == (0.0, 0.0)
    assert abs(segments[0]["end_xy"][1]) < 1e-9
    assert segments[0]["material_type"] == "drywall"
    assert segments[0]["attenuation_db"] == 4


def test_material_group_update():
    segments = [
        {"group_id": "layer:A", "material_type": "drywall", "attenuation_db": 4},
        {"group_id": "layer:A", "material_type": "drywall", "attenuation_db": 4},
        {"group_id": "layer:B", "material_type": "glass", "attenuation_db": 3},
    ]
    groups = group_segments_by_group_id(segments)
    assert groups == {"layer:A": [0, 1], "layer:B": [2]}
    updated = update_group_material(segments, "layer:A", "concrete")
    assert updated[0]["attenuation_db"] == 12
    assert updated[1]["material_type"] == "concrete"
    assert updated[2]["material_type"] == "glass"


def test_generic_pso_optimizes_simple_quadratic():
    target = np.array([2.0, -1.0])

    def fitness(x):
        return -float(np.sum((x - target) ** 2))

    result = pso_optimize(
        fitness,
        bounds=[(-5, 5), (-5, 5)],
        params=PSOParams(particles=20, iterations=40, seed=3, plateau_iterations=12),
    )
    assert np.linalg.norm(result.best_position - target) < 0.35
    assert result.best_fitness > -0.15
