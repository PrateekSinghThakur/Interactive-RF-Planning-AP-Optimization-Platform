import numpy as np

from backend.modules.cad.dxf_parser import StructuralSegment
from backend.modules.rf.speed_map import rasterize_segments_to_speed_map


def test_speed_map_rasterizes_attenuating_segment():
    segments = [
        StructuralSegment(
            start_xy=(1.0, 0.0),
            end_xy=(1.0, 3.0),
            layer_name="A-WALL-CONC",
            group_id="layer:A-WALL-CONC",
            material_type="concrete",
            attenuation_db=12,
            requires_classification=False,
            source_entity="LINE",
        )
    ]
    result = rasterize_segments_to_speed_map(
        segments,
        {"min_x": 0, "min_y": 0, "max_x": 4, "max_y": 3, "width": 4, "height": 3},
        resolution_cells_per_m=10,
    )

    assert result.speed_map.shape == (31, 41)
    assert np.min(result.speed_map) < 1.0
    assert np.max(result.attenuation_map_db) == 12
    row, col = result.xy_m_to_row_col(1.0, 1.5)
    assert result.speed_map[row, col] < 1.0


def test_unclassified_segments_do_not_become_obstacles():
    segments = [
        {
            "start_xy": (0.0, 0.0),
            "end_xy": (3.0, 0.0),
            "attenuation_db": 0,
        }
    ]
    result = rasterize_segments_to_speed_map(
        segments,
        {"min_x": 0, "min_y": 0, "max_x": 3, "max_y": 2, "width": 3, "height": 2},
        resolution_cells_per_m=5,
    )
    assert np.all(result.speed_map == 1.0)
    assert np.all(result.attenuation_map_db == 0.0)
