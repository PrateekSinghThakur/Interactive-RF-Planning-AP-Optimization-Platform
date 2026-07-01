from pathlib import Path

import ezdxf

from backend.modules.cad.dxf_parser import parse_dxf


def create_sample_dxf(path: Path):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # millimeters
    msp = doc.modelspace()
    msp.add_line((0, 0), (5000, 0), dxfattribs={"layer": "A-WALL-CONC"})
    msp.add_line((5000, 0), (5000, 3000), dxfattribs={"layer": "A-GLAZ"})
    msp.add_lwpolyline([(0, 0), (0, 3000), (5000, 3000)], dxfattribs={"layer": "UNKNOWN-FURN"})
    doc.saveas(path)


def test_parse_dxf_layers_materials_and_units(tmp_path):
    dxf_path = tmp_path / "sample.dxf"
    create_sample_dxf(dxf_path)

    result = parse_dxf(dxf_path)

    assert result.unit_scale_to_m == 0.001
    assert result.bbox_m["width"] == 5.0
    assert result.bbox_m["height"] == 3.0
    assert result.layer_materials["A-WALL-CONC"] == "concrete"
    assert result.layer_materials["A-GLAZ"] == "glass"
    assert result.layer_materials["UNKNOWN-FURN"] == "unclassified"
    assert result.unclassified_layers == ["UNKNOWN-FURN"]
    assert len(result.segments) == 4  # 2 LINE + 2 LWPOLYLINE edges

    wall = next(seg for seg in result.segments if seg.layer_name == "A-WALL-CONC")
    assert wall.start_xy == (0.0, 0.0)
    assert wall.end_xy == (5.0, 0.0)
    assert wall.group_id == "layer:A-WALL-CONC"
    assert wall.material_type == "concrete"
    assert wall.attenuation_db == 12
    assert wall.requires_classification is False


def test_parse_dxf_explicit_material_mapping(tmp_path):
    dxf_path = tmp_path / "sample.dxf"
    create_sample_dxf(dxf_path)

    result = parse_dxf(dxf_path, layer_material_map={"UNKNOWN-FURN": "wood"})

    assert result.layer_materials["UNKNOWN-FURN"] == "wood"
    assert result.unclassified_layers == []
    assert all(not segment.requires_classification for segment in result.segments)
    assert any(segment.material_type == "wood" for segment in result.segments)
