"""Standalone DXF parser for RF Coverage Engine Phase 2.

Phase 2 scope only:
- Read DXF LINE, LWPOLYLINE, and POLYLINE entities from modelspace.
- Group geometry by DXF layer.
- Guess material from layer names, but never silently assume unmatched layers are walls.
- Normalize coordinates to meters.
- Return structural segment dictionaries compatible with later rasterizer/FMM modules.

No UI, no FMM integration, no PSO, no raster floorplan detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import ezdxf  # type: ignore
except Exception:  # pragma: no cover - handled by _require_ezdxf
    ezdxf = None


from backend.modules.geometry.segments import MATERIAL_ATTENUATION_DB, GeometrySegment, make_geometry_segment


# Backward-compatible name for tests/imports. The core contract is GeometrySegment.
StructuralSegment = GeometrySegment


@dataclass(frozen=True)
class DxfParseResult:
    segments: list[StructuralSegment]
    layers: list[str]
    layer_materials: dict[str, str]
    unclassified_layers: list[str]
    bbox_m: dict[str, float]
    unit_scale_to_m: float
    unit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "layers": self.layers,
            "layer_materials": self.layer_materials,
            "unclassified_layers": self.unclassified_layers,
            "bbox_m": self.bbox_m,
            "unit_scale_to_m": self.unit_scale_to_m,
            "unit_reason": self.unit_reason,
        }


def _require_ezdxf() -> Any:
    if ezdxf is None:
        raise ImportError("ezdxf is required. Install with: python -m pip install ezdxf")
    return ezdxf


def guess_material_from_layer(layer_name: str) -> str | None:
    """Keyword-based material guess from CAD layer name.

    Important: unmatched layers return None. The parser must not silently treat
    unknown layers as walls/concrete because that recreates the old false-wall
    problem in vector form.
    """
    name = layer_name.upper().replace("_", "-")

    if any(token in name for token in ["GLAZ", "GLASS", "WINDOW", "WIND", "A-GLAZ"]):
        return "glass"
    if "BRICK" in name or "MASON" in name:
        return "brick"
    if "RCC" in name or "REINF" in name or "RC-" in name:
        return "reinforced_concrete"
    if "CONC" in name or "CONCRETE" in name:
        return "concrete"
    if "METAL" in name or "STEEL" in name:
        return "metal"
    if "DOOR" in name and "MET" in name:
        return "door_metal"
    if "DOOR" in name:
        return "door_wood"
    if "WOOD" in name or "PLY" in name:
        return "wood"
    if "DRYWALL" in name or "GYPS" in name or "GYP" in name:
        return "drywall"
    # Generic wall layers are likely structural, but still only a material guess.
    if "WALL" in name or "A-WALL" in name or "PARTITION" in name:
        return "concrete"
    return None


def _entity_points(entity: Any) -> list[tuple[float, float]]:
    etype = entity.dxftype()
    if etype == "LINE":
        s = entity.dxf.start
        e = entity.dxf.end
        return [(float(s.x), float(s.y)), (float(e.x), float(e.y))]
    if etype == "LWPOLYLINE":
        return [(float(x), float(y)) for x, y, *_ in entity.get_points()]
    if etype == "POLYLINE":
        return [(float(vertex.dxf.location.x), float(vertex.dxf.location.y)) for vertex in entity.vertices]
    return []


def _segments_from_points(points: list[tuple[float, float]], closed: bool) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    if len(points) < 2:
        return []
    pairs = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if closed and len(points) > 2:
        pairs.append((points[-1], points[0]))
    return pairs


def _dxf_units_scale(doc: Any) -> tuple[float | None, str | None]:
    """Return drawing-unit to meter scale from DXF $INSUNITS when available."""
    units = int(doc.header.get("$INSUNITS", 0) or 0)
    # Common AutoCAD INSUNITS values.
    mapping = {
        1: (0.0254, "DXF INSUNITS=inches"),
        2: (0.3048, "DXF INSUNITS=feet"),
        4: (0.001, "DXF INSUNITS=millimeters"),
        5: (0.01, "DXF INSUNITS=centimeters"),
        6: (1.0, "DXF INSUNITS=meters"),
    }
    return mapping.get(units, (None, None))


def _heuristic_units_scale(raw_bbox: dict[str, float]) -> tuple[float, str]:
    """Heuristic fallback when DXF unit metadata is absent.

    Architectural DXFs in India/AutoCAD workflows are often in millimeters. If a
    building bbox is thousands to hundreds of thousands of units wide, millimeters
    are likely. If it is tens/hundreds, meters are likely.
    """
    width = raw_bbox["width"]
    height = raw_bbox["height"]
    max_dim = max(width, height)
    if max_dim > 1000:
        return 0.001, "heuristic: bbox dimension > 1000, treating drawing units as millimeters"
    if max_dim > 200:
        return 0.01, "heuristic: bbox dimension > 200, treating drawing units as centimeters"
    return 1.0, "heuristic: bbox dimensions look like meters"


def _bbox(points: list[tuple[float, float]]) -> dict[str, float]:
    if not points:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0, "width": 0, "height": 0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def parse_dxf(
    path: str | Path,
    layer_material_map: dict[str, str] | None = None,
    include_unclassified: bool = True,
) -> DxfParseResult:
    """Parse a DXF into normalized structural segments.

    Parameters
    ----------
    path:
        DXF file path.
    layer_material_map:
        Optional explicit user mapping: layer_name -> material_type. This takes
        precedence over keyword guesses.
    include_unclassified:
        If True, unmatched layers are returned as material_type='unclassified'
        with requires_classification=True. If False, they are omitted from the
        segment list but still reported in unclassified_layers.

    Returns
    -------
    DxfParseResult with segments in meters and layer/material metadata.
    """
    _ezdxf = _require_ezdxf()
    doc = _ezdxf.readfile(str(path))
    msp = doc.modelspace()
    explicit_map = layer_material_map or {}

    raw_entities: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = []
    all_points: list[tuple[float, float]] = []
    layer_names: set[str] = set()

    for entity in msp:
        etype = entity.dxftype()
        if etype not in {"LINE", "LWPOLYLINE", "POLYLINE"}:
            continue
        layer = str(entity.dxf.layer)
        layer_names.add(layer)
        points = _entity_points(entity)
        all_points.extend(points)
        closed = bool(getattr(entity, "closed", False)) if etype != "LINE" else False
        for start, end in _segments_from_points(points, closed):
            if start == end:
                continue
            raw_entities.append((etype, layer, start, end))

    raw_bbox = _bbox(all_points)
    unit_scale, unit_reason = _dxf_units_scale(doc)
    if unit_scale is None:
        unit_scale, unit_reason = _heuristic_units_scale(raw_bbox)

    min_x = raw_bbox["min_x"]
    min_y = raw_bbox["min_y"]

    def normalize(point: tuple[float, float]) -> tuple[float, float]:
        # Normalize origin to bbox minimum and convert to meters.
        return (round((point[0] - min_x) * unit_scale, 6), round((point[1] - min_y) * unit_scale, 6))

    sorted_layers = sorted(layer_names)
    layer_materials: dict[str, str] = {}
    unclassified_layers: list[str] = []
    for layer in sorted_layers:
        material = explicit_map.get(layer) or guess_material_from_layer(layer)
        if material is None:
            material = "unclassified"
            unclassified_layers.append(layer)
        if material not in MATERIAL_ATTENUATION_DB:
            raise ValueError(f"Unknown material '{material}' for layer '{layer}'")
        layer_materials[layer] = material

    segments: list[StructuralSegment] = []
    for etype, layer, start, end in raw_entities:
        material = layer_materials[layer]
        requires = material == "unclassified"
        if requires and not include_unclassified:
            continue
        segments.append(
            make_geometry_segment(
                start_xy=normalize(start),
                end_xy=normalize(end),
                layer_name=layer,
                group_id=f"layer:{layer}",
                material_type=material,
                source="dxf",
                source_entity=etype,
                requires_classification=requires,
            )
        )

    bbox_m = {
        "min_x": 0.0,
        "min_y": 0.0,
        "max_x": round(raw_bbox["width"] * unit_scale, 6),
        "max_y": round(raw_bbox["height"] * unit_scale, 6),
        "width": round(raw_bbox["width"] * unit_scale, 6),
        "height": round(raw_bbox["height"] * unit_scale, 6),
    }

    return DxfParseResult(
        segments=segments,
        layers=sorted_layers,
        layer_materials=layer_materials,
        unclassified_layers=sorted(unclassified_layers),
        bbox_m=bbox_m,
        unit_scale_to_m=float(unit_scale),
        unit_reason=str(unit_reason),
    )


def print_layer_report(result: DxfParseResult) -> None:
    """Human-readable layer inspection report for CLI/manual workflow."""
    print("DXF layers:")
    for layer in result.layers:
        material = result.layer_materials[layer]
        suffix = "  <-- needs classification" if material == "unclassified" else ""
        print(f"- {layer}: {material}{suffix}")
    print(f"Bounding box (m): {result.bbox_m}")
    print(f"Units: scale={result.unit_scale_to_m} ({result.unit_reason})")
