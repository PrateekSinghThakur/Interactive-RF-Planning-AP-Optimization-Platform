"""Core geometry contract for the RF planning platform.

This is now the central invariant of the RF engine:

    every input source -> GeometrySegment[]

DXF parsing, JPEG/manual spline tracing, and any future AI detector should all
produce this same representation. Downstream modules (material assignment,
rasterization, FMM propagation, PSO, visualization) must not care where geometry
came from.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

SegmentType = Literal["wall", "glass", "door", "window", "obstacle", "ignore", "unclassified"]
SegmentSource = Literal["dxf", "manual_trace", "ai", "synthetic"]

MATERIAL_ATTENUATION_DB: dict[str, float] = {
    "drywall": 4,
    "wood": 5,
    "glass": 3,
    "brick": 9,
    "concrete": 12,
    "reinforced_concrete": 18,
    "metal": 25,
    "door_wood": 4,
    "door_metal": 6,
    "window_standard": 3,
    "unclassified": 0,
    "ignore": 0,
}

MATERIAL_TO_TYPE: dict[str, SegmentType] = {
    "glass": "glass",
    "window_standard": "window",
    "door_wood": "door",
    "door_metal": "door",
    "unclassified": "unclassified",
    "ignore": "ignore",
}


@dataclass(frozen=True)
class GeometrySegment:
    """One RF-relevant geometry segment.

    The solver only cares about geometry and attenuation/speed-map effect. The
    `segment_type`, `layer_name`, `group_id`, and `source` fields are primarily
    for UI, material editing, and traceability.

    Coordinates are meters in a local building coordinate system.
    """

    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    material_type: str
    attenuation_db: float
    segment_type: SegmentType = "wall"
    layer_name: str = ""
    group_id: str = "default"
    thickness_m: float = 0.15
    source: SegmentSource = "dxf"
    source_entity: str = "UNKNOWN"
    requires_classification: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_obstacle(self) -> bool:
        return self.attenuation_db > 0 and self.segment_type != "ignore"


def infer_segment_type(material_type: str, layer_name: str = "") -> SegmentType:
    if material_type in MATERIAL_TO_TYPE:
        return MATERIAL_TO_TYPE[material_type]
    lname = layer_name.upper()
    if "WINDOW" in lname or "GLAZ" in lname or "GLASS" in lname:
        return "glass"
    if "DOOR" in lname:
        return "door"
    return "wall"


def make_geometry_segment(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    material_type: str,
    layer_name: str = "",
    group_id: str | None = None,
    source: SegmentSource = "dxf",
    source_entity: str = "UNKNOWN",
    thickness_m: float = 0.15,
    requires_classification: bool | None = None,
) -> GeometrySegment:
    if material_type not in MATERIAL_ATTENUATION_DB:
        raise ValueError(f"Unknown material_type: {material_type}")
    req = material_type == "unclassified" if requires_classification is None else requires_classification
    return GeometrySegment(
        start_xy=(round(float(start_xy[0]), 6), round(float(start_xy[1]), 6)),
        end_xy=(round(float(end_xy[0]), 6), round(float(end_xy[1]), 6)),
        material_type=material_type,
        attenuation_db=float(MATERIAL_ATTENUATION_DB[material_type]),
        segment_type=infer_segment_type(material_type, layer_name),
        layer_name=layer_name,
        group_id=group_id or (f"layer:{layer_name}" if layer_name else "default"),
        thickness_m=float(thickness_m),
        source=source,
        source_entity=source_entity,
        requires_classification=bool(req),
    )


def geometry_segments_to_dicts(segments: list[GeometrySegment]) -> list[dict[str, Any]]:
    return [segment.to_dict() for segment in segments]


def coerce_geometry_segment(segment: GeometrySegment | dict[str, Any] | Any) -> GeometrySegment:
    """Convert legacy segment-like objects into GeometrySegment."""
    if isinstance(segment, GeometrySegment):
        return segment
    if isinstance(segment, dict):
        material = segment.get("material_type", "unclassified")
        return GeometrySegment(
            start_xy=tuple(segment["start_xy"]),
            end_xy=tuple(segment["end_xy"]),
            material_type=material,
            attenuation_db=float(segment.get("attenuation_db", MATERIAL_ATTENUATION_DB.get(material, 0))),
            segment_type=segment.get("segment_type") or infer_segment_type(material, segment.get("layer_name", "")),
            layer_name=segment.get("layer_name", ""),
            group_id=segment.get("group_id", "default"),
            thickness_m=float(segment.get("thickness_m", 0.15)),
            source=segment.get("source", "dxf"),
            source_entity=segment.get("source_entity", "UNKNOWN"),
            requires_classification=bool(segment.get("requires_classification", material == "unclassified")),
        )
    # Dataclass/legacy object support.
    material = getattr(segment, "material_type", "unclassified")
    return GeometrySegment(
        start_xy=tuple(getattr(segment, "start_xy")),
        end_xy=tuple(getattr(segment, "end_xy")),
        material_type=material,
        attenuation_db=float(getattr(segment, "attenuation_db", MATERIAL_ATTENUATION_DB.get(material, 0))),
        segment_type=getattr(segment, "segment_type", infer_segment_type(material, getattr(segment, "layer_name", ""))),
        layer_name=getattr(segment, "layer_name", ""),
        group_id=getattr(segment, "group_id", "default"),
        thickness_m=float(getattr(segment, "thickness_m", 0.15)),
        source=getattr(segment, "source", "dxf"),
        source_entity=getattr(segment, "source_entity", "UNKNOWN"),
        requires_classification=bool(getattr(segment, "requires_classification", material == "unclassified")),
    )
