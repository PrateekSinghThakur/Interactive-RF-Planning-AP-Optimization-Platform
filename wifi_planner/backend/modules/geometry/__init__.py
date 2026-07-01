"""Geometry-first core types for RF planning."""
from .segments import GeometrySegment, MATERIAL_ATTENUATION_DB, geometry_segments_to_dicts

__all__ = ["GeometrySegment", "MATERIAL_ATTENUATION_DB", "geometry_segments_to_dicts"]
