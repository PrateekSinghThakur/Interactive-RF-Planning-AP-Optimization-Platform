"""Schema utilities shared by orchestration code and tests.

This package is not a domain module. It only loads/validates the canonical Building
Model schema so each module can be checked at its boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "building_model.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_building_model(model: dict[str, Any]) -> None:
    validator = Draft202012Validator(load_schema(), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(model), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        loc = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ValueError(f"Building Model schema validation failed at {loc}: {first.message}")


def assert_grid_invariants(model: dict[str, Any]) -> None:
    grid = model["grid"]
    if len(grid["cells"]) != grid["rows"] * grid["cols"]:
        raise ValueError("grid.cells must be flattened row-major rows*cols length")
