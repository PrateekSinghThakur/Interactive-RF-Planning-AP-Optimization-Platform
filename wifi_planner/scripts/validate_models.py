#!/usr/bin/env python3
"""Validate Building Model JSON files against the canonical schema plus cross-field invariants.

This script is intentionally not a product module. It is the schema gate used before
implementation work and later in CI for module input/output fixtures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_cross_field(model: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []

    grid = model.get("grid", {})
    rows = grid.get("rows")
    cols = grid.get("cols")
    cells = grid.get("cells", [])
    if isinstance(rows, int) and isinstance(cols, int) and isinstance(cells, list):
        expected = rows * cols
        if len(cells) != expected:
            errors.append(f"{path}: grid.cells length {len(cells)} != rows*cols {expected}")

    wall_ids = {w.get("id") for w in model.get("walls", []) if isinstance(w, dict)}
    room_ids = {r.get("id") for r in model.get("rooms", []) if isinstance(r, dict)}

    for door in model.get("doors", []):
        wall_id = door.get("wall_id") if isinstance(door, dict) else None
        if wall_id not in wall_ids:
            errors.append(f"{path}: door {door.get('id')} references missing wall_id {wall_id}")

    for window in model.get("windows", []):
        wall_id = window.get("wall_id") if isinstance(window, dict) else None
        if wall_id not in wall_ids:
            errors.append(f"{path}: window {window.get('id')} references missing wall_id {wall_id}")

    for room in model.get("rooms", []):
        if not isinstance(room, dict):
            continue
        rid = room.get("id")
        for adj in room.get("adjacent_room_ids", []):
            if adj not in room_ids:
                errors.append(f"{path}: room {rid} references missing adjacent_room_id {adj}")
            if adj == rid:
                errors.append(f"{path}: room {rid} cannot be adjacent to itself")

    for idx, cell in enumerate(cells if isinstance(cells, list) else []):
        if not isinstance(cell, dict):
            continue
        room_id = cell.get("room_id")
        if room_id is not None and room_id not in room_ids:
            errors.append(f"{path}: grid cell {idx} references missing room_id {room_id}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="building_model.schema.json", type=Path)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("validation_models")])
    args = parser.parse_args()

    schema = load_json(args.schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)

    if not files:
        print("No JSON files to validate", file=sys.stderr)
        return 2

    failures = 0
    for path in files:
        model = load_json(path)
        schema_errors = sorted(validator.iter_errors(model), key=lambda e: list(e.path))
        cross_errors = validate_cross_field(model, path)
        if schema_errors or cross_errors:
            failures += 1
            print(f"FAIL {path}")
            for error in schema_errors:
                loc = "/".join(str(p) for p in error.absolute_path) or "<root>"
                print(f"  schema {loc}: {error.message}")
            for error in cross_errors:
                print(f"  invariant {error}")
        else:
            print(f"PASS {path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
