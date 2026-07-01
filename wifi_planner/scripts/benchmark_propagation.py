#!/usr/bin/env python3
"""Benchmark propagation on a Building Model fixture.

Use this during the week 5-6 performance gate and replace the fixture with the
largest tagged evaluation floor as the dataset matures.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.modules.propagation.engine import compute_coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="validation_models/wifi_validation_model.json")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text())
    # Warm-up includes Numba compilation; don't count it.
    compute_coverage(model, preview=args.preview)
    durations = []
    for _ in range(args.runs):
        start = time.perf_counter()
        compute_coverage(model, preview=args.preview)
        durations.append((time.perf_counter() - start) * 1000)
    print(f"model={args.model}")
    print(f"cells={model['grid']['rows'] * model['grid']['cols']} aps={len(model.get('access_points', []))} preview={args.preview}")
    print(f"median_ms={statistics.median(durations):.2f} p95_ms={sorted(durations)[max(0, int(args.runs * 0.95) - 1)]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
