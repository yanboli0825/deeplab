#!/usr/bin/env python
"""Aggregate a numeric field from one or more run_summary.json files."""

import glob
import json
import math
import statistics
import sys
from typing import List


def _load_values(pattern: str, field: str) -> List[float]:
    values: List[float] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        value = payload.get(field)
        if value is None:
            continue

        numeric = float(value)
        if math.isnan(numeric):
            continue
        values.append(numeric)
    return values


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: aggregate_json_metrics.py '<glob>' <field>", file=sys.stderr)
        return 1

    pattern, field = sys.argv[1], sys.argv[2]
    values = _load_values(pattern, field)
    if not values:
        print(json.dumps({"field": field, "count": 0, "mean": None}, ensure_ascii=False))
        return 0

    payload = {
        "field": field,
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
