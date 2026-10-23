"""Shared CLI for the five dispatch baselines."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from common import dispatch, read_fjs, validate_schedule


def main(rule):
    parser = argparse.ArgumentParser(description=f"Run {rule.upper()} dispatch on .fjs files")
    parser.add_argument("data_folder", type=Path)
    parser.add_argument("--reference-root", type=Path, default=Path("CP-SAT"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random rule seed; ignored by deterministic rules")
    args = parser.parse_args()
    paths = sorted(args.data_folder.rglob("*.fjs"))
    if not paths:
        parser.error(f"No .fjs files in {args.data_folder}")
    output = args.output or Path(f"baselines/results/{rule}_{args.data_folder.name}_seed{args.seed}.csv")
    rows = []
    for path in paths:
        instance = read_fjs(path)
        instance_key = str(path.resolve().relative_to(Path("data").resolve()))
        stable_id = int.from_bytes(hashlib.sha256(instance_key.encode()).digest()[:8], "big")
        begin = perf_counter()
        result = dispatch(instance, rule, seed=args.seed + stable_id)
        elapsed_ms = (perf_counter() - begin) * 1000
        validate_schedule(instance, result)
        relative = Path(instance_key).with_suffix(".json")
        reference_file = args.reference_root / relative
        reference = json.loads(reference_file.read_text()) if reference_file.exists() else None
        ref_makespan = reference.get("makespan") if reference else None
        rows.append({
            "rule": rule,
            "seed": args.seed if rule == "random" else "",
            "instance": str(path),
            "makespan": result["makespan"],
            "reference_makespan": ref_makespan,
            "reference_status": reference.get("status", "") if reference else "",
            "gap_percent": round(100 * (result["makespan"] / ref_makespan - 1), 4) if ref_makespan else "",
            "runtime_ms": round(elapsed_ms, 4),
            "feasible": True,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    gaps = [row["gap_percent"] for row in rows if row["gap_percent"] != ""]
    print(f"{rule.upper()}: {len(rows)} feasible schedules; mean makespan {mean(row['makespan'] for row in rows):.2f}")
    if gaps:
        print(f"Mean per-instance gap: {mean(gaps):.2f}% ({len(gaps)} references)")
    print(f"Mean dispatch runtime: {mean(row['runtime_ms'] for row in rows):.3f} ms; saved {output}")
