"""Evaluate the frozen pretrained Song policy on all five FJSP sets.

Clone https://github.com/songwenas12/fjsp-drl separately, then pass its path.
Requires the packages in requirements-song.txt in the chosen Python environment.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.song import run_folder  # noqa: E402

DATASETS = [
    ("SD_original/8x4/test", "8x4_test"),
    ("SD_original/10x5/test", "10x5_test"),
    ("SD1_lit/10x5", "sd1_10x5"),
    ("SD2_lit/10x5+mix", "sd2_10x5"),
    ("BenchData/Brandimarte", "brandimarte"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.upstream_root / "model/save_10_5.pt"
    suffix = "" if args.samples == 1 else f"_sample{args.samples}"
    for folder, tag in DATASETS:
        csv_path = ROOT / "baselines/results" / f"song_{tag}{suffix}.csv"
        schedules = ROOT / "baselines/results" / f"song_{tag}{suffix}_schedules.jsonl"
        run_folder(ROOT / "data" / folder, args.upstream_root, checkpoint,
                   csv_path, schedules, ROOT / "CP-SAT", args.samples)


if __name__ == "__main__":
    main()
