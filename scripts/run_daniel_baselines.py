"""Evaluate the frozen pretrained SD2 DANIEL policy on all five FJSP sets.

Clone https://github.com/wrqccc/FJSP-DRL separately, then pass its path.
Requires PyTorch in the chosen Python environment.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.daniel import run_folder  # noqa: E402

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
    args = parser.parse_args()
    checkpoint = args.upstream_root / "trained_network/SD2/10x5+mix.pth"
    for folder, tag in DATASETS:
        csv_path = ROOT / "baselines/results" / f"daniel_{tag}.csv"
        schedules = ROOT / "baselines/results" / f"daniel_{tag}_schedules.jsonl"
        run_folder(ROOT / "data" / folder, args.upstream_root, checkpoint,
                   csv_path, schedules, ROOT / "CP-SAT")


if __name__ == "__main__":
    main()
