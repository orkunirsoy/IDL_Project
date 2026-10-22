"""Run all dispatch baselines on the five midterm evaluation folders.

Usage: python3 scripts/run_midterm_baselines.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("data/SD_original/8x4/test", "8x4_test"),
    ("data/SD_original/10x5/test", "10x5_test"),
    ("data/SD1_lit/10x5", "sd1_10x5"),
    ("data/SD2_lit/10x5+mix", "sd2_10x5"),
    ("data/BenchData/Brandimarte", "brandimarte"),
]


def main():
    for folder, tag in DATASETS:
        for rule in ("fifo", "spt", "mor", "mwkr"):
            subprocess.run([sys.executable, f"baselines/{rule}.py", folder,
                            "--output", f"baselines/results/{rule}_{tag}.csv"],
                           cwd=ROOT, check=True)
        for seed in range(5):
            subprocess.run([sys.executable, "baselines/random_dispatch.py", folder,
                            "--seed", str(seed), "--output",
                            f"baselines/results/random_{tag}_seed{seed}.csv"],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
