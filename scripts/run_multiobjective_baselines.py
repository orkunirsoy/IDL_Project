"""Run NSGA-II and MOEA/D on the five midterm FJSP evaluation sets."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.common import read_fjs  # noqa: E402
from baselines.multiobjective import run_folder, solve  # noqa: E402

DATASETS = [
    ("data/SD_original/8x4/test", "8x4_test"),
    ("data/SD_original/10x5/test", "10x5_test"),
    ("data/SD1_lit/10x5", "sd1_10x5"),
    ("data/SD2_lit/10x5+mix", "sd2_10x5"),
    ("data/BenchData/Brandimarte", "brandimarte"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    example = read_fjs(ROOT / DATASETS[0][0] / "test_0.fjs")
    for method in ("nsga2", "moead"):
        solve(example, method, 40, seed=0)  # warm up compiled kernels
    for folder, tag in DATASETS:
        for method in ("nsga2", "moead"):
            for seed in range(args.seeds):
                output = ROOT / "baselines/results" / f"{method}_{tag}_seed{seed}.csv"
                run_folder(ROOT / folder, method, args.n_eval, seed, output,
                           ROOT / "CP-SAT")


if __name__ == "__main__":
    main()
