"""Evaluate the authors' pretrained DANIEL FJSP policy on our instances.

Requires a local clone of https://github.com/wrqccc/FJSP-DRL and PyTorch.
The clone remains separate; this adapter converts our .fjs data to the
authors' environment and validates its full schedules independently.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

try:
    from .common import read_fjs, validate_schedule
except ImportError:
    from common import read_fjs, validate_schedule


def load_policy(upstream_root, checkpoint):
    import torch

    upstream_root = Path(upstream_root).resolve()
    if not (upstream_root / "model/main_model.py").exists():
        raise FileNotFoundError(f"DANIEL source is missing from {upstream_root}")
    sys.path.insert(0, str(upstream_root))
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "--device", "cpu"]
        from params import configs
        from model.main_model import DANIEL
        from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    finally:
        sys.argv = original_argv
    torch.set_num_threads(1)
    policy = DANIEL(configs)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    policy.eval()
    return policy, FJSPEnvForSameOpNums


def as_arrays(instance):
    import numpy as np

    job_lengths = np.array([len(job) for job in instance["jobs"]], dtype=np.int32)
    op_times = np.zeros((int(job_lengths.sum()), instance["n_machines"]), dtype=np.int32)
    row = 0
    for job in instance["jobs"]:
        for options in job:
            for machine, duration in options.items():
                op_times[row, machine] = duration
            row += 1
    return job_lengths, op_times


def solve(instance, policy, env_class):
    import numpy as np
    import torch

    job_lengths, op_times = as_arrays(instance)
    env = env_class(n_j=len(job_lengths), n_m=instance["n_machines"])
    schedule = [[None] * length for length in job_lengths]
    offsets = np.r_[0, np.cumsum(job_lengths)[:-1]]
    start_time = perf_counter()
    state = env.set_initial_data([job_lengths], [op_times])
    for _ in range(len(op_times)):
        with torch.inference_mode():
            probabilities, _ = policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
        action = int(torch.argmax(probabilities, dim=1).item())
        job, machine = divmod(action, instance["n_machines"])
        op = int(env.candidate[0, job] - offsets[job])
        start = max(env.true_candidate_free_time[0, job],
                    env.true_mch_free_time[0, machine])
        schedule[job][op] = [machine + 1, int(start)]
        state, _, done = env.step(actions=np.array([action], dtype=np.int64))
    elapsed_ms = (perf_counter() - start_time) * 1000
    if not bool(done[0]):
        raise ValueError("DANIEL environment did not finish")
    result = {"id": instance["id"], "makespan": int(env.current_makespan[0]),
              "schedule": schedule}
    validate_schedule(instance, result)
    return result, elapsed_ms


def run_folder(data_dir, upstream_root, checkpoint, output, schedules_output,
               reference_root=Path("CP-SAT")):
    policy, env_class = load_policy(upstream_root, checkpoint)
    paths = sorted(Path(data_dir).glob("*.fjs"))
    if not paths:
        raise ValueError(f"No .fjs instances in {data_dir}")
    # Warm the model before measuring each schedule.
    solve(read_fjs(paths[0]), policy, env_class)
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    schedules_output.parent.mkdir(parents=True, exist_ok=True)
    with schedules_output.open("w") as schedule_stream:
        for number, path in enumerate(paths, 1):
            instance = read_fjs(path)
            result, runtime_ms = solve(instance, policy, env_class)
            key = str(path.resolve().relative_to(Path.cwd().resolve()))
            reference_path = (reference_root / path.resolve().relative_to(
                Path("data").resolve())).with_suffix(".json")
            reference = json.loads(reference_path.read_text())
            rows.append({
                "rule": "daniel", "seed": "pretrained-greedy", "instance": key,
                "makespan": result["makespan"],
                "reference_makespan": reference["makespan"],
                "reference_status": reference["status"],
                "gap_percent": round(100 * (result["makespan"] / reference["makespan"] - 1), 4),
                "runtime_ms": round(runtime_ms, 4), "evaluations": 1,
                "feasible": True,
            })
            schedule_stream.write(json.dumps({"instance": key, **result}) + "\n")
            if number % 10 == 0 or number == len(paths):
                print(f"DANIEL {number}/{len(paths)}", flush=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output} and {schedules_output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/SD2_lit/10x5+mix"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("baselines/results/daniel_sd2_10x5.csv"))
    parser.add_argument("--schedules-output", type=Path,
                        default=Path("baselines/results/daniel_sd2_10x5_schedules.jsonl"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.upstream_root / "trained_network/SD2/10x5+mix.pth"
    run_folder(args.data_dir, args.upstream_root, checkpoint, args.output,
               args.schedules_output)


if __name__ == "__main__":
    main()
