"""Evaluate the authors' pretrained Song FJSP policy on our instances.

Requires a local clone of https://github.com/songwenas12/fjsp-drl and PyTorch.
The clone remains separate; this adapter converts our .fjs data to the
authors' environment and validates its full schedules independently.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter

try:
    from .common import read_fjs, validate_schedule
except ImportError:
    from common import read_fjs, validate_schedule

ROOT = Path(__file__).resolve().parents[1]


def load_policy(upstream_root, checkpoint):
    import torch

    upstream_root = Path(upstream_root).resolve()
    if not (upstream_root / "PPO_model.py").exists():
        raise FileNotFoundError(f"Song source is missing from {upstream_root}")
    sys.path.insert(0, str(upstream_root))
    from PPO_model import HGNNScheduler
    from env.fjsp_env import FJSPEnv

    class SongEnv(FJSPEnv):
        next_time = advance_time

    configs = json.loads((upstream_root / "config.json").read_text())
    device = torch.device("cpu")
    model_paras = configs["model_paras"]
    model_paras["device"] = device
    model_paras["actor_in_dim"] = 2 * (model_paras["out_size_ma"] + model_paras["out_size_ope"])
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    torch.set_num_threads(1)
    policy = HGNNScheduler(model_paras).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    policy.eval()
    return policy, SongEnv, {**configs["env_paras"], "device": device, "batch_size": 1}


def advance_time(env, flag_trans_2_next_time):
    """Use upstream time transitions with machine-axis indexing corrected.

    The upstream transpose before boolean assignment indexes feature rows
    instead of machines on current PyTorch, corrupting machine finish times.
    """
    import torch

    needs_time = (flag_trans_2_next_time == 0) & (~env.done_batch)
    available = env.machines_batch[:, :, 1]
    future = torch.where(available > env.time[:, None], available,
                         torch.max(env.feat_opes_batch[:, 4, :]) + 1.0)
    next_time = torch.min(future, dim=1)[0]
    completed = ((available == next_time[:, None]) &
                 (env.machines_batch[:, :, 0] == 0) & needs_time[:, None])
    env.time = torch.where(needs_time, next_time, env.time)
    env.machines_batch[:, :, 0][completed] = 1

    utilization = env.machines_batch[:, :, 2]
    current_time = env.time[:, None].expand_as(utilization)
    env.feat_mas_batch[:, 2, :] = torch.minimum(utilization, current_time).div(
        env.time[:, None] + 1e-5)
    batches, machines = torch.where(completed)
    jobs = env.machines_batch[batches, machines, 3].long()
    env.mask_job_procing_batch[batches, jobs] = False
    env.mask_ma_procing_batch[completed] = False
    env.mask_job_finish_batch = torch.where(
        env.ope_step_batch == env.end_ope_biases_batch + 1,
        True, env.mask_job_finish_batch)


class FjspCase:
    """Pass canonical .fjs lines to the upstream environment without file I/O."""

    def __init__(self, instance):
        n_jobs, n_machines = len(instance["jobs"]), instance["n_machines"]
        self.lines = [f"{n_jobs} {n_machines} 0\n"]
        for job in instance["jobs"]:
            parts = [str(len(job))]
            for options in job:
                parts.append(str(len(options)))
                for machine, duration in options.items():
                    parts.extend((str(machine + 1), str(duration)))
            self.lines.append(" ".join(parts) + "\n")

    def get_case(self, index):
        return self.lines, None, None


def integer(value):
    """Reject fractional or nonfinite upstream values before exporting them."""
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"Song returned a nonintegral schedule value: {value}")
    return int(value)


def solve(instance, policy, env_class, env_paras):
    import torch

    case = FjspCase(instance)
    parameters = {**env_paras, "num_jobs": len(instance["jobs"]),
                  "num_mas": instance["n_machines"]}
    n_operations = sum(len(job) for job in instance["jobs"])
    start_time = perf_counter()
    env = env_class(case=case, env_paras=parameters, data_source="case")
    for _ in range(n_operations):
        if bool(env.done_batch.all()):
            raise ValueError("Song environment finished before scheduling every operation")
        with torch.inference_mode():
            actions = policy.act(env.state, None, env.done_batch,
                                 flag_sample=False, flag_train=False)
        env.step(actions)
    elapsed_ms = (perf_counter() - start_time) * 1000
    if not bool(env.done_batch.all()):
        raise ValueError("Song environment did not finish")

    operations = env.schedules_batch[0].tolist()
    if len(operations) != n_operations or any(row[0] != 1 for row in operations):
        raise ValueError("Song returned an incomplete schedule")
    schedule = []
    offset = 0
    for job in instance["jobs"]:
        job_schedule = []
        for options, (_, machine, start, end) in zip(job, operations[offset:offset + len(job)]):
            machine, start, end = integer(machine), integer(start), integer(end)
            if machine not in options or end != start + options[machine]:
                raise ValueError("Song returned an invalid machine or operation duration")
            job_schedule.append([machine + 1, start])
        schedule.append(job_schedule)
        offset += len(job)
    result = {"id": instance["id"], "makespan": integer(env.makespan_batch[0].item()),
              "schedule": schedule}
    validate_schedule(instance, result)
    return result, elapsed_ms


def run_folder(data_dir, upstream_root, checkpoint, output, schedules_output,
               reference_root=Path("CP-SAT")):
    policy, env_class, env_paras = load_policy(upstream_root, checkpoint)
    paths = sorted(Path(data_dir).glob("*.fjs"))
    if not paths:
        raise ValueError(f"No .fjs instances in {data_dir}")
    # Warm the model before measuring each schedule.
    solve(read_fjs(paths[0]), policy, env_class, env_paras)
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    schedules_output.parent.mkdir(parents=True, exist_ok=True)
    with schedules_output.open("w") as schedule_stream:
        for number, path in enumerate(paths, 1):
            instance = read_fjs(path)
            result, runtime_ms = solve(instance, policy, env_class, env_paras)
            key = path.resolve().relative_to(ROOT).as_posix()
            reference_path = (reference_root / path.resolve().relative_to(
                ROOT / "data")).with_suffix(".json")
            reference = json.loads(reference_path.read_text())
            rows.append({
                "rule": "song", "seed": "pretrained-greedy", "instance": key,
                "makespan": result["makespan"],
                "reference_makespan": reference["makespan"],
                "reference_status": reference["status"],
                "gap_percent": round(100 * (result["makespan"] / reference["makespan"] - 1), 4),
                "runtime_ms": round(runtime_ms, 4), "evaluations": 1,
                "feasible": True,
            })
            schedule_stream.write(json.dumps({"instance": key, **result}) + "\n")
            if number % 10 == 0 or number == len(paths):
                print(f"Song {number}/{len(paths)}", flush=True)
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
                        default=Path("data/SD1_lit/10x5"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("baselines/results/song_sd1_10x5.csv"))
    parser.add_argument("--schedules-output", type=Path,
                        default=Path("baselines/results/song_sd1_10x5_schedules.jsonl"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.upstream_root / "model/save_10_5.pt"
    run_folder(args.data_dir, args.upstream_root, checkpoint, args.output,
               args.schedules_output)


if __name__ == "__main__":
    main()
