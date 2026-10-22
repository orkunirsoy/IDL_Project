"""Single-objective GA and BRKGA baselines for flexible job shop scheduling.

The chromosome has two random keys per operation: one gives dispatch priority,
the other selects one of that operation's eligible machines. The decoder only
selects a job's next operation and appends it to its chosen machine, so every
chromosome produces a feasible schedule.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

try:
    from .common import dispatch, read_fjs, validate_schedule
except ImportError:  # direct script execution
    from common import dispatch, read_fjs, validate_schedule


def decode(instance, chromosome):
    jobs = instance["jobs"]
    offsets = []
    n_operations = 0
    for job in jobs:
        offsets.append(n_operations)
        n_operations += len(job)
    priority = chromosome[:n_operations]
    machine_keys = chromosome[n_operations:]
    machine_ready = [0] * instance["n_machines"]
    job_ready = [0] * len(jobs)
    next_operation = [0] * len(jobs)
    schedule = [[None] * len(job) for job in jobs]
    for _ in range(n_operations):
        j = min((j for j, job in enumerate(jobs) if next_operation[j] < len(job)),
                key=lambda j: (priority[offsets[j] + next_operation[j]], j))
        k = next_operation[j]
        op_index = offsets[j] + k
        options = sorted(jobs[j][k].items())
        option_index = min(int(machine_keys[op_index] * len(options)), len(options) - 1)
        machine, duration = options[option_index]
        start = max(job_ready[j], machine_ready[machine])
        end = start + duration
        schedule[j][k] = [machine + 1, start]
        job_ready[j] = end
        machine_ready[machine] = end
        next_operation[j] += 1
    return {"id": instance["id"], "makespan": max(job_ready), "schedule": schedule}


class FjspProblem(ElementwiseProblem):
    def __init__(self, instance):
        self.instance = instance
        n_operations = sum(len(job) for job in instance["jobs"])
        super().__init__(n_var=2 * n_operations, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = decode(self.instance, x)["makespan"]


def seeded_population(instance, seed, population_size=40):
    """Seed four valid dispatch chromosomes, then fill with random keys."""
    n_operations = sum(len(job) for job in instance["jobs"])
    population = np.random.default_rng(seed).random((population_size, 2 * n_operations))
    offsets = []
    count = 0
    for job in instance["jobs"]:
        offsets.append(count)
        count += len(job)
    for row, rule in enumerate(("fifo", "mor", "mwkr", "spt")):
        heuristic = dispatch(instance, rule, return_order=True)
        for rank, (j, k) in enumerate(heuristic["dispatch_order"]):
            op_index = offsets[j] + k
            population[row, op_index] = (rank + 0.5) / n_operations
            options = sorted(instance["jobs"][j][k])
            machine = heuristic["schedule"][j][k][0] - 1
            chosen = options.index(machine)
            population[row, n_operations + op_index] = (chosen + 0.5) / len(options)
        if decode(instance, population[row])["makespan"] != heuristic["makespan"]:
            raise ValueError(f"Could not reproduce {rule} seed schedule")
    return population


def solve(instance, method="ga", n_eval=500, seed=0):
    if n_eval < 40:
        raise ValueError("n_eval must be at least 40")
    problem = FjspProblem(instance)
    sampling = seeded_population(instance, seed)
    if method == "ga":
        algorithm = GA(pop_size=40, sampling=sampling, eliminate_duplicates=False)
    elif method == "brkga":
        algorithm = BRKGA(n_elites=8, n_offsprings=28, n_mutants=4,
                          bias=0.7, sampling=sampling, eliminate_duplicates=False)
    else:
        raise ValueError(f"Unknown evolutionary method: {method}")
    begin = perf_counter()
    result = minimize(problem, algorithm, ("n_eval", n_eval), seed=seed, verbose=False)
    elapsed_ms = (perf_counter() - begin) * 1000
    schedule = decode(instance, np.asarray(result.X))
    validate_schedule(instance, schedule)
    if int(result.F[0]) != schedule["makespan"]:
        raise ValueError("Evolutionary objective and decoded schedule disagree")
    return schedule, elapsed_ms, result.algorithm.evaluator.n_eval


def stable_seed(path, seed):
    key = str(path.resolve().relative_to(Path("data").resolve()))
    digest = hashlib.sha256(key.encode()).digest()
    return (int.from_bytes(digest[:4], "big") + seed) % (2 ** 32)


def run_folder(data_folder, method, n_eval, seed, output, reference_root=Path("CP-SAT")):
    paths = sorted(data_folder.rglob("*.fjs"))
    if not paths:
        raise ValueError(f"No .fjs files in {data_folder}")
    rows = []
    for number, path in enumerate(paths, 1):
        instance = read_fjs(path)
        result, runtime_ms, eval_count = solve(instance, method, n_eval,
                                                stable_seed(path, seed))
        relative = path.resolve().relative_to(Path("data").resolve()).with_suffix(".json")
        reference_file = reference_root / relative
        reference = json.loads(reference_file.read_text()) if reference_file.exists() else None
        ref_makespan = reference.get("makespan") if reference else None
        rows.append({
            "rule": method,
            "seed": seed,
            "instance": str(path.resolve().relative_to(Path.cwd().resolve())),
            "makespan": result["makespan"],
            "reference_makespan": ref_makespan,
            "reference_status": reference.get("status", "") if reference else "",
            "gap_percent": round(100 * (result["makespan"] / ref_makespan - 1), 4) if ref_makespan else "",
            "runtime_ms": round(runtime_ms, 4),
            "evaluations": eval_count,
            "feasible": True,
        })
        if number % 10 == 0 or number == len(paths):
            print(f"{method.upper()} {data_folder.name} seed {seed} "
                  f"{number}/{len(paths)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{method.upper()} saved {output}: mean makespan "
          f"{mean(row['makespan'] for row in rows):.2f}, mean time "
          f"{mean(row['runtime_ms'] for row in rows):.2f} ms", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_folder", type=Path)
    parser.add_argument("--method", choices=["ga", "brkga"], required=True)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("CP-SAT"))
    args = parser.parse_args()
    run_folder(args.data_folder, args.method, args.n_eval, args.seed,
               args.output, args.reference_root)


if __name__ == "__main__":
    main()
