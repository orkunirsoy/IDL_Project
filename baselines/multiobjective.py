"""NSGA-II and MOEA/D for makespan and normalized temporal slack.

The machine ordering and assignment in a decoded schedule define a DAG. For
each operation, ES/LS are its earliest/latest feasible start in that DAG with
the schedule's makespan as horizon. Following GOAL, R=sum(LS-ES)/Cmax. We
minimize (Cmax, -R), then report the Pareto member with minimum makespan.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

try:
    from .common import read_fjs, validate_schedule
    from .evolutionary import decode, seeded_population, stable_seed
except ImportError:
    from common import read_fjs, validate_schedule
    from evolutionary import decode, seeded_population, stable_seed


def resilience(instance, result):
    """GOAL-style sum of operation slack divided by schedule makespan."""
    validate_schedule(instance, result)
    jobs, schedule = instance["jobs"], result["schedule"]
    operations = []
    by_machine = [[] for _ in range(instance["n_machines"])]
    for j, job in enumerate(jobs):
        for k, choices in enumerate(job):
            machine_id, start = schedule[j][k]
            node = (j, k)
            duration = choices[machine_id - 1]
            operations.append((start, node, duration))
            by_machine[machine_id - 1].append((start, node))
    successors = {node: [] for _, node, _ in operations}
    predecessors = {node: [] for _, node, _ in operations}
    duration = {node: p for _, node, p in operations}
    for j, job in enumerate(jobs):
        for k in range(len(job) - 1):
            a, b = (j, k), (j, k + 1)
            successors[a].append(b)
            predecessors[b].append(a)
    for machine in by_machine:
        machine.sort()
        for (_, a), (_, b) in zip(machine, machine[1:]):
            successors[a].append(b)
            predecessors[b].append(a)
    order = [node for _, node, _ in sorted(operations)]
    earliest = {}
    for node in order:
        earliest[node] = max((earliest[p] + duration[p] for p in predecessors[node]),
                             default=0)
    latest = {}
    horizon = result["makespan"]
    for node in reversed(order):
        latest[node] = min((latest[s] - duration[node] for s in successors[node]),
                           default=horizon - duration[node])
        if latest[node] < earliest[node]:
            raise ValueError(f"Negative slack at {node}")
    return sum(latest[n] - earliest[n] for n in order) / horizon


class MultiobjectiveFjsp(ElementwiseProblem):
    def __init__(self, instance):
        self.instance = instance
        self.archive = []
        n_ops = sum(map(len, instance["jobs"]))
        super().__init__(n_var=2 * n_ops, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        schedule = decode(self.instance, x)
        value = (schedule["makespan"], -resilience(self.instance, schedule))
        out["F"] = value
        if not any(a <= value[0] and b <= value[1] for a, b, _ in self.archive):
            self.archive = [(a, b, y) for a, b, y in self.archive
                            if not (value[0] <= a and value[1] <= b)]
            self.archive.append((value[0], value[1], np.array(x, copy=True)))


def solve(instance, method="nsga2", n_eval=1000, seed=0):
    if n_eval < 40:
        raise ValueError("n_eval must be at least 40")
    sampling = seeded_population(instance, seed)
    if method == "nsga2":
        algorithm = NSGA2(pop_size=40, sampling=sampling, eliminate_duplicates=False)
    elif method == "moead":
        directions = get_reference_directions("das-dennis", 2, n_partitions=39)
        algorithm = MOEAD(ref_dirs=directions, sampling=sampling, n_neighbors=10)
    else:
        raise ValueError(f"Unknown method: {method}")
    problem = MultiobjectiveFjsp(instance)
    start = perf_counter()
    result = minimize(problem, algorithm, ("n_eval", n_eval),
                      seed=seed, verbose=False)
    elapsed_ms = (perf_counter() - start) * 1000
    best = min(problem.archive, key=lambda entry: (entry[0], entry[1]))
    schedule = decode(instance, best[2])
    r_value = resilience(instance, schedule)
    if abs(best[0] - schedule["makespan"]) > 1e-8 or abs(best[1] + r_value) > 1e-8:
        raise ValueError("Objective mismatch")
    return schedule, r_value, len(problem.archive), elapsed_ms, result.algorithm.evaluator.n_eval


def run_folder(data_folder, method, n_eval, seed, output, reference_root=Path("CP-SAT")):
    paths = sorted(data_folder.glob("*.fjs"))
    if not paths:
        raise ValueError(f"No instances in {data_folder}")
    rows = []
    for number, path in enumerate(paths, 1):
        instance = read_fjs(path)
        schedule, r_value, pareto_size, runtime_ms, evaluations = solve(
            instance, method, n_eval, stable_seed(path, seed))
        relative = path.resolve().relative_to(Path("data").resolve()).with_suffix(".json")
        reference = json.loads((reference_root / relative).read_text())
        ref_makespan = reference["makespan"]
        rows.append({
            "rule": method, "seed": seed,
            "instance": str(path.resolve().relative_to(Path.cwd().resolve())),
            "makespan": schedule["makespan"], "resilience": round(r_value, 6),
            "pareto_size": pareto_size,
            "reference_makespan": ref_makespan,
            "reference_status": reference["status"],
            "gap_percent": round(100 * (schedule["makespan"] / ref_makespan - 1), 4),
            "runtime_ms": round(runtime_ms, 4), "evaluations": evaluations,
            "feasible": True,
        })
        if number % 10 == 0 or number == len(paths):
            print(f"{method.upper()} {data_folder.name} seed {seed} {number}/{len(paths)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}: mean makespan {mean(row['makespan'] for row in rows):.2f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_folder", type=Path)
    parser.add_argument("--method", choices=["nsga2", "moead"], required=True)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_folder(args.data_folder, args.method, args.n_eval, args.seed, args.output)


if __name__ == "__main__":
    main()
