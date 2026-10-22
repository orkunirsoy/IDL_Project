"""Step 2: solve every instance in a folder with OR-Tools CP-SAT.

DATA_FOLDER is the input. The script solves every .fjs file in that folder,
including files in subfolders. Results go under CP-SAT, with the same path
the files have under data:

    data/SD_original/10x5/test/test_0.fjs
        -> CP-SAT/SD_original/10x5/test/test_0.json
    data/BenchData/Brandimarte/BrandimarteMk1.fjs
        -> CP-SAT/BenchData/Brandimarte/BrandimarteMk1.json

A result is the machine and start time of every operation. Machine numbers
start at 1, matching the .fjs file. "OPTIMAL" means the solver proved no
shorter schedule exists. "FEASIBLE" means it ran out of time before it could
prove that.

Paste the folder into DATA_FOLDER below, then run this file. The folder is
relative to data/, for example "SD_original/10x5/test" or
"BenchData/Brandimarte".

Needs: pip install ortools tqdm
"""
import json
import os
import sys

from ortools.sat.python import cp_model
from tqdm import tqdm

TIME_LIMIT = 600  # seconds per instance
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "data")
OUT_ROOT = os.path.join(ROOT, "CP-SAT")
DATA_FOLDER = "SD1_lit/10x5"  # folder under data/. Change this, then run the file.


def resolve_folder(path):
    if os.path.isabs(path) and os.path.isdir(path):
        return os.path.abspath(path)
    for candidate in (
        os.path.join(DATA_ROOT, path),
        os.path.join(ROOT, path),
        os.path.join(os.path.dirname(ROOT), path),
        path,
    ):
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    sys.exit(f"folder not found: {path}")


def read_fjs(path):
    """Read one .fjs file. Machines in the file start at 1; the solver uses 0."""
    lines = open(path).read().strip().splitlines()
    head = lines[0].split()
    n_jobs = int(float(head[0]))
    n_machines = int(float(head[1]))
    tokens = iter(" ".join(lines[1:]).split())
    jobs = []
    for _ in range(n_jobs):
        job = []
        for _operation in range(int(next(tokens))):
            operation = []
            for _option in range(int(next(tokens))):
                machine = int(next(tokens)) - 1
                operation.append([machine, int(next(tokens))])
            job.append(operation)
        jobs.append(job)
    return {
        "id": os.path.splitext(os.path.basename(path))[0],
        "n_machines": n_machines,
        "jobs": jobs,
    }


def result_path(src):
    rel = os.path.relpath(os.path.abspath(src), DATA_ROOT)
    if rel.startswith(".."):
        sys.exit(f"{src} is not inside {DATA_ROOT}")
    return os.path.join(OUT_ROOT, os.path.splitext(rel)[0] + ".json")


def solve(instance):
    model = cp_model.CpModel()
    # No schedule can be longer than running every operation one after another
    # on its slowest machine, so this is a safe upper limit for all times.
    horizon = sum(max(time for _, time in operation) for job in instance["jobs"] for operation in job)

    intervals_on_machine = [[] for _ in range(instance["n_machines"])]
    variables = []  # variables[job][operation] = (start, {machine: chosen})
    last_ends = []

    for job in instance["jobs"]:
        job_variables = []
        previous_end = None
        for operation in job:
            start = model.new_int_var(0, horizon, "start")
            end = model.new_int_var(0, horizon, "end")
            chosen = {}
            for machine, time in operation:
                # The interval only exists if this machine is the chosen one.
                chosen[machine] = model.new_bool_var("chosen")
                interval = model.new_optional_interval_var(start, time, end, chosen[machine], "interval")
                intervals_on_machine[machine].append(interval)
            model.add_exactly_one(chosen.values())

            # Operations of one job run in order.
            if previous_end is not None:
                model.add(start >= previous_end)
            previous_end = end
            job_variables.append((start, chosen))
        variables.append(job_variables)
        last_ends.append(previous_end)

    # A machine runs one operation at a time.
    for intervals in intervals_on_machine:
        model.add_no_overlap(intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, last_ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"id": instance["id"], "makespan": None, "status": status_name, "schedule": None}

    schedule = []
    for job_variables in variables:
        job_schedule = []
        for start, chosen in job_variables:
            machine = next(m for m, is_chosen in chosen.items() if solver.value(is_chosen))
            # .fjs machine numbers start at 1.
            job_schedule.append([machine + 1, solver.value(start)])
        schedule.append(job_schedule)

    return {
        "id": instance["id"],
        "makespan": solver.value(makespan),
        "status": status_name,
        "schedule": schedule,
    }


def fjs_files(folder):
    paths = []
    for dirpath, _, names in os.walk(folder):
        for name in names:
            if name.endswith(".fjs"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def main():
    folder = resolve_folder(DATA_FOLDER)
    if os.path.relpath(folder, DATA_ROOT).startswith(".."):
        sys.exit(f"{folder} is not inside {DATA_ROOT}")
    paths = fjs_files(folder)
    if not paths:
        sys.exit(f"no .fjs files in {folder}")

    out_folder = os.path.join(OUT_ROOT, os.path.relpath(folder, DATA_ROOT))
    n_optimal = 0
    for path in tqdm(paths, desc=os.path.basename(folder)):
        solution = solve(read_fjs(path))
        n_optimal += solution["status"] == "OPTIMAL"
        out = result_path(path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(solution, f)
    print(f"solved {len(paths)} instances, {n_optimal} proven optimal -> {out_folder}")


if __name__ == "__main__":
    main()
