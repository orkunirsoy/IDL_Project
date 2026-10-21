"""Step 2: solve every instance with OR-Tools CP-SAT (shortest makespan).

Reads data/instances/{train,val,test}.json and writes data/solutions/ with the
same file names. A solution has the same shape as the instance's jobs:

    schedule[job][operation] = [machine, start_time]

The makespan is the time the last operation finishes. "OPTIMAL" means the
solver proved no shorter schedule exists. "FEASIBLE" means it ran out of time
before it could prove that.

Needs: pip install ortools
"""
import json
import os

from ortools.sat.python import cp_model

TIME_LIMIT = 60  # seconds per instance
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


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

    schedule = []
    for job_variables in variables:
        job_schedule = []
        for start, chosen in job_variables:
            machine = next(m for m, is_chosen in chosen.items() if solver.value(is_chosen))
            job_schedule.append([machine, solver.value(start)])
        schedule.append(job_schedule)

    return {
        "id": instance["id"],
        "makespan": solver.value(makespan),
        "status": solver.status_name(status),
        "schedule": schedule,
    }


os.makedirs(os.path.join(DATA_FOLDER, "solutions"), exist_ok=True)

for split in ["train", "val", "test"]:
    with open(os.path.join(DATA_FOLDER, "instances", f"{split}.json")) as f:
        instances = json.load(f)

    solutions = [solve(instance) for instance in instances]
    n_optimal = sum(solution["status"] == "OPTIMAL" for solution in solutions)

    path = os.path.join(DATA_FOLDER, "solutions", f"{split}.json")
    with open(path, "w") as f:
        json.dump(solutions, f)
    print(f"{split}: solved {len(solutions)} instances, {n_optimal} proven optimal -> {path}")
