"""Step 3: build many schedules of different quality for every instance.

A model conditioned on a target makespan needs to see good and bad schedules
of the same problem, not only the optimal one.

Each operation stays on the machine the solver chose. We place operations one
at a time, in order of their start time in the optimal schedule plus random
noise. With no noise this rebuilds the optimal makespan. More noise moves the
order further from optimal, so the makespan gets longer.

Reads data/instances/ and data/solutions/, writes data/schedules/:

    {"id": "train_0", "schedules": [{"makespan": 62, "schedule": [...]}, ...]}

Entry 0 is always the optimal one. Each schedule has the same shape as in
step 2: schedule[job][operation] = [machine, start_time].
"""
import json
import os
import random

SCHEDULES_PER_INSTANCE = 50
MAX_NOISE = 0.2  # largest noise, as a fraction of the optimal makespan
SEED = 0
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


def build_schedule(instance, machines, priority):
    """Place operations one at a time. Each step takes the next operation of
    the job with the lowest priority number, and starts it as soon as both its
    job and its machine are free."""
    jobs = instance["jobs"]
    next_op = [0] * len(jobs)
    job_free = [0] * len(jobs)
    machine_free = [0] * instance["n_machines"]
    schedule = [[None] * len(job) for job in jobs]

    for _ in range(sum(len(job) for job in jobs)):
        unfinished = [j for j in range(len(jobs)) if next_op[j] < len(jobs[j])]
        job = min(unfinished, key=lambda j: priority[j][next_op[j]])
        op = next_op[job]
        machine = machines[job][op]
        time = dict(jobs[job][op])[machine]
        start = max(job_free[job], machine_free[machine])

        schedule[job][op] = [machine, start]
        job_free[job] = start + time
        machine_free[machine] = start + time
        next_op[job] += 1

    return schedule, max(job_free)


random.seed(SEED)
os.makedirs(os.path.join(DATA_FOLDER, "schedules"), exist_ok=True)

for split in ["train", "val", "test"]:
    with open(os.path.join(DATA_FOLDER, "instances", f"{split}.json")) as f:
        instances = json.load(f)
    with open(os.path.join(DATA_FOLDER, "solutions", f"{split}.json")) as f:
        solutions = json.load(f)

    results = []
    for instance, solution in zip(instances, solutions):
        machines = [[machine for machine, _ in job] for job in solution["schedule"]]
        optimal_starts = [[start for _, start in job] for job in solution["schedule"]]

        schedule, makespan = build_schedule(instance, machines, optimal_starts)
        schedules = [{"makespan": makespan, "schedule": schedule}]
        seen = {str(schedule)}

        # Some small instances have fewer than SCHEDULES_PER_INSTANCE different
        # schedules, so stop trying after a while.
        for _ in range(20 * SCHEDULES_PER_INSTANCE):
            if len(schedules) == SCHEDULES_PER_INSTANCE:
                break
            noise = random.uniform(0, MAX_NOISE) * solution["makespan"]
            priority = [[start + random.gauss(0, noise) for start in job] for job in optimal_starts]
            schedule, makespan = build_schedule(instance, machines, priority)
            if str(schedule) not in seen:
                seen.add(str(schedule))
                schedules.append({"makespan": makespan, "schedule": schedule})

        results.append({"id": instance["id"], "schedules": schedules})

    path = os.path.join(DATA_FOLDER, "schedules", f"{split}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"{split}: {len(results)} instances -> {path}")
