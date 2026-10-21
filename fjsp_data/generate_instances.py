"""Step 1: generate random FJSP instances (only the problems, no solutions).

An instance is a list of jobs. A job is a list of operations that must run
in order. An operation is a list of [machine, time] pairs: the machines that
can run it, and how long it takes on each one.

    jobs = [
        [[[0, 3], [1, 5]], [[1, 2]]],    # job 0: two operations
        [[[0, 4]], [[0, 2], [1, 3]]],    # job 1: two operations
    ]

Job 0's first operation can run on machine 0 (3 minutes) or machine 1 (5 minutes).

Change the settings below, then run this file.
"""
import json
import os
import random

# ---- settings ---------------------------------------------------------------
N_JOBS = [6,7,8,9]              # several values, e.g. [5, 7, 10]: each instance picks one
N_MACHINES = [4,5,6,7]
OPS_PER_JOB = (3, 6)      # min and max operations in one job
MACHINES_PER_OP = (1, 3)  # min and max machines that can run one operation
TIME_RANGE = (1, 20)      # min and max base processing time
VARIATION = 0.3           # each machine's time is up to 30% above or below the base time
SPLITS = {"train": 2000, "val": 250, "test": 250}
SEED = 0                  # same seed = same instances every run
OUT_FOLDER = os.path.join(os.path.dirname(__file__), "data", "instances")
# -----------------------------------------------------------------------------


def random_instance(n_jobs, n_machines):
    jobs = []
    for _ in range(n_jobs):
        job = []
        for _ in range(random.randint(OPS_PER_JOB[0], OPS_PER_JOB[1])):
            n_options = random.randint(MACHINES_PER_OP[0], min(MACHINES_PER_OP[1], n_machines))
            machines = sorted(random.sample(range(n_machines), n_options))
            # Each machine gets a time close to the base time, so the choice of
            # machine changes how long the operation takes.
            base_time = random.randint(TIME_RANGE[0], TIME_RANGE[1])
            operation = []
            for machine in machines:
                time = round(base_time * (1 + random.uniform(-VARIATION, VARIATION)))
                operation.append([machine, max(1, time)])
            job.append(operation)
        jobs.append(job)
    return jobs


random.seed(SEED)
os.makedirs(OUT_FOLDER, exist_ok=True)

for split, count in SPLITS.items():
    instances = []
    for i in range(count):
        n_jobs = random.choice(N_JOBS)
        n_machines = random.choice(N_MACHINES)
        jobs = random_instance(n_jobs, n_machines)
        instances.append({"id": f"{split}_{i}", "n_machines": n_machines, "jobs": jobs})

    path = os.path.join(OUT_FOLDER, f"{split}.json")
    with open(path, "w") as f:
        json.dump(instances, f)
    print(f"wrote {count} instances to {path}")
