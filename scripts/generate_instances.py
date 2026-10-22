"""Step 1: generate random FJSP instances (only the problems, no solutions).

Writes one .fjs file per instance, the same text format as SD1, SD2, and the
benchmark files. The first line is:

    n_jobs    n_machines    average number of machines that can run one operation

Then one line per job. A job line starts with the number of operations. Each
operation is the number of machines that can run it, followed by machine and
time pairs. Machine numbers start at 1.

    10  5  2.4
    2  2 1 3 2 5  1 2 2

Job 0 has two operations. The first can run on machine 1 (3 minutes) or
machine 2 (5 minutes). The second runs only on machine 2 (2 minutes).

Change the settings below, then run this file.
"""
import os
import random

# ---- settings ---------------------------------------------------------------
N_JOBS = [8]              # several values, e.g. [5, 7, 10]: each instance picks one
N_MACHINES = [4]
OPS_PER_JOB = (3, 6)      # min and max operations in one job
MACHINES_PER_OP = (1, N_MACHINES)  # min and max machines that can run one operation
TIME_RANGE = (1, 20)      # min and max base processing time
VARIATION = 0.3           # each machine's time is up to 30% above or below the base time
SPLITS = {"train": 1000, "val": 100, "test": 100}
SEED = 0                  # same seed = same instances every run
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "SD_original")
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


def write_fjs(path, jobs, n_machines):
    n_ops = sum(len(job) for job in jobs)
    n_options = sum(len(operation) for job in jobs for operation in job)
    average = f"{n_options / n_ops:.2f}".rstrip("0").rstrip(".")
    lines = [f"{len(jobs)}\t{n_machines}\t{average}"]
    for job in jobs:
        parts = [str(len(job))]
        for operation in job:
            parts.append(str(len(operation)))
            for machine, time in operation:
                parts.append(str(machine + 1))
                parts.append(str(time))
        lines.append(" ".join(parts))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


random.seed(SEED)
written = {}

for split, count in SPLITS.items():
    for i in range(count):
        n_jobs = random.choice(N_JOBS)
        n_machines = random.choice(N_MACHINES)
        jobs = random_instance(n_jobs, n_machines)
        folder = os.path.join(DATA_FOLDER, f"{n_jobs}x{n_machines}", split)
        os.makedirs(folder, exist_ok=True)
        write_fjs(os.path.join(folder, f"{split}_{i}.fjs"), jobs, n_machines)
        written[folder] = written.get(folder, 0) + 1

for folder, count in written.items():
    print(f"wrote {count} instances to {folder}")
