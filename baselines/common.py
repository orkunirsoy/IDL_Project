"""Shared FJSP parsing, schedule construction, and validation.

Machine IDs are zero based internally and one based in .fjs/CP-SAT files.
"""

from pathlib import Path
import random


def read_fjs(path):
    tokens = Path(path).read_text().split()
    if len(tokens) < 3:
        raise ValueError(f"Incomplete header: {path}")
    n_jobs, n_machines = int(tokens[0]), int(tokens[1])
    position = 3  # average eligibility in the header is descriptive
    jobs = []
    for _ in range(n_jobs):
        n_operations = int(tokens[position])
        position += 1
        job = []
        for _ in range(n_operations):
            n_options = int(tokens[position])
            position += 1
            options = {}
            for _ in range(n_options):
                machine = int(tokens[position]) - 1
                duration = int(tokens[position + 1])
                position += 2
                if not (0 <= machine < n_machines) or duration <= 0 or machine in options:
                    raise ValueError(f"Invalid machine option in {path}")
                options[machine] = duration
            if not options:
                raise ValueError(f"Operation without an eligible machine in {path}")
            job.append(options)
        jobs.append(job)
    if position != len(tokens):
        raise ValueError(f"Unexpected trailing data in {path}")
    return {"id": Path(path).stem, "n_machines": n_machines, "jobs": jobs}


def dispatch(instance, rule, seed=0, return_order=False):
    """Construct a schedule with one dispatch rule and earliest-completion routing.

    Only the next operation of each unfinished job is considered. Every rule
    chooses a job/operation; its eligible machine is chosen by earliest finish.
    MWKR estimates remaining work using the fastest eligible duration of each
    remaining operation. Random dispatch uses a local seeded generator.
    """
    if rule not in {"fifo", "spt", "mor", "mwkr", "random"}:
        raise ValueError(f"Unknown dispatch rule: {rule}")
    rng = random.Random(seed)
    jobs = instance["jobs"]
    machine_ready = [0] * instance["n_machines"]
    job_ready = [0] * len(jobs)
    next_operation = [0] * len(jobs)
    schedule = [[None] * len(job) for job in jobs]
    order = []
    remaining = sum(map(len, jobs))

    while remaining:
        candidates = []
        for j, job in enumerate(jobs):
            k = next_operation[j]
            if k == len(job):
                continue
            choices = []
            for machine, duration in job[k].items():
                start = max(job_ready[j], machine_ready[machine])
                choices.append((start + duration, start, machine, duration))
            end, start, machine, duration = min(choices)
            if rule == "fifo":
                priority = (job_ready[j], start, end)
            elif rule == "spt":
                priority = (duration, start, end)
            elif rule == "mor":
                priority = (-(len(job) - k), start, end)
            elif rule == "mwkr":
                work = sum(min(option.values()) for option in job[k:])
                priority = (-work, start, end)
            else:
                priority = ()
            candidates.append((priority, j, k, machine, start, end))
        _, j, k, machine, start, end = (rng.choice(candidates) if rule == "random"
                                        else min(candidates))
        schedule[j][k] = [machine + 1, start]
        order.append((j, k))
        machine_ready[machine] = end
        job_ready[j] = end
        next_operation[j] += 1
        remaining -= 1

    result = {"id": instance["id"], "makespan": max(job_ready), "schedule": schedule}
    if return_order:
        result["dispatch_order"] = order
    return result


def dispatch_spt(instance):
    """Backward-compatible name for the first baseline."""
    return dispatch(instance, "spt")


def validate_schedule(instance, result):
    """Return the true makespan, raising ValueError on any infeasibility."""
    jobs, schedule = instance["jobs"], result["schedule"]
    if len(schedule) != len(jobs):
        raise ValueError("Wrong number of jobs")
    by_machine = [[] for _ in range(instance["n_machines"])]
    makespan = 0
    for j, job in enumerate(jobs):
        if len(schedule[j]) != len(job):
            raise ValueError(f"Wrong operation count for job {j}")
        previous_end = 0
        for k, options in enumerate(job):
            machine_id, start = schedule[j][k]
            machine = machine_id - 1
            if machine not in options or not isinstance(start, int) or start < previous_end:
                raise ValueError(f"Invalid assignment or job order at J{j + 1}-O{k + 1}")
            end = start + options[machine]
            by_machine[machine].append((start, end, j, k))
            previous_end = end
            makespan = max(makespan, end)
    for m, intervals in enumerate(by_machine):
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                raise ValueError(f"Overlap on M{m + 1}: {left} and {right}")
    if result.get("makespan") != makespan:
        raise ValueError(f"Reported makespan {result.get('makespan')} != {makespan}")
    return makespan
