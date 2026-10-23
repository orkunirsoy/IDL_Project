# Midterm plan: datasets and baselines

This note explains what is in the repo and how I think we should split the baseline work for the midterm report. Our own model, which assigns each operation to a machine and orders the operations on each machine, is still in progress. The baselines below can be built in parallel and do not depend on it.

## Datasets

Every instance in `data/` is a `.fjs` file, which is the standard text format for the flexible job shop problem. The first line gives the number of jobs, the number of machines, and the average number of machines that can process one operation. Each following line describes one job: the number of operations, then for each operation the number of eligible machines followed by `machine processing_time` pairs. Machine numbers start at 1. The literature sets and our generated set all use this format, so one reader works for everything.

The repo has four groups of data.

`data/SD_original` holds the instances we generated ourselves with `scripts/generate_instances.py`. Each size has its own folder, split into `train`, `val`, and `test`. Right now there are two sizes, `8x4` and `10x5`, each with 1000 training, 100 validation, and 100 test instances. This is the only set we split, and it is the one our model will be trained on.

`data/SD1_lit` comes from Song et al. (2023). It has 100 instances for each of six sizes, from `10x5` up to `40x10`. Processing times are between 1 and 20, and the number of operations per job varies (4 to 6 operations on 5 machines).

`data/SD2_lit` comes from Wang et al. (2024). It covers the same six sizes, but the folders are named with a `+mix` suffix, for example `10x5+mix`. The processing times have a much wider range, 1 to 99, and every job has exactly as many operations as there are machines.

`data/BenchData` contains the classic public benchmarks: the 10 Brandimarte instances (`mk01` to `mk10`) and the three Hurink sets (`edata`, `rdata`, `vdata`, 40 instances each). `BenchDataSolution.csv` lists published lower and upper bounds on the makespan for all of them. These instances come in mixed sizes and are only meant for testing.

For the midterm I think we should use a small subset of this:

| dataset | folder | instances |
|---|---|---|
| our data, 8 jobs, 4 machines | `data/SD_original/8x4/test` | 100 |
| our data, 10 jobs, 5 machines | `data/SD_original/10x5/test` | 100 |
| SD1, 10 jobs, 5 machines | `data/SD1_lit/10x5` | 100 |
| SD2, 10 jobs, 5 machines | `data/SD2_lit/10x5+mix` | 100 |
| Brandimarte | `data/BenchData/Brandimarte` | 10 |

The `10x5` size appears in all three synthetic sets, so it lets us compare our data with the two papers directly. Brandimarte gives us one standard benchmark with larger and less regular instances. The larger SD1 and SD2 sizes and the Hurink sets can wait until the final report.

Please evaluate only on the folders above. The SD1 and SD2 folders are the papers' test sets, so nobody should train on them, and for our own data we report on `test`, not on `train` or `val`.

## Baselines

The midterm rubric asks us to implement a baseline ourselves and to show results that come from our own runs. CP-SAT and the dispatching rules cover that requirement. The two deep learning methods are evaluated with the authors' released models, which we run ourselves on our data, and the report should say so.

### CP-SAT (done)

CP-SAT is the exact solver from Google OR-Tools. It gives us the reference makespan that every other method is compared against. The solver is implemented in `scripts/solve_instances.py`. Pass a folder under `data/`, a per-instance time limit, and `--skip-existing` to preserve earlier results; for example, `python3 scripts/solve_instances.py --data-folder SD_original/10x5/test --time-limit 10 --skip-existing`. The default time limit is 600 seconds. It writes one JSON file per instance under `CP-SAT/`, with the same path as the input:

```
data/SD_original/10x5/test/test_0.fjs  ->  CP-SAT/SD_original/10x5/test/test_0.json
data/BenchData/Brandimarte/BrandimarteMk1.fjs  ->  CP-SAT/BenchData/Brandimarte/BrandimarteMk1.json
```

Each result file contains the makespan, the solver status (`OPTIMAL` if it proved optimality, `FEASIBLE` if it found a schedule but could not prove optimality), and the full schedule as `[machine, start_time]` for every operation. Newly generated files also contain `wall_time_sec`, `time_limit_sec`, and `best_bound`; older files lack those fields.

All five evaluation folders now have CP-SAT schedules. Both `SD_original` test sets are 100/100 proven optimal. SD1 has 89/100 proven optimal, SD2 97/100, and Brandimarte 7/10; the rest are feasible incumbents from a 10-second run. The first 35 SD1 files predated this run and were already proven optimal. For Brandimarte, benchmark upper bounds should be reported separately from our CP-SAT incumbents. The local `BenchDataSolution.csv` lists Mk1 upper bound 39, but our solver proved 40 and the [SchedulingLab benchmark catalog](https://github.com/SchedulingLab/fjsp-instances/blob/main/instances.json) also lists 40 as the optimum, so that local bound should not be used without correction.

### Deep learning baselines

We will compare against two reinforcement learning methods for the flexible job shop problem. Both build a schedule step by step, choosing at each step which operation to schedule next and on which machine. Both papers also released their code and trained models, so we can evaluate them without retraining.

DRL (Song et al., 2023) represents the current state of the schedule as a graph with operation nodes and machine nodes, encodes it with a heterogeneous graph neural network, and trains the policy with PPO. It was trained on the SD1 distribution.

- Paper: W. Song, X. Chen, Q. Li, and Z. Cao, "Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning," IEEE Transactions on Industrial Informatics, 2023.
- Code: https://github.com/songwenas12/fjsp-drl
- Evaluation script: `test.py`, with the trained models in `model/`.

Our adapter is now `baselines/song.py`, with `scripts/run_song_baselines.py` evaluating the frozen `save_10_5.pt` checkpoint greedily on all five folders. CSV results and independently validated full schedules are saved under `baselines/results/song_*`. The adapter corrects an upstream machine-indexing issue in the time transition; see `baselines/README.md` for the reproduction command and environment details.

DANIEL (Wang et al., 2024) replaces the graph network with a dual attention network, one set of attention blocks for operations and one for machines, and is also trained with PPO. The paper reports better results than DRL, especially on SD2, where processing times vary more.

- Paper: R. Wang, G. Wang, J. Sun, F. Deng, and J. Chen, "Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning," IEEE Transactions on Neural Networks and Learning Systems, 2024.
- Code: https://github.com/wrqccc/FJSP-DRL
- Evaluation script: `test_trained_model.py`, with the trained models in `trained_network/`. For example, `python test_trained_model.py --data_source SD2 --model_source SD2 --test_data 10x5+mix --test_model 10x5+mix` evaluates the SD2 model on the SD2 10x5 test set.

A few things to keep in mind when running these:

- Both repos expect their own folder names (`SD1` and `SD2` instead of our `SD1_lit` and `SD2_lit`), so you may need to point the scripts at our folders or copy the files over. The `.fjs` files themselves are identical.
- Each model was trained on one data distribution. A DRL model trained on SD1 that we test on SD2 or on our own data is measuring generalization, and the report should describe it that way.
- Both methods can decode greedily (one schedule) or by sampling (the best of many schedules, 100 in the DANIEL paper). Sampling gives better makespans but takes longer, so please record which mode was used, and report greedy results first.
- For every instance, please save the makespan and the runtime under the same instance name as the `.fjs` file, so we can line the results up with CP-SAT.

### Priority dispatching rules

Priority dispatching rules are simple heuristics that build a schedule one operation at a time. At each step, the rule looks at the operations that are ready to start (their previous operation in the job is finished), scores each ready operation with an eligible machine, picks the best pair, and starts that operation as early as both the job and the machine allow. This repeats until every operation is scheduled. The rules need no training and run in milliseconds, so they give a useful lower bar for the learned methods.

Both papers compare against the same four rules, and we should implement all of them ourselves:

- FIFO (first in, first out): schedule the operation that became ready earliest.
- SPT (shortest processing time): schedule the operation with the shortest processing time.
- MOR (most operations remaining): schedule from the job with the most operations still to do.
- MWKR (most work remaining): schedule from the job with the most total processing time still to do. In both papers this is the strongest of the four.

The papers pair these with a machine choice, usually the eligible machine that can finish the operation earliest. Whatever we choose, we should use the same machine choice for all four rules and state it in the report. The DANIEL repo implements these rules in `common_utils.py`, which is a good reference if a definition is unclear, but the numbers we report should come from our own code.

The rules should be run on all five evaluation folders, saving the makespan and runtime for each instance in the same way as above.

## Suggested table for the midterm report

The main table can report the average makespan gap to CP-SAT on each dataset, computed per instance as (method makespan minus CP-SAT makespan) divided by the CP-SAT makespan:

| dataset | CP-SAT | FIFO | SPT | MOR | MWKR | DRL | DANIEL |
|---|---|---|---|---|---|---|---|
| SD_original 8x4 | 0.00% | | | | | | |
| SD_original 10x5 | 0.00% | | | | | | |
| SD1 10x5 | | | | | | | |
| SD2 10x5 | | | | | | | |
| Brandimarte | | | | | | | |

CP-SAT is the reference, so its gap is zero wherever it proves optimality. On Brandimarte, if it does not, we can compare against the published upper bound instead and add a short note under the table. If we have both greedy and sampling results for DRL and DANIEL, we can add two extra columns, with the number of samples in the header. A second, smaller table with the average runtime per instance would also be useful, since the speed of the learned methods is one of their main advantages over CP-SAT.
