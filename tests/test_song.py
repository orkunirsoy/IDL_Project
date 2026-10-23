"""Check the Song adapter's instance and schedule boundaries."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from baselines.common import read_fjs
from baselines.song import FjspCase, advance_time, solve


class SongAdapterTests(unittest.TestCase):
    def setUp(self):
        self.instance = {
            "id": "tiny",
            "n_machines": 2,
            "jobs": [[{0: 2}, {1: 3}], [{1: 4}]],
        }
        self.rows = [[1, 0, 0, 2], [1, 1, 4, 7], [1, 1, 0, 4]]

    def solve_rows(self, rows=None, makespan=7, complete=True):
        env = SimpleNamespace(
            state=object(),
            done_batch=torch.tensor([False]),
            schedules_batch=torch.tensor(
                [self.rows if rows is None else rows], dtype=torch.float64,
            ),
            makespan_batch=torch.tensor([makespan], dtype=torch.float64),
        )
        count = 0

        def step(actions):
            nonlocal count
            count += 1
            env.done_batch = torch.tensor([complete and count == 3])
            return env.state, torch.tensor([0.0]), env.done_batch

        env.step = Mock(side_effect=step)
        policy = Mock()
        policy.act.return_value = torch.zeros((3, 1), dtype=torch.int64)
        result, elapsed_ms = solve(
            self.instance, policy, Mock(return_value=env), {},
        )
        return result, elapsed_ms, policy, env

    def test_case_round_trip_preserves_variable_job_lengths_and_options(self):
        instance = copy.deepcopy(self.instance)
        instance["jobs"][0][0][1] = 5
        lines = FjspCase(instance).get_case(0)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.fjs"
            path.write_text("".join(lines))
            restored = read_fjs(path)
        self.assertEqual(restored, instance)

    def test_reconstructs_complete_schedule_in_job_order(self):
        result, elapsed_ms, policy, env = self.solve_rows()
        self.assertEqual(result, {
            "id": "tiny",
            "makespan": 7,
            "schedule": [[[1, 0], [2, 4]], [[2, 0]]],
        })
        self.assertGreaterEqual(elapsed_ms, 0)
        self.assertEqual(policy.act.call_count, 3)
        self.assertEqual(env.step.call_count, 3)
        for call in policy.act.call_args_list:
            self.assertFalse(call.kwargs["flag_sample"])
            self.assertFalse(call.kwargs["flag_train"])

    def test_rejects_ineligible_machine(self):
        rows = copy.deepcopy(self.rows)
        rows[0][1] = 1
        with self.assertRaises(ValueError):
            self.solve_rows(rows=rows)

    def test_rejects_broken_job_order(self):
        rows = [[1, 0, 0, 2], [1, 1, 1, 4], [1, 1, 4, 8]]
        with self.assertRaises(ValueError):
            self.solve_rows(rows=rows, makespan=8)

    def test_rejects_machine_overlap(self):
        rows = [[1, 0, 0, 2], [1, 1, 3, 6], [1, 1, 0, 4]]
        with self.assertRaises(ValueError):
            self.solve_rows(rows=rows, makespan=6)

    def test_rejects_incorrect_reported_makespan(self):
        with self.assertRaises(ValueError):
            self.solve_rows(makespan=8)

    def test_rejects_unscheduled_operation(self):
        rows = copy.deepcopy(self.rows)
        rows[1][0] = 0
        with self.assertRaises(ValueError):
            self.solve_rows(rows=rows)

    def test_rejects_incomplete_environment(self):
        with self.assertRaises(ValueError):
            self.solve_rows(complete=False)

    def test_rejects_nonintegral_schedule_values(self):
        for column, value in [(1, 0.5), (2, 0.5)]:
            with self.subTest(column=column):
                rows = copy.deepcopy(self.rows)
                rows[0][column] = value
                with self.assertRaises(ValueError):
                    self.solve_rows(rows=rows)
        with self.assertRaises(ValueError):
            self.solve_rows(makespan=7.5)

    def test_rejects_nonfinite_schedule_values(self):
        for column, value in [(1, float("nan")), (2, float("inf")),
                              (2, float("-inf"))]:
            with self.subTest(column=column, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0][column] = value
                with self.assertRaises(ValueError):
                    self.solve_rows(rows=rows)
        for makespan in [float("nan"), float("inf")]:
            with self.subTest(makespan=makespan):
                with self.assertRaises(ValueError):
                    self.solve_rows(makespan=makespan)


class SongTimeTransitionTests(unittest.TestCase):
    def make_environment(self, n_machines):
        machines = [[0, 9, 3, 2], [0, 5, 5, 0], [0, 5, 2, 3]]
        if n_machines == 5:
            machines.append([0, 12, 8, 1])
        machines.append([1, 0, 0, n_machines - 1 if n_machines == 5 else 1])
        job_processing = torch.zeros((1, n_machines), dtype=torch.bool)
        for idle, _, _, job in machines:
            if not idle:
                job_processing[0, job] = True
        operation_steps = torch.zeros((1, n_machines), dtype=torch.int64)
        operation_steps[0, 3] = 1
        return SimpleNamespace(
            machines_batch=torch.tensor([machines], dtype=torch.float32),
            time=torch.tensor([2.0]),
            done_batch=torch.tensor([False]),
            feat_opes_batch=torch.full((1, 6, 8), 40.0),
            feat_mas_batch=torch.zeros((1, 3, n_machines)),
            mask_job_procing_batch=job_processing,
            mask_ma_procing_batch=torch.tensor([[not row[0] for row in machines]]),
            mask_job_finish_batch=torch.zeros((1, n_machines), dtype=torch.bool),
            ope_step_batch=operation_steps,
            end_ope_biases_batch=torch.zeros((1, n_machines), dtype=torch.int64),
        )

    def check_machine_completions(self, n_machines):
        env = self.make_environment(n_machines)
        before = env.machines_batch.clone()
        expected_jobs = env.mask_job_procing_batch.clone()
        expected_jobs[0, [0, 3]] = False
        expected_machine_mask = env.mask_ma_procing_batch.clone()
        expected_machine_mask[0, [1, 2]] = False

        advance_time(env, torch.tensor([0]))

        self.assertEqual(env.time.item(), 5)
        # Finishing a machine must preserve every finish time, usage and job ID.
        torch.testing.assert_close(env.machines_batch[:, :, 1:], before[:, :, 1:])
        expected_idle = before[:, :, 0].clone()
        expected_idle[0, [1, 2]] = 1
        torch.testing.assert_close(env.machines_batch[:, :, 0], expected_idle)
        torch.testing.assert_close(env.mask_job_procing_batch, expected_jobs)
        torch.testing.assert_close(env.mask_ma_procing_batch, expected_machine_mask)
        self.assertTrue(env.mask_job_finish_batch[0, 3].item())
        self.assertEqual(env.mask_job_finish_batch.sum().item(), 1)
        torch.testing.assert_close(
            env.feat_mas_batch[:, 2, :], before[:, :, 2].clamp(max=5) / (5 + 1e-5),
        )

    def test_four_machine_transition_preserves_machine_data(self):
        self.check_machine_completions(4)

    def test_five_machine_transition_preserves_machine_data(self):
        self.check_machine_completions(5)

    def test_eligible_or_done_environment_keeps_time_and_processing_masks(self):
        for done, flag in [(False, 1), (True, 0)]:
            with self.subTest(done=done, flag=flag):
                env = self.make_environment(5)
                env.done_batch[0] = done
                machines = env.machines_batch.clone()
                jobs = env.mask_job_procing_batch.clone()
                processing = env.mask_ma_procing_batch.clone()

                advance_time(env, torch.tensor([flag]))

                self.assertEqual(env.time.item(), 2)
                torch.testing.assert_close(env.machines_batch, machines)
                torch.testing.assert_close(env.mask_job_procing_batch, jobs)
                torch.testing.assert_close(env.mask_ma_procing_batch, processing)


if __name__ == "__main__":
    unittest.main()
