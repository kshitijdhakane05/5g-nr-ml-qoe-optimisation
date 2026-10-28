"""Focused safety and data-contract tests for the isolated TU Wien experiment."""

from __future__ import annotations

import unittest

from experiments.tuwien_kpi_baseline_20261026.baseline import (
    FEATURE_COLUMNS,
    EXPECTED_GROUPS,
    EXPECTED_REPETITIONS,
    EXPECTED_TRIALS,
    TARGETS,
    make_condition_folds,
    merge_trial_tables,
    run_cross_validation,
    target_rows,
)


def _fixture_tables():
    raw = []
    joined = []
    achieved = []
    for q in range(1, 8):
        for rate in (1, 10, 50, 100, 200, 500):
            for trial in range(1, 6):
                invalid = q == 1 and rate == 50 and trial in (3, 4, 5)
                key = {"qIdx": str(q), "trial": str(trial)}
                raw.append({
                    **key,
                    "target_Mbps": str(rate),
                    "valid_owd": "0" if invalid else "1",
                    "achieved_Mbps": str(rate + 0.5),
                    "latency_avg_ms": "9999999" if invalid else "10.0",
                    "jitter_ms": "0.2",
                    "loss_pct": "0.05",
                })
                joined.append({
                    **key,
                    "B_Mbps": str(rate),
                    "latency_corrupted_flag": "1" if invalid else "0",
                    "achieved_Mbps_server": str(rate + 0.5),
                    "jitter_ms_raw": "0.2",
                    "loss_pct_raw": "0.05",
                })
                achieved.append({
                    **key,
                    "B_target_Mbps": str(rate),
                    "achieved_Mbps_server": str(rate + 0.5),
                })
    return raw, joined, achieved


class TestTuwienKpiBaseline(unittest.TestCase):

    def setUp(self):
        self.raw, self.joined, self.achieved = _fixture_tables()

    def test_matching_keys_and_expected_group_counts(self):
        rows = merge_trial_tables(self.raw, self.joined, self.achieved)
        self.assertEqual(len(rows), EXPECTED_TRIALS)
        groups = {row["group_id"] for row in rows}
        self.assertEqual(len(groups), EXPECTED_GROUPS)
        self.assertEqual(
            {sum(row["group_id"] == group for row in rows) for group in groups},
            {EXPECTED_REPETITIONS},
        )

    def test_duplicate_key_fails_closed(self):
        raw = self.raw + [dict(self.raw[0])]
        with self.assertRaisesRegex(ValueError, "row count"):
            merge_trial_tables(raw, self.joined, self.achieved)
        # Keep the expected row count while introducing a duplicate.
        raw = [dict(row) for row in self.raw]
        raw[-1] = dict(raw[0])
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            merge_trial_tables(raw, self.joined, self.achieved)

    def test_invalid_latency_excluded_without_dropping_other_targets(self):
        rows = merge_trial_tables(self.raw, self.joined, self.achieved)
        self.assertEqual(len(target_rows(rows, "latency")), 207)
        for target in ("throughput", "jitter", "packet_loss"):
            self.assertEqual(len(target_rows(rows, target)), 210)
        invalid = [row for row in rows if not row["valid_owd"]]
        self.assertEqual(
            {(row["qIdx"], row["target_Mbps"], row["trial"]) for row in invalid},
            {("1", 50, 3), ("1", 50, 4), ("1", 50, 5)},
        )

    def test_predictor_contract_has_only_documented_pre_run_inputs(self):
        self.assertEqual(FEATURE_COLUMNS, ("target_Mbps", "qIdx"))
        self.assertEqual(set(TARGETS), {"throughput", "latency", "jitter", "packet_loss"})
        forbidden = {
            "trial", "achieved_Mbps", "latency_avg_ms", "jitter_ms", "loss_pct",
            "lost_pkts", "total_pkts", "valid_owd", "latency_corrupted_flag",
        }
        self.assertFalse(forbidden.intersection(FEATURE_COLUMNS))

    def test_condition_folds_keep_repetitions_together_and_use_all_groups(self):
        rows = merge_trial_tables(self.raw, self.joined, self.achieved)
        assignments = make_condition_folds(rows, seed=42)
        self.assertEqual(len(assignments), EXPECTED_GROUPS)
        for group in assignments:
            self.assertEqual(len({assignments[row["group_id"]] for row in rows if row["group_id"] == group}), 1)
        self.assertEqual(
            [sum(fold == assigned for assigned in assignments.values()) for fold in range(6)],
            [7] * 6,
        )

        predictions, _, _ = run_cross_validation(rows, seed=42)
        for target in TARGETS:
            by_group = {}
            trial_keys = set()
            for prediction in predictions:
                if prediction["target"] != target:
                    continue
                by_group.setdefault(prediction["group_id"], set()).add(prediction["fold"])
                trial_keys.add((prediction["qIdx"], prediction["target_Mbps"], prediction["trial"]))
            self.assertTrue(all(len(folds) == 1 for folds in by_group.values()))
            self.assertEqual(len(trial_keys), 207 if target == "latency" else 210)


if __name__ == "__main__":
    unittest.main()
