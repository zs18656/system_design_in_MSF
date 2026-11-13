import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thesis_forecasting.analyse import score_configurations, select_validation, selection_pools
from thesis_forecasting.bundle import load_bundle
from thesis_forecasting.candidates import fit_predict, registry, rollout
from thesis_forecasting.metrics import horizon_metrics
from thesis_forecasting.realism import opportunities, pareto_mask
from thesis_forecasting.surface import evaluate_search, variance_decomposition
from thesis_forecasting.tasks import split_windows
from thesis_forecasting.train import train

ROOT = Path(__file__).resolve().parents[1]


class TaskAndRolloutTest(unittest.TestCase):
    def test_chronological_alignment_and_scaling(self):
        values = np.arange(20, dtype=float)
        original, scaling = split_windows(values, 3, 2, [0.5, 0.25, 0.25])
        altered = values.copy()
        altered[10:] += 1000
        shifted, new_scaling = split_windows(altered, 3, 2, [0.5, 0.25, 0.25])
        self.assertEqual(scaling, new_scaling)
        self.assertEqual(scaling["mean"], 4.5)
        np.testing.assert_array_equal(original["train"].targets, shifted["train"].targets)
        np.testing.assert_array_equal(original["train"].history, shifted["train"].history)
        for split, (start, end) in zip(original.values(), [(3, 10), (10, 15), (15, 20)]):
            np.testing.assert_array_equal(split.origins, np.arange(start - 1, end - 2))
            np.testing.assert_array_equal(split.targets[:, 0], split.origins + 1)
            np.testing.assert_array_equal(split.targets[:, -1], split.origins + 2)
            np.testing.assert_allclose(split.history[:, -1] * scaling["scale"] + scaling["mean"],
                                       split.origins, atol=1e-12)

    def test_constant_training_series_has_unit_scale(self):
        _, scaling = split_windows(np.ones(20), 3, 2, [0.5, 0.25, 0.25])
        self.assertEqual(scaling["std"], 0.0)
        self.assertEqual(scaling["scale"], 1.0)

    def test_recmo_endpoints_and_nondivisor(self):
        history = np.array([[0, 1], [10, 11]])
        expected = history[:, -1:] + np.arange(1, 6)
        for size, expected_calls in [(1, 5), (5, 1), (3, 2), (2, 3)]:
            with self.subTest(size=size):
                windows = []

                def predict(context):
                    windows.append(context.copy())
                    block = context[:, -1:] + np.arange(1, size + 1)
                    return block.ravel() if size == 1 else block

                np.testing.assert_array_equal(rollout(predict, history, 5, size), expected)
                self.assertEqual(len(windows), expected_calls)
                self.assertTrue(all(window.shape == history.shape for window in windows))

    def test_seed_registry_keeps_independent_deployments(self):
        config = json.loads((ROOT / "configs" / "single_task.json").read_text())
        config["fit_seeds"] = [1, 2, 3]
        candidates = registry(config)
        self.assertEqual(len(candidates), 140)
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 140)
        self.assertEqual({row["fit_seed"] for row in candidates if row["architecture"] == "ridge"}, {1})
        bases = {row["fit_id"]: row for row in candidates if row["strategy"]["rectifier"] is None}
        self.assertEqual(len(bases), 28)
        for fit_id, base in bases.items():
            pairs = [row for row in candidates if row.get("base_fit_id") == fit_id]
            self.assertEqual({row["strategy"]["rectifier"]["block_size"] for row in pairs},
                             {1, 2, 4, 16})
            self.assertTrue(all(row["fit_seed"] == base["fit_seed"] for row in pairs))

    def test_rectifier_rollout_feedback_and_final_block(self):
        history = np.array([[2, 3], [4, 5]])
        base = np.array([[10, 20, 30, 40, 50], [20, 30, 40, 50, 60]])
        seen = []

        def predict(context):
            seen.append(context.copy())
            return context[:, -1:] + np.arange(2)

        corrected = rollout(predict, history, 5, 2, base_forecast=base)
        np.testing.assert_array_equal(corrected, [[7, 16, 14, 23, 27], [15, 24, 16, 25, 35]])
        np.testing.assert_array_equal(seen[1], corrected[:, :2])
        np.testing.assert_array_equal(seen[2], corrected[:, 2:4])

    def test_rectifier_fits_first_block_of_training_residuals_and_reuses_base(self):
        config = {"models": {"ridge": {}}, "block_sizes": [1], "rectifier_block_sizes": [2, 5],
                  "horizon": 5, "fit_seeds": [0]}
        candidates = registry(config)
        windows, scaling = split_windows(np.arange(60.0)**2, 3, 5, [0.5, 0.25, 0.25])
        train_windows = windows["train"]
        fit_targets = []

        class Base:
            def fit(self, history, targets):
                np.testing.assert_array_equal(history, train_windows.history)
                fit_targets.append(targets.copy())
                return self

            def predict(self, history):
                return history[:, -1] + 1

        class ZeroCorrection(Base):
            def fit(self, history, targets):
                super().fit(history, targets)
                self.size = targets.shape[1]
                return self

            def predict(self, inputs):
                return np.zeros((len(inputs), self.size))

        with patch("thesis_forecasting.candidates.Ridge",
                   side_effect=[Base(), ZeroCorrection(), ZeroCorrection()]):
            predictions = fit_predict(config, windows, scaling, candidates)
        normalized_targets = (train_windows.targets - scaling["mean"]) / scaling["scale"]
        base_train = train_windows.history[:, -1:] + np.arange(1, 6)
        residual = base_train - normalized_targets
        self.assertEqual(len(fit_targets), 3)
        np.testing.assert_allclose(fit_targets[0], normalized_targets[:, 0])
        np.testing.assert_allclose(fit_targets[1], residual[:, :2])
        np.testing.assert_allclose(fit_targets[2], residual)
        for split, array in predictions.items():
            self.assertEqual(array.shape, (3, len(windows[split].origins), 5))
            np.testing.assert_array_equal(array[0], array[1])
            np.testing.assert_array_equal(array[0], array[2])


class AnalysisTest(unittest.TestCase):
    def test_hand_computed_metrics(self):
        target = [[0, 0], [2, 2]]
        prediction = [[2, 1], [0, 3]]
        metrics = horizon_metrics(target, prediction)
        np.testing.assert_allclose(metrics["mse"], [4, 1])
        np.testing.assert_allclose(metrics["mae"], [2, 1])
        np.testing.assert_allclose(metrics["w1"], [0, 1])
        with self.assertRaises(ValueError):
            horizon_metrics([], [])

    def test_selection_averages_seeds_and_ignores_test_targets(self):
        config = json.loads((ROOT / "configs" / "tiny.json").read_text())
        config.update(block_sizes=[1], rectifier_block_sizes=[], fit_seeds=[0, 1])
        candidates = registry(config)
        predictions = np.sqrt([2, 0, 6]).reshape(3, 1, 1)
        val = horizon_metrics([[0]], predictions)
        for label in [0, 100]:
            rows, _ = score_configurations(candidates, {
                "val": val, "test": horizon_metrics([[label]], predictions)
            }, "mse")
            self.assertEqual(select_validation(rows, {"ridge-recmo-1", "mlp-recmo-1"})
                             ["configuration_id"], "ridge-recmo-1")
            chosen = select_validation(rows, {"mlp-recmo-1"})
            self.assertAlmostEqual(chosen["val_loss"], 3)
            self.assertEqual(chosen["fit_count"], 2)
            self.assertAlmostEqual(chosen["test_loss"], np.mean((predictions[1:] - label)**2))
            self.assertIsNone(select_validation(rows, set()))

    def test_expanded_selection_is_nested_and_ignores_test_targets(self):
        candidates = registry({"models": {"ridge": {}}, "block_sizes": [1, 2],
                               "rectifier_block_sizes": [1], "horizon": 2, "fit_seeds": [0]})
        predictions = np.array([2, 1, 3, 4], dtype=float).reshape(4, 1, 1)
        validation = horizon_metrics([[0]], predictions)
        for label in [0, 100]:
            rows, _ = score_configurations(candidates, {
                "val": validation, "test": horizon_metrics([[label]], predictions)
            }, "mse")
            pools = selection_pools(rows, {})
            self.assertEqual(len(pools["unrectified_recmo"]), 2)
            self.assertEqual(len(pools["expanded_recmo_recmo"]), 4)
            self.assertEqual(select_validation(rows, pools["unrectified_recmo"])["configuration_id"],
                             "ridge-recmo-1")
            self.assertEqual(select_validation(rows, pools["expanded_recmo_recmo"])["configuration_id"],
                             "ridge-recmo-1-rect-recmo-1")

    def test_pareto_keeps_ties_and_tradeoffs(self):
        np.testing.assert_array_equal(pareto_mask([1, 1, 2, 3, 4], [5, 5, 4, 4, 3]),
                                      [True, True, True, False, True])
        self.assertEqual(pareto_mask([], []).size, 0)

    def test_opportunity_denominators_and_zero_baselines(self):
        ids = ["base", "alternative"]
        rows, summary = opportunities(ids, [[1, 1, 1, 0], [1.04, 2, 1.05, 0]],
                                      [[10, 10, 10, 0], [8, 0, 10, 0]], 0.05)
        self.assertEqual([r["eligible"] for r in rows], [True, False, True, True])
        self.assertEqual([r["success"] for r in rows], [True, False, False, False])
        self.assertIsNone(rows[3]["realism_gain"])
        self.assertEqual(rows[3]["mse_cost"], 0)
        self.assertAlmostEqual(rows[0]["mse_cost"], 0.04)
        self.assertAlmostEqual(summary["median_eligible_realism_gain"], 0.1)
        self.assertEqual(summary["success_rate"], 0.25)
        _, summary = opportunities(["base"], [[1, 1]], [[1, 1]], 0.05)
        self.assertIsNone(summary["median_eligible_realism_gain"])
        self.assertEqual(summary["success_rate"], 0)
        _, summary = opportunities([], np.empty((0, 2)), np.empty((0, 2)), 0.05)
        self.assertEqual(summary["evaluated_horizons"], 2)
        self.assertEqual(summary["success_rate"], 0)

    def test_variance_and_search_use_correct_partitions(self):
        result = variance_decomposition([[1, 0], [0, 1]])
        self.assertEqual(result["fraction"]["interaction"], 1)
        self.assertIsNone(variance_decomposition([[1, 1]])["fraction"]["interaction"])
        validation = [[[1], [2]], [[3], [4]]]
        test = np.array([[1, 2], [3, 4]])
        original = evaluate_search(validation, test, ["a", "b"], [1, 2], 7, 3)
        changed = evaluate_search(validation, test[::-1], ["a", "b"], [1, 2], 7, 3)
        fields = ("architecture", "block_size", "selected_val_loss", "queried_architecture",
                  "queried_block_size")
        self.assertEqual([[row[k] for k in fields] for row in original],
                         [[row[k] for k in fields] for row in changed])
        self.assertEqual({(r["architecture"], r["block_size"]) for r in original if r["budget"] == 4},
                         {("a", 1)})


class SmokeTest(unittest.TestCase):
    def test_tiny_fit_reload_reproducible_and_analysis_without_fitting(self):
        config = ROOT / "configs" / "tiny.json"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first, second = tmp / "first", tmp / "second"
            train(config, first)
            train(config, second)
            meta, data = load_bundle(first)
            new_meta, new_data = load_bundle(second)
            self.assertEqual(meta, new_meta)
            self.assertEqual(len(meta["candidates"]), 12)
            for split, count in [("train", 537), ("val", 73), ("test", 153)]:
                origins, targets, predictions = data[split]
                self.assertEqual(origins.shape, (count,))
                self.assertEqual(predictions.shape, (12, count, 8))
                self.assertEqual(targets.shape, predictions.shape[1:])
                self.assertTrue(np.isfinite(predictions).all())
                np.testing.assert_array_equal(origins, new_data[split][0])
                np.testing.assert_array_equal(predictions, new_data[split][2])

            # A fresh process forbids importing model fitting dependencies.
            script = """
import sys
sys.modules['sklearn'] = None
sys.modules['thesis_forecasting.candidates'] = None
sys.modules['thesis_forecasting.train'] = None
from thesis_forecasting.analyse import main
main()
"""
            results = tmp / "analysis"
            subprocess.run([sys.executable, "-c", script, "--bundle", str(first),
                            "--out", str(results)], cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads((results / "summary.json").read_text())
            self.assertEqual(summary["configuration_count"], 12)
            selected = summary["chapter_5_validation_selection"]
            self.assertEqual(selected["unrectified_recmo"]["configuration_count"], 4)
            self.assertEqual(selected["expanded_recmo_recmo"]["configuration_count"], 12)
            self.assertEqual(summary["chapter_6_test_surface"]["unrectified_configuration_count"], 4)
            with (results / "top_concentration.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            with (results / "budget_search.csv").open() as stream:
                self.assertEqual(max(int(row["budget"]) for row in csv.DictReader(stream)), 4)
            with (results / "horizon_pareto.csv").open() as stream:
                frontier = list(csv.DictReader(stream))
            self.assertEqual(len(frontier), 12 * 8)
            self.assertEqual(sum(bool(row["rectifier_block_size"]) for row in frontier), 8 * 8)
            self.assertEqual(summary["chapter_7_retrospective_opportunity"]["evaluated_horizons"], 8)
            self.assertEqual(len(list(results.glob("*.png"))), 3)
            metrics = first / "metrics.csv"
            metrics.write_text(metrics.read_text() + "edited")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_bundle(first)
