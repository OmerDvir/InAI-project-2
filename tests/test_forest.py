"""Run with: python -B -m unittest discover -s tests -v."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forest import ForestClassifier, cross_validate, make_folds, save_predictions


class ForestTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(48).reshape(24, 2)
        self.y = np.array(["SEKER"] * 12 + ["SIRA"] * 12)

    def test_bootstrap_samples_and_distinct_tree_seeds(self):
        with patch("forest.DecisionTreeClassifier") as tree:
            ForestClassifier(n_estimators=3).fit(self.X, self.y)
        samples = [call.args[0] for call in tree.return_value.fit.call_args_list]
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertEqual(sample.shape, self.X.shape)
            self.assertLess(len(np.unique(sample, axis=0)), len(self.X))
            self.assertTrue(set(sample[:, 0]) <= set(self.X[:, 0]))
        self.assertFalse(np.array_equal(samples[0], samples[1]))
        seeds = [call.kwargs["random_state"] for call in tree.call_args_list]
        self.assertEqual(len(set(seeds)), 3)
        self.assertTrue(all(call.kwargs["max_features"] == "sqrt"
                            for call in tree.call_args_list))

    def test_predictions_match_manual_vote_and_are_reproducible(self):
        model = ForestClassifier(n_estimators=9, max_depth=3).fit(self.X, self.y)
        tree_predictions = np.array([tree.predict(self.X) for tree in model.trees_])
        expected = [sorted(Counter(row).items(), key=lambda p: (-p[1], p[0]))[0][0]
                    for row in tree_predictions.T]
        np.testing.assert_array_equal(model.predict(self.X), expected)
        other = ForestClassifier(n_estimators=9, max_depth=3).fit(self.X, self.y)
        np.testing.assert_array_equal(model.predict(self.X), other.predict(self.X))
        model.fit(self.X, self.y)
        self.assertEqual(len(model.trees_), 9)

    def test_vote_handles_ties_and_trees_missing_classes(self):
        class FixedTree:
            def __init__(self, label):
                self.label = label

            def predict(self, X):
                return np.repeat(self.label, len(X))

        model = ForestClassifier(n_estimators=1).fit(self.X, self.y)
        model.trees_ = [FixedTree("SIRA"), FixedTree("SEKER")]
        self.assertTrue((model.predict(self.X) == "SEKER").all())
        model.trees_.append(FixedTree("SIRA"))
        self.assertTrue((model.predict(self.X) == "SIRA").all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ForestClassifier(n_estimators=0)
        with self.assertRaises(ValueError):
            ForestClassifier().predict(self.X)
        with self.assertRaises(ValueError):
            ForestClassifier().fit(self.X, self.y[:-1])
        model = ForestClassifier(n_estimators=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.predict(self.X[:, :1])
        self.assertEqual(model.predict(np.empty((0, 2))).shape, (0,))

    def test_stratified_folds_cover_each_row_once(self):
        y = np.array(["A"] * 13 + ["B"] * 7)
        folds = make_folds(y, 5)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(y)))
        for label in np.unique(y):
            counts = [np.sum(y[fold] == label) for fold in folds]
            self.assertLessEqual(max(counts) - min(counts), 1)
        for a, b in zip(folds, make_folds(y, 5)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            make_folds(y, 1)
        with self.assertRaises(ValueError):
            make_folds(y, 8)

    def test_cv_excludes_validation_rows_and_reports_balanced_accuracy(self):
        fitted_models = []

        class RecordingClassifier:
            def __init__(self, random_state):
                self.calls = []
                fitted_models.append(self)

            def fit(self, X, y):
                self.training_rows = set(X[:, 0])
                return self

            def predict(self, X):
                self.calls.append(set(X[:, 0]))
                return np.repeat("A", len(X))

        X = np.arange(16).reshape(-1, 1)
        y = np.array(["A"] * 12 + ["B"] * 4)
        scores = cross_validate({}, X, y, k=4, model_class=RecordingClassifier)
        self.assertEqual(len(fitted_models), 4)
        validation_rows = []
        for model in fitted_models:
            self.assertTrue(model.training_rows.isdisjoint(model.calls[0]))
            self.assertEqual(model.training_rows | model.calls[0], set(range(16)))
            self.assertEqual(model.training_rows, model.calls[1])
            validation_rows.extend(model.calls[0])
        self.assertEqual(sorted(validation_rows), list(range(16)))
        np.testing.assert_allclose(scores["balanced_accuracy"], 0.5)
        np.testing.assert_allclose(scores["accuracy"], 0.75)

    def test_csv_preserves_rows_columns_and_feature_alignment(self):
        model = ForestClassifier(n_estimators=5).fit(self.X, self.y)
        # Test columns can arrive in a different order from training features.
        test_df = pd.DataFrame({"second": self.X[::-1, 1], "first": self.X[::-1, 0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forest.csv"
            save_predictions(model, test_df, ["first", "second"], path)
            result = pd.read_csv(path)
            pd.testing.assert_frame_equal(result.drop(columns="Target"), test_df)
            np.testing.assert_array_equal(result["Target"], model.predict(self.X[::-1]))
            self.assertEqual(result.columns.tolist(), ["second", "first", "Target"])
            with self.assertRaises(ValueError):
                save_predictions(model, test_df, ["missing"], path)


if __name__ == "__main__":
    unittest.main()
