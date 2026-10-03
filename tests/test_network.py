"""Checks for the custom loss, training, CV isolation, and saved predictions."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import network


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        network.set_seed(0)
        self.device = torch.device("cpu")
        self.config = {**network.experiment_configs()[0], "hidden_sizes": [8],
                       "batch_size": 8}
        self.X = np.random.default_rng(0).normal(size=(40, 3))
        self.y = np.arange(40) % 2

    def test_custom_loss_values_gradients_and_normalization(self):
        logits = torch.tensor([[10000., 9999., -10000.], [-10000., 9999., 10000.]],
                              dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 2])
        reference_logits = logits.detach().clone().requires_grad_()
        actual = network.cross_entropy(logits, labels)
        # Built-in loss is used only as an independent test oracle.
        expected = torch.nn.functional.cross_entropy(reference_logits, labels)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(logits.grad, reference_logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        duplicated = network.cross_entropy(logits.repeat(3, 1), labels.repeat(3))
        torch.testing.assert_close(actual, duplicated)

    def test_scaling_uses_training_statistics_and_handles_constant_features(self):
        train = np.array([[0., 5.], [2., 5.], [4., 5.]])
        mean, scale = network.fit_scaler(train)
        scaled = network.transform(train, mean, scale)
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(scaled[:, 0].std(), 1, atol=1e-7)
        self.assertTrue(np.isfinite(scaled).all())
        validation = network.transform([[100., 5.]], mean, scale)
        self.assertGreater(validation[0, 0], 50)
        np.testing.assert_array_equal(mean, [2., 5.])

    def test_train_updates_weights_and_eval_preserves_batchnorm_state(self):
        model = network.build_model(3, 2, {**self.config, "dropout": 0.3}, self.device)
        before = model.output.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = network.train_epoch(model, network.make_loader(
            self.X, self.y, batch_size=8, training=True), optimizer, self.device)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.output.weight))
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        first = network.predict(model, self.X, self.device)
        second = network.predict(model, self.X, self.device)
        np.testing.assert_array_equal(first, second)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name])
        self.assertFalse(model.training)

    def test_loss_logging_weights_unequal_batches_correctly(self):
        model = network.build_model(3, 2, self.config, self.device)
        loader = network.make_loader(self.X[:11], self.y[:11], batch_size=8)
        metrics, predictions = network.evaluate(model, loader, self.device)
        with torch.no_grad():
            expected = network.cross_entropy(model(torch.tensor(
                self.X[:11], dtype=torch.float32)), torch.tensor(self.y[:11])).item()
        self.assertAlmostEqual(metrics["loss"], expected, places=6)
        self.assertEqual(len(predictions), 11)

    def test_singleton_last_batch_is_safe_for_batchnorm(self):
        loader = network.make_loader(self.X[:17], self.y[:17], batch_size=8, training=True)
        self.assertEqual([len(y) for _, y in loader], [8, 8])
        evaluation = network.make_loader(self.X[:17], self.y[:17], batch_size=8)
        self.assertEqual([len(y) for _, y in evaluation], [8, 8, 1])

    def test_cv_scaler_sees_only_training_rows_and_writes_logs(self):
        folds = network.make_folds(self.y, 2)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(40))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("network.fit_scaler", wraps=network.fit_scaler) as scaler:
                with contextlib.redirect_stdout(io.StringIO()):
                    records, histories, predictions = network.cross_validate(
                        self.X, self.y, self.config, folds, 1, self.device,
                        root / "logs", root)
            for i, call in enumerate(scaler.call_args_list):
                training_rows = np.concatenate([f for j, f in enumerate(folds) if j != i])
                np.testing.assert_array_equal(call.args[0], self.X[training_rows])
            self.assertEqual(len(records), 2)
            self.assertEqual(len(histories), 2)
            self.assertEqual(predictions.shape, self.y.shape)
            self.assertEqual(len(list(root.glob("logs/**/events.out.tfevents.*"))), 2)
            self.assertEqual(len(list(root.glob("histories/**/*.csv"))), 2)

    def test_checkpoint_roundtrip_and_csv_format(self):
        mean, scale = network.fit_scaler(self.X)
        X_scaled = network.transform(self.X, mean, scale)
        model = network.build_model(3, 2, self.config, self.device)
        columns = ["a", "b", "c"]
        test_df = pd.DataFrame(self.X[::-1], columns=columns)[["c", "a", "b"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.pt"
            network.save_checkpoint(path, model, self.config, mean, scale,
                                    ["SEKER", "SIRA"], columns, 1, 0.5)
            loaded, checkpoint = network.load_checkpoint(path, self.device)
            np.testing.assert_array_equal(network.predict(model, X_scaled, self.device),
                                          network.predict(loaded, X_scaled, self.device))
            output = Path(directory) / "network.csv"
            network.save_predictions(loaded, checkpoint, test_df, output, self.device)
            result = pd.read_csv(output)
            pd.testing.assert_frame_equal(result.drop(columns="Target"), test_df)
            expected = np.asarray(checkpoint["classes"])[network.predict(
                loaded, X_scaled[::-1].copy(), self.device)]
            np.testing.assert_array_equal(result["Target"], expected)
            self.assertEqual(result.columns.tolist(), ["c", "a", "b", "Target"])
            with self.assertRaises(ValueError):
                network.save_predictions(loaded, checkpoint, test_df.drop(columns="a"),
                                         output, self.device)


if __name__ == "__main__":
    unittest.main()
