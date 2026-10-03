# Cross validation Balance Accuracy = 94.5233 %
"""Practical 5: a custom PyTorch MLP for the Dry Bean dataset.

Run python network.py to compare models with manual five-fold CV, retrain the
winner, save/reload its state_dict, and write network.csv. All helpers are here.
Run tensorboard --logdir artifacts/network/tensorboard to view live losses.

Recorded result: supplied training data, 60 epochs, five folds, seed 0, CPU.
Selected model: hidden layers [128, 64], ReLU, dropout 0.1, weight decay 0.0001,
Adam learning rate 0.001, batch size 128. CV is used for model selection.
Use --predict-only to regenerate predictions from the saved checkpoint.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

DATA_DIR = Path(__file__).resolve().parent
SEED = 0
K = 5
EPOCHS = 60
LABEL_COL = "Class"


class BeanDataset(Dataset):
    """Numeric features and integer class indices; labels are optional at inference."""

    def __init__(self, X, y=None):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
        if self.X.ndim != 2 or (self.y is not None and
                               (self.y.ndim != 1 or len(self.X) != len(self.y))):
            raise ValueError("Expected 2D X and one label per row")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index] if self.y is None else (self.X[index], self.y[index])


class BeanMLP(nn.Module):
    """Linear -> BatchNorm -> activation -> optional dropout for each hidden layer."""

    def __init__(self, n_features, n_classes, hidden_sizes=(64,),
                 activation="relu", dropout=0.0):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
        if not hidden_sizes or any(width < 1 for width in hidden_sizes):
            raise ValueError("At least one positive hidden-layer width is required")
        layers = []
        previous = n_features
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.BatchNorm1d(width),
                           activations[activation]()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(previous, n_classes)

    def forward(self, X):
        # Return logits; the custom loss performs probability normalization.
        return self.output(self.hidden(X))


def cross_entropy(logits, targets):
    """Mean negative log probability, implemented without a built-in loss.

    Subtracting the row maximum avoids exp overflow. The exponential sum
    normalizes each row over classes, and mean() normalizes by batch size.
    """
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_normalizer = shifted.exp().sum(dim=1).log()
    target_scores = shifted[torch.arange(len(targets), device=logits.device), targets]
    return (log_normalizer - target_scores).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu")
    return torch.device(name)


def fit_scaler(X):
    """Fit only on training rows, never on validation or test rows."""
    X = np.asarray(X, dtype=np.float64)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def transform(X, mean, scale):
    return ((np.asarray(X, dtype=np.float64) - mean) / scale).astype(np.float32)


def make_folds(y, k=K, seed=SEED):
    """Hand-written stratified partitions; no sklearn CV/split functions."""
    y = np.asarray(y)
    if y.ndim != 1 or not isinstance(k, int) or not 2 <= k <= len(y):
        raise ValueError("Use 1D labels and 2 <= k <= number of samples")
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < k:
        raise ValueError("Each class needs at least k samples")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in classes:
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for i, part in enumerate(np.array_split(indices, k)):
            folds[i].extend(part)
    return [np.asarray(fold, dtype=int) for fold in folds]


def make_loader(X, y=None, batch_size=128, training=False, seed=SEED):
    if batch_size < 2 or (training and len(X) < 2):
        raise ValueError("Training and batch size must allow at least two samples")
    # BatchNorm cannot estimate variance from a singleton training batch.
    # Shuffle each epoch; omit only the last row if it would be a singleton.
    return DataLoader(BeanDataset(X, y), batch_size=batch_size, shuffle=training,
                      drop_last=training and len(X) % batch_size == 1,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, count = 0.0, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
        # Weight batch means by actual batch size, including the final batch.
        total_loss += loss.item() * len(targets)
        count += len(targets)
    return total_loss / count


def evaluate(model, loader, device):
    model.eval()
    total_loss, count = 0.0, 0
    predictions, labels = [], []
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            total_loss += cross_entropy(logits, targets).item() * len(targets)
            count += len(targets)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    y, pred = np.concatenate(labels), np.concatenate(predictions)
    return {"loss": total_loss / count,
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0)}, pred


def build_model(n_features, n_classes, config, device):
    return BeanMLP(n_features, n_classes, hidden_sizes=config["hidden_sizes"],
                   activation=config["activation"], dropout=config["dropout"]).to(device)


def fit_network(X, y, config, n_classes, epochs, device, seed,
                log_dir, history_path, validation=None):
    """Train for a fixed epoch budget, logging live TensorBoard loss curves.

    Validation is for evaluation only: no epoch is chosen using the held-out
    fold. Every candidate gets the same budget and fold-specific random seed.
    """
    set_seed(seed)
    model = build_model(X.shape[1], n_classes, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    train_loader = make_loader(X, y, config["batch_size"], training=True, seed=seed)
    val_loader = None if validation is None else make_loader(*validation)
    history = []
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(log_dir), flush_secs=5) as writer:
        for epoch in range(1, epochs + 1):
            row = {"epoch": epoch,
                   "train_loss": train_epoch(model, train_loader, optimizer, device)}
            writer.add_scalar("loss/train", row["train_loss"], epoch)
            if val_loader is not None:
                metrics, _ = evaluate(model, val_loader, device)
                row.update({f"val_{name}": value for name, value in metrics.items()})
                writer.add_scalar("loss/validation", metrics["loss"], epoch)
                writer.add_scalar("balanced_accuracy/validation",
                                  metrics["balanced_accuracy"], epoch)
            history.append(row)
            # CSV remains readable during training, as do TensorBoard events.
            pd.DataFrame(history).to_csv(history_path, index=False)
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                detail = (f" | val loss {row['val_loss']:.4f} | "
                          f"val balanced accuracy {row['val_balanced_accuracy']:.4f}"
                          if val_loader is not None else "")
                print(f"  {config['name']} epoch {epoch}/{epochs}: "
                      f"train loss {row['train_loss']:.4f}{detail}", flush=True)
                writer.flush()
    return model, history


def experiment_configs():
    baseline = {"name": "baseline", "hidden_sizes": [64], "activation": "relu",
                "dropout": 0.0, "weight_decay": 0.0,
                "learning_rate": 0.001, "batch_size": 128}
    return [baseline,
            {**baseline, "name": "dropout", "dropout": 0.15},
            {**baseline, "name": "weight_decay", "weight_decay": 0.001},
            {**baseline, "name": "deeper", "hidden_sizes": [128, 64],
             "dropout": 0.1, "weight_decay": 0.0001},
            {**baseline, "name": "tanh", "activation": "tanh", "dropout": 0.1,
             "weight_decay": 0.0001, "learning_rate": 0.0005, "batch_size": 256}]


def cross_validate(X, y, config, folds, epochs, device, log_root, artifact_dir):
    records, histories = [], []
    oof_predictions = np.empty(len(y), dtype=int)
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        mean, scale = fit_scaler(X[train_idx])
        X_train = transform(X[train_idx], mean, scale)
        X_val = transform(X[val_idx], mean, scale)
        print(f"\n{config['name']} - fold {i + 1}/{len(folds)}", flush=True)
        model, history = fit_network(
            X_train, y[train_idx], config, len(np.unique(y)), epochs, device, SEED + i,
            log_root / config["name"] / f"fold_{i + 1}",
            artifact_dir / "histories" / config["name"] / f"fold_{i + 1}.csv",
            validation=(X_val, y[val_idx]))
        val_scores, oof_predictions[val_idx] = evaluate(
            model, make_loader(X_val, y[val_idx]), device)
        train_scores, _ = evaluate(model, make_loader(X_train, y[train_idx]), device)
        records.append({"config": config["name"], "fold": i + 1, **val_scores,
                        "train_balanced_accuracy": train_scores["balanced_accuracy"]})
        histories.extend({"config": config["name"], "fold": i + 1, **row}
                         for row in history)
    return records, histories, oof_predictions


def predict(model, X, device):
    model.eval()
    if len(X) == 0:
        return np.empty(0, dtype=int)
    predictions = []
    with torch.no_grad():
        for features in make_loader(X):
            predictions.append(model(features.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def save_checkpoint(path, model, config, mean, scale, classes, feature_columns,
                    epochs, cv_score):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {name: value.detach().cpu()
                               for name, value in model.state_dict().items()},
                "config": config, "mean": torch.tensor(mean), "scale": torch.tensor(scale),
                "classes": [str(label) for label in classes],
                "feature_columns": list(feature_columns), "epochs": epochs,
                "seed": SEED, "cv_balanced_accuracy": float(cv_score),
                "torch_version": str(torch.__version__)}, path)


def load_checkpoint(path, device):
    # Load tensor data and metadata, not a pickled model object.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(len(checkpoint["feature_columns"]), len(checkpoint["classes"]),
                        checkpoint["config"], device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def save_predictions(model, checkpoint, test_df, output_path, device):
    columns = checkpoint["feature_columns"]
    if set(test_df.columns) != set(columns):
        raise ValueError("Test columns must match the training feature columns")
    X = transform(test_df.loc[:, columns].to_numpy(),
                  checkpoint["mean"].numpy(), checkpoint["scale"].numpy())
    result = test_df.copy()
    result["Target"] = np.asarray(checkpoint["classes"])[predict(model, X, device)]
    result.to_csv(output_path, index=False)
    return result


def plot_histories(histories, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frame = pd.DataFrame(histories)
    names = list(frame["config"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        averages = frame[frame["config"] == name].groupby("epoch").mean(numeric_only=True)
        ax.plot(averages.index, averages["train_loss"], label="Training", color="#2563eb")
        ax.plot(averages.index, averages["val_loss"], label="Validation", color="#ea580c")
        ax.set(title=name.replace("_", " ").title(), xlabel="Epoch",
               ylabel="Mean cross-entropy", ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Dry Bean MLP: mean losses across validation folds", fontsize=16)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", nargs="?", type=Path, default=DATA_DIR / "dry_bean_train.csv")
    parser.add_argument("test_csv", nargs="?", type=Path, default=DATA_DIR / "dry_bean_test.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "network.csv")
    parser.add_argument("--artifacts", type=Path, default=DATA_DIR / "artifacts" / "network")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--folds", type=int, default=K)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--predict-only", action="store_true", help="Reuse artifacts/network.pt")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    # Small tabular networks are fast and repeatable on one CPU thread.
    torch.set_num_threads(1)
    device = choose_device(args.device)
    test_df = pd.read_csv(args.test_csv)
    checkpoint_path = args.artifacts / "network.pt"
    if args.predict_only:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        result = save_predictions(model, checkpoint, test_df, args.output, device)
        print(f"Loaded {checkpoint_path}; wrote {len(result)} rows to {args.output}")
        return

    train_df = pd.read_csv(args.train_csv)
    feature_columns = train_df.drop(columns=LABEL_COL).columns.tolist()
    if set(test_df.columns) != set(feature_columns):
        raise ValueError("Test columns must match the training feature columns")
    X = train_df[feature_columns].to_numpy(dtype=np.float64)
    classes, y = np.unique(train_df[LABEL_COL].to_numpy(), return_inverse=True)
    if not np.isfinite(X).all() or not np.isfinite(test_df.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite numeric values")
    folds = make_folds(y, args.folds)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    log_root = args.artifacts / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    print(f"Training {X.shape} on {device}; {args.folds} folds, {args.epochs} epochs")
    print(f"Live curves: tensorboard --logdir {args.artifacts / 'tensorboard'}", flush=True)
    records, histories, summaries = [], [], []
    best_score, best_config, best_oof = -1.0, None, None
    for config in experiment_configs():
        rows, curves, oof = cross_validate(X, y, config, folds, args.epochs,
                                          device, log_root, args.artifacts)
        records.extend(rows)
        histories.extend(curves)
        scores = np.array([row["balanced_accuracy"] for row in rows])
        summary = {"config": config["name"], "balanced_accuracy": float(scores.mean()),
                   "balanced_accuracy_std": float(scores.std()),
                   "accuracy": float(np.mean([r["accuracy"] for r in rows])),
                   "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                   "train_balanced_accuracy": float(np.mean(
                       [r["train_balanced_accuracy"] for r in rows]))}
        summaries.append(summary)
        pd.DataFrame(records).to_csv(args.artifacts / "cv_folds.csv", index=False)
        pd.DataFrame(summaries).to_csv(args.artifacts / "cv_comparison.csv", index=False)
        plot_histories(histories, args.artifacts / "loss_curves.png")
        print(f"\n{config['name']}: balanced accuracy {scores.mean():.4f} "
              f"+/- {scores.std():.4f} | train {summary['train_balanced_accuracy']:.4f}",
              flush=True)
        if scores.mean() > best_score:
            best_score, best_config, best_oof = float(scores.mean()), config, oof

    print(f"\nSelected configuration: {best_config}")
    print(f"# Cross validation Balance Accuracy = {100 * best_score:.4f} %")
    print("CV was used for model selection; this is not an independent test score.")
    pd.DataFrame({"row_index": np.arange(len(y)), "Class": classes[y],
                  "Prediction": classes[best_oof]}).to_csv(
                      args.artifacts / "out_of_fold_predictions.csv", index=False)
    summary = {"selected_config": best_config, "cv_balanced_accuracy": best_score,
               "folds": args.folds, "epochs": args.epochs, "seed": SEED,
               "device": str(device), "torch_version": str(torch.__version__),
               "comparisons": summaries}
    (args.artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Final model uses all labelled rows and exactly the same epoch budget.
    mean, scale = fit_scaler(X)
    X_scaled = transform(X, mean, scale)
    model, _ = fit_network(X_scaled, y, best_config, len(classes), args.epochs,
                           device, SEED, log_root / "final",
                           args.artifacts / "histories" / "final.csv")
    save_checkpoint(checkpoint_path, model, best_config, mean, scale, classes,
                    feature_columns, args.epochs, best_score)
    reloaded, checkpoint = load_checkpoint(checkpoint_path, device)
    # Exercise save/load before producing the submission.
    np.testing.assert_array_equal(predict(model, X_scaled, device),
                                  predict(reloaded, X_scaled, device))
    result = save_predictions(reloaded, checkpoint, test_df, args.output, device)
    print(f"Saved/reloaded {checkpoint_path}; wrote {len(result)} rows to {args.output}")
    print("Predicted class counts:", result["Target"].value_counts().to_dict())


if __name__ == "__main__":
    main()
