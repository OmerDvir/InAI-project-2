# Cross validation Balance Accuracy = 93.5626 %
"""Practical 4: Dry Bean trees and a hand-written forest.

Bootstrap sampling, voting, stratified CV, and parameter selection are manual.
Only individual trees and evaluation metrics come from sklearn.
The recorded score uses the supplied training CSV, five folds, and SEED=0.
Selected: 200 trees, max_features=0.5, max_depth=None, min_samples_leaf=1.
Run: python forest.py [train.csv] [test.csv] [--output forest.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

DATA_DIR = Path(__file__).resolve().parent
TRAIN_CSV = DATA_DIR / "dry_bean_train.csv"
TEST_CSV = DATA_DIR / "dry_bean_test.csv"
LABEL_COL = "Class"
PRED_CSV = DATA_DIR / "forest.csv"
K = 5
SEED = 0
SELECT_BY = "balanced_accuracy"


# ---------- our own forest ----------
class ForestClassifier:
    """Bootstrap decision trees and classify by majority vote.

    Each tree trains on a random sample of rows drawn with replacement.
    max_features controls the random feature subset considered at each split.
    """

    def __init__(self, n_estimators=100, max_features="sqrt",
                 random_state=SEED, **tree_params):
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.tree_params = tree_params

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("Expected nonempty 2D X and matching 1D y")

        self.n_features_in_ = X.shape[1]
        # Global class indices keep votes aligned if a sample omits a class.
        self.classes_ = np.unique(y)
        self.trees_ = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            sample_idx = rng.integers(0, len(y), size=len(y))
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                **self.tree_params,
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        if not getattr(self, "trees_", None):
            raise ValueError("Call fit before predict")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same features as the training data")
        if len(X) == 0:
            return self.classes_[:0]

        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        rows = np.arange(len(X))
        for tree in self.trees_:
            predicted_class = np.searchsorted(self.classes_, tree.predict(X))
            votes[rows, predicted_class] += 1
        # Ties go to the first class in sorted order for reproducibility.
        return self.classes_[votes.argmax(axis=1)]


# ---------- data ----------
def load(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[LABEL_COL]).to_numpy()
    y = df[LABEL_COL].to_numpy()
    return X, y


# ---------- our own CV ----------
def make_folds(y, k, seed=SEED):
    """Stratified folds: split each class's indices into k parts separately,
    then fold i = part i of every class. Returns a list of k index arrays."""
    y = np.asarray(y)
    if y.ndim != 1 or not isinstance(k, int) or not 2 <= k <= len(y):
        raise ValueError("Use 1D labels and 2 <= k <= number of samples")
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < k:
        raise ValueError("Each class needs at least k samples for stratified CV")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].extend(part)
    return [np.array(f, dtype=int) for f in folds]


def cross_validate(params, X, y, k=K, model_class=ForestClassifier):
    """Return per-fold scores, fitting a fresh model on training rows only."""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Expected 2D X with one row per label")
    folds = make_folds(y, k)
    scores = {name: [] for name in (
        "balanced_accuracy", "accuracy", "macro_f1", "train_balanced_accuracy"
    )}
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        # Bootstrap sampling happens inside fit, after validation rows are removed.
        clf = model_class(random_state=SEED + i, **params)
        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[val_idx])
        scores["balanced_accuracy"].append(balanced_accuracy_score(y[val_idx], pred))
        scores["accuracy"].append(accuracy_score(y[val_idx], pred))
        scores["macro_f1"].append(f1_score(y[val_idx], pred, average="macro"))
        scores["train_balanced_accuracy"].append(
            balanced_accuracy_score(y[train_idx], clf.predict(X[train_idx]))
        )
    return {metric: np.array(values) for metric, values in scores.items()}


def report(name, params, X, y, model_class=ForestClassifier):
    scores = cross_validate(params, X, y, model_class=model_class)
    balanced = scores["balanced_accuracy"]
    print(f"{name}: CV balanced accuracy {balanced.mean():.4f} "
          f"+/- {balanced.std():.4f} | accuracy {scores['accuracy'].mean():.4f} | "
          f"macro-F1 {scores['macro_f1'].mean():.4f} | "
          f"train balanced accuracy {scores['train_balanced_accuracy'].mean():.4f}",
          flush=True)
    return {metric: values.mean() for metric, values in scores.items()}


def save_predictions(model, test_df, feature_columns, output_path):
    """Preserve original test columns and row order, adding the required Target."""
    if set(test_df.columns) != set(feature_columns):
        raise ValueError("Test columns must match the training feature columns")
    result = test_df.copy()
    result["Target"] = model.predict(test_df.loc[:, feature_columns].to_numpy())
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", nargs="?", type=Path, default=TRAIN_CSV)
    parser.add_argument("test_csv", nargs="?", type=Path, default=TEST_CSV)
    parser.add_argument("--output", type=Path, default=PRED_CSV)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    feature_columns = train_df.drop(columns=[LABEL_COL]).columns.tolist()
    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[LABEL_COL].to_numpy()
    test_df = pd.read_csv(args.test_csv)
    if set(test_df.columns) != set(feature_columns):
        raise ValueError("Test columns must match the training feature columns")
    print(f"Train {X_train.shape}")
    print("Class counts (train):", pd.Series(y_train).value_counts().to_dict(), "\n")

    # Compare shallow and unrestricted trees to inspect the train/CV gap.
    print("-- Single-tree depth comparison --")
    for depth in [2, 4, 6, 8, 10, 12, 15, 20, None]:
        report(f"Tree max_depth={depth}", {"max_depth": depth},
               X_train, y_train, DecisionTreeClassifier)

    # Identical folds make comparisons fair. Balanced accuracy gives each bean
    # class equal importance, regardless of how many samples it has.
    print("\n-- Forest grid search --")
    grid = [
        {"n_estimators": 100, "max_features": features,
         "max_depth": depth, "min_samples_leaf": leaf}
        for features in ["sqrt", 0.5]
        for depth in [12, None]
        for leaf in [1, 3]
    ]
    scores = [(report(str(p), p, X_train, y_train), p) for p in grid]
    _, chosen_tree_params = max(scores, key=lambda item: item[0][SELECT_BY])

    # Also check how the number of voters affects the best configuration.
    print("\n-- Forest size comparison --")
    for size in [25, 50, 200]:
        params = {**chosen_tree_params, "n_estimators": size}
        scores.append((report(str(params), params, X_train, y_train), params))
    best_scores, best_params = max(scores, key=lambda item: item[0][SELECT_BY])
    print(f"\nBest forest by {SELECT_BY}: {best_params}")
    print(f"# Cross validation Balance Accuracy = "
          f"{100 * best_scores[SELECT_BY]:.4f} %")
    print("This CV score was used for parameter selection; it is not an "
          "independent test score.")

    # Refit on all training rows only after model selection is complete.
    final = ForestClassifier(random_state=SEED, **best_params).fit(X_train, y_train)
    result = save_predictions(final, test_df, feature_columns, args.output)
    print(f"\nWrote {len(result)} predictions to {args.output}")
    print("Predicted class counts:", result["Target"].value_counts().to_dict())


if __name__ == "__main__":
    main()
