"""
2.1 Build a Tree -- Dry Bean dataset
DecisionTreeClassifier + hand-written (stratified) k-fold cross validation.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

TRAIN_CSV = "dry_bean_train.csv"
TEST_CSV = "dry_bean_test.csv"
LABEL_COL = "Class"
PRED_CSV = "dry_bean_test_predictions.csv"
K = 5
SEED = 0
SELECT_BY = "accuracy"


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
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].extend(part)
    return [np.array(f) for f in folds]


def cross_validate(params, X, y, k=K):
    """Returns per-fold macro-F1, per-fold accuracy, and per-fold train macro-F1."""
    folds = make_folds(y, k)
    val_f1, val_acc, train_f1 = [], [], []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        clf = DecisionTreeClassifier(random_state=SEED, **params)  # fresh model per fold
        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[val_idx])
        val_f1.append(f1_score(y[val_idx], pred, average="macro"))
        val_acc.append(accuracy_score(y[val_idx], pred))
        train_f1.append(f1_score(y[train_idx], clf.predict(X[train_idx]), average="macro"))
    return np.array(val_f1), np.array(val_acc), np.array(train_f1)


def report(name, params, X, y):
    f1, acc, tr = cross_validate(params, X, y)
    print(f"{name:<38} CV macro-F1 {f1.mean():.4f} ± {f1.std():.4f} | "
          f"CV acc {acc.mean():.4f} | train F1 {tr.mean():.4f}")
    return {"macro_f1": f1.mean(), "accuracy": acc.mean()}


if __name__ == "__main__":
    import os, sys
    train_path = sys.argv[1] if len(sys.argv) > 1 else TRAIN_CSV
    test_path = sys.argv[2] if len(sys.argv) > 2 else TEST_CSV
    X_train, y_train = load(train_path)
    has_test = os.path.exists(test_path)
    print(f"Train {X_train.shape}")
    print("Class counts (train):", pd.Series(y_train).value_counts().to_dict(), "\n")

    # 1) Baseline: fully grown tree -> watch the train/CV gap
    report("default (unrestricted)", {}, X_train, y_train)

    # 2) Effect of depth (bias-variance curve)
    print("\n-- max_depth --")
    depth_results = {d: report(f"max_depth={d}", {"max_depth": d}, X_train, y_train)
                     for d in [2, 4, 6, 8, 10, 12, 15, 20, None]}

    # 3) Small grid over the most important options
    print("\n-- grid search --")
    grid = [
        {"criterion": c, "max_depth": d, "min_samples_leaf": leaf, "class_weight": cw}
        for c in ["gini", "entropy"]
        for d in [6, 8, 10, 12]
        for leaf in [1, 5, 10, 20]
        for cw in [None, "balanced"]
    ]
    scores = [(report(str(p), p, X_train, y_train), p) for p in grid]
    best_scores, best_params = max(scores, key=lambda t: t[0][SELECT_BY])
    print(f"\nBest params by {SELECT_BY}: {best_params}\n"
          f"  CV accuracy {best_scores['accuracy']:.4f} | CV macro-F1 {best_scores['macro_f1']:.4f}")

    # 4) Retrain on ALL training data with the chosen params
    final = DecisionTreeClassifier(random_state=SEED, **best_params).fit(X_train, y_train)
    print(f"Final tree: depth {final.get_depth()}, leaves {final.get_n_leaves()}")

    # 5) Predict the unlabeled test set and save the answers
    if not os.path.exists(test_path):
        print(f"(No test file at {test_path} - nothing to predict yet.)")
        sys.exit(0)
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[LABEL_COL], errors="ignore").to_numpy()
    pred = final.predict(X_test)
    pd.DataFrame({LABEL_COL: pred}).to_csv(PRED_CSV, index=False)
    print(f"\nWrote {len(pred)} predictions to {PRED_CSV}")
    print("Predicted class counts:", pd.Series(pred).value_counts().to_dict())