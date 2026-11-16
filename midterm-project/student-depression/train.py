#!/usr/bin/env python3
"""Train and compare models, then save the selected pipeline.

Usage:
    python train.py                  # compares 4 models on both feature sets,
                                     # saves the best model of the default set
    python train.py --feature-set no_suicidal

Outputs:
    artifacts/model.joblib   - sklearn Pipeline (preprocessing + model)
    artifacts/metrics.json   - cross-validation + hold-out results
"""
import argparse
import json
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import features as F

SEED = 42
FEATURE_SETS = {"full": F.FEATURES_FULL, "no_suicidal": F.FEATURES_NO_SUICIDAL}


def make_preprocessor(columns):
    num = [c for c in F.NUMERIC if c in columns]
    cat = [c for c in columns if c not in num]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            (
                "cat",
                # rare / unseen categories (e.g. the 12-18 "Others" rows in
                # sleep_duration / dietary_habits) are folded into one
                # "infrequent" bucket instead of failing
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=100,
                    sparse_output=False,
                ),
                cat,
            ),
        ]
    )


def candidates():
    """name -> (estimator, parameter grid). The first one is the baseline."""
    return {
        "logreg (baseline)": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [20, 100]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
            {"clf__max_depth": [8, 14, 20], "clf__min_samples_leaf": [5, 20]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [8, 16], "clf__l2_regularization": [0.0, 1.0]},
        ),
    }


def holdout_metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def run_feature_set(name, df, quiet=False):
    cols = FEATURE_SETS[name]
    X, y = df[cols], df[F.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, best = {}, (None, -1.0, None)
    for model_name, (est, grid) in candidates().items():
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor(cols)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        m = holdout_metrics(gs.best_estimator_, X_test, y_test)
        results[model_name] = {
            "cv_roc_auc": float(gs.best_score_),
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            **{f"test_{k}": float(v) for k, v in m.items()},
            "fit_seconds": round(time.time() - t0, 1),
        }
        if not quiet:
            r = results[model_name]
            print(
                f"[{name:11s}] {model_name:24s} cv_auc={r['cv_roc_auc']:.4f} "
                f"test_auc={r['test_roc_auc']:.4f} acc={r['test_accuracy']:.4f} "
                f"rec={r['test_recall']:.4f} prec={r['test_precision']:.4f} ({r['fit_seconds']}s)"
            )
        if gs.best_score_ > best[1]:
            best = (model_name, gs.best_score_, gs.best_estimator_)
    return results, best, (X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="full",
                    help="feature set of the model that gets saved (default: full)")
    args = ap.parse_args()

    df = F.load_data()
    print(f"Loaded {len(df)} rows, positive rate {df[F.TARGET].mean():.3f}")

    all_results, chosen = {}, None
    for name in FEATURE_SETS:
        res, best, data = run_feature_set(name, df)
        all_results[name] = {"models": res, "selected": best[0]}
        if name == args.feature_set:
            chosen = (name, best, data)

    name, (model_name, _, pipe), (X, y) = chosen
    pipe.fit(X, y)  # refit the selected configuration on all data for deployment
    F.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": pipe, "features": list(X.columns), "feature_set": name,
                 "model_name": model_name}, F.MODEL_PATH)
    all_results["saved_model"] = {"feature_set": name, "model": model_name}
    F.METRICS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {model_name} ({name}) -> {F.MODEL_PATH}")


if __name__ == "__main__":
    main()
