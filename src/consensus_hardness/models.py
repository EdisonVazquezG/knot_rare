# src/consensus_hardness/models.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic_model(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """
    Fit class-balanced logistic regression with standardized predictors.
    """

    data = df[predictors + [target]].dropna().copy()

    X = data[predictors].astype(float).values
    y = data[target].astype(int).values

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]

    clf = pipe.named_steps["clf"]

    coef_table = pd.DataFrame(
        {
            "predictor": predictors,
            "coef_standardized": clf.coef_[0],
            "odds_ratio_standardized": np.exp(clf.coef_[0]),
        }
    )

    metrics = {
        "n": len(y),
        "positives": int(y.sum()),
        "roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
    }

    return coef_table, metrics, pipe


def cv_logistic_metrics(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Cross-validated ROC-AUC and PR-AUC for class-balanced logistic regression.
    """

    data = df[predictors + [target]].dropna().copy()

    X = data[predictors].astype(float).values
    y = data[target].astype(int).values

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    probs = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        pipe.fit(X[train_idx], y[train_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    return {
        "n": len(y),
        "positives": int(y.sum()),
        "cv_roc_auc": roc_auc_score(y, probs),
        "cv_pr_auc": average_precision_score(y, probs),
    }


def logistic_model_comparison(
    df: pd.DataFrame,
    target: str,
    model_specs: dict[str, list[str]],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare several logistic models using cross-validated metrics.
    """

    rows = []

    for label, predictors in model_specs.items():
        out = cv_logistic_metrics(
            df=df,
            target=target,
            predictors=predictors,
            n_splits=n_splits,
            seed=seed,
        )
        out["model"] = label
        rows.append(out)

    return pd.DataFrame(rows)