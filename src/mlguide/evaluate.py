"""Evaluation helpers: cross-validation, held-out metrics and baselines."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from .data import RANDOM_STATE


def cv_scores(
    estimator, X, y, *, scoring, task: str = "classification", folds: int = 5
) -> dict[str, float]:
    """Return mean cross-validated test scores keyed by metric name."""
    if task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(clone(estimator), X, y, cv=cv, scoring=scoring)
    return {
        key.removeprefix("test_"): float(np.mean(values))
        for key, values in result.items()
        if key.startswith("test_")
    }


def classification_report_dict(model, X_test, y_test) -> dict[str, float]:
    """Accuracy, macro F1 and ROC AUC (one-vs-rest when there are >2 classes)."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    if proba.shape[1] == 2:
        auc = roc_auc_score(y_test, proba[:, 1])
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr")
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "roc_auc": float(auc),
    }


def regression_report_dict(model, X_test, y_test) -> dict[str, float]:
    """RMSE and MAE in target units, plus R^2 as a unitless summary."""
    pred = model.predict(X_test)
    return {
        "rmse": float(root_mean_squared_error(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def clustering_report_dict(pipeline, X, y_true=None) -> dict[str, float]:
    """Silhouette uses only the data; ARI compares with known labels if given."""
    labels = pipeline.fit_predict(X)
    embedded = pipeline[:-1].transform(X)
    report = {"silhouette": float(silhouette_score(embedded, labels))}
    if y_true is not None:
        report["adjusted_rand"] = float(adjusted_rand_score(y_true, labels))
    return report


def baseline_classifier_accuracy(X_train, y_train, X_test, y_test) -> float:
    """Accuracy of always predicting the majority class."""
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return float(accuracy_score(y_test, dummy.predict(X_test)))


def baseline_regressor_rmse(X_train, y_train, X_test, y_test) -> float:
    """RMSE of always predicting the training mean."""
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    return float(root_mean_squared_error(y_test, dummy.predict(X_test)))
