"""End-to-end trainer: fit every model, evaluate it and save the artifacts.

Run with:  python -m mlguide.train --out artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from . import __version__
from .data import load_dataset, make_customer_frame, split
from .evaluate import (
    baseline_classifier_accuracy,
    baseline_regressor_rmse,
    classification_report_dict,
    clustering_report_dict,
    regression_report_dict,
)
from .pipelines import churn_pipeline, clustering_pipeline, forest_pipeline, ridge_pipeline
from .tuning import tune_logistic


def run(out_dir: Path | None = None) -> dict:
    """Train all five models and return a JSON-serialisable report."""
    report: dict = {"version": __version__}

    # 1. Binary classification: breast cancer, tuned logistic regression.
    X, y = load_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = split(X, y)
    search = tune_logistic(X_train, y_train)
    report["breast_cancer_logistic"] = {
        "best_params": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
        "baseline_accuracy": baseline_classifier_accuracy(X_train, y_train, X_test, y_test),
        **classification_report_dict(search.best_estimator_, X_test, y_test),
    }

    # 2. Multiclass classification: wine, random forest.
    X, y = load_dataset("wine")
    X_train, X_test, y_train, y_test = split(X, y)
    forest = forest_pipeline().fit(X_train, y_train)
    report["wine_forest"] = classification_report_dict(forest, X_test, y_test)

    # 3. Regression: diabetes progression, ridge.
    X, y = load_dataset("diabetes")
    X_train, X_test, y_train, y_test = split(X, y, stratify=False)
    ridge = ridge_pipeline(alpha=1.0).fit(X_train, y_train)
    report["diabetes_ridge"] = {
        "baseline_rmse": baseline_regressor_rmse(X_train, y_train, X_test, y_test),
        **regression_report_dict(ridge, X_test, y_test),
    }

    # 4. Mixed-type tabular data: synthetic churn with missing values.
    X, y = make_customer_frame()
    X_train, X_test, y_train, y_test = split(X, y)
    churn = churn_pipeline().fit(X_train, y_train)
    report["churn_logistic"] = classification_report_dict(churn, X_test, y_test)

    # 5. Unsupervised: iris clusters compared against the true species.
    X, y = load_dataset("iris")
    report["iris_kmeans"] = clustering_report_dict(clustering_pipeline(n_clusters=3), X, y)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, out_dir / "breast_cancer_logistic.joblib")
        joblib.dump(churn, out_dir / "churn_logistic.joblib")
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the guide's models.")
    parser.add_argument("--out", type=Path, default=None, help="directory for model artifacts")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.out), indent=2))


if __name__ == "__main__":
    main()
