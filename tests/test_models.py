"""Behavioural checks for the companion pipelines."""

import joblib
import numpy as np
import pandas as pd
import pytest

from mlguide.data import load_dataset, make_customer_frame, split
from mlguide.evaluate import (
    baseline_classifier_accuracy,
    baseline_regressor_rmse,
    classification_report_dict,
    clustering_report_dict,
    cv_scores,
    regression_report_dict,
)
from mlguide.pipelines import (
    boosting_pipeline,
    churn_pipeline,
    clustering_pipeline,
    forest_pipeline,
    logistic_pipeline,
    ridge_pipeline,
)
from mlguide.train import run
from mlguide.tuning import tune_logistic


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("mnist")


def test_stratified_split_preserves_class_ratio():
    X, y = load_dataset("breast_cancer")
    _, _, y_train, y_test = split(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_logistic_beats_majority_baseline():
    X, y = load_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = split(X, y)
    model = logistic_pipeline().fit(X_train, y_train)
    metrics = classification_report_dict(model, X_test, y_test)
    baseline = baseline_classifier_accuracy(X_train, y_train, X_test, y_test)
    assert metrics["accuracy"] > baseline + 0.2
    assert metrics["roc_auc"] > 0.97


def test_forest_multiclass_wine():
    X, y = load_dataset("wine")
    X_train, X_test, y_train, y_test = split(X, y)
    model = forest_pipeline(100).fit(X_train, y_train)
    assert classification_report_dict(model, X_test, y_test)["accuracy"] > 0.9


def test_cross_validation_returns_every_metric():
    X, y = load_dataset("iris")
    scores = cv_scores(logistic_pipeline(), X, y, scoring=["accuracy", "f1_macro"])
    assert set(scores) == {"accuracy", "f1_macro"}
    assert scores["accuracy"] > 0.9


def test_ridge_beats_mean_baseline():
    X, y = load_dataset("diabetes")
    X_train, X_test, y_train, y_test = split(X, y, stratify=False)
    metrics = regression_report_dict(ridge_pipeline().fit(X_train, y_train), X_test, y_test)
    assert metrics["rmse"] < baseline_regressor_rmse(X_train, y_train, X_test, y_test)
    assert metrics["r2"] > 0.3


def test_customer_frame_has_missing_values_and_categories():
    X, y = make_customer_frame(300)
    assert X.isna().any().any()
    assert not pd.api.types.is_numeric_dtype(X["plan"])
    assert set(np.unique(y)) == {0, 1}


@pytest.mark.parametrize("factory", [churn_pipeline, boosting_pipeline])
def test_mixed_type_pipelines_handle_nan_and_unseen_categories(factory):
    X, y = make_customer_frame()
    X_train, X_test, y_train, y_test = split(X, y)
    model = factory().fit(X_train, y_train)
    assert classification_report_dict(model, X_test, y_test)["roc_auc"] > 0.7

    unseen = X_test.head(3).copy()
    unseen["region"] = "antarctica"
    assert model.predict(unseen).shape == (3,)


def test_preprocessor_fits_on_training_data_only():
    X, y = make_customer_frame()
    X_train, _, y_train, _ = split(X, y)
    model = churn_pipeline().fit(X_train, y_train)
    numeric_branch = model.named_steps["prep"].named_transformers_["num"]
    numeric = X_train.select_dtypes("number")
    expected = numeric.fillna(numeric.median()).mean().to_numpy()
    np.testing.assert_allclose(numeric_branch.named_steps["scale"].mean_, expected, rtol=1e-6)


def test_grid_search_selects_a_value_from_the_grid():
    X, y = load_dataset("breast_cancer")
    X_train, _, y_train, _ = split(X, y)
    search = tune_logistic(X_train, y_train, folds=3)
    assert search.best_params_["model__C"] in {0.01, 0.1, 1.0, 10.0}
    assert search.best_score_ > 0.95


def test_kmeans_recovers_iris_structure():
    X, y = load_dataset("iris")
    report = clustering_report_dict(clustering_pipeline(3), X, y)
    assert report["silhouette"] > 0.4
    # Scaling stretches petal and sepal features equally, which blurs the
    # versicolor/virginica boundary; ARI lands near 0.43 rather than ~0.73.
    assert report["adjusted_rand"] > 0.4


def test_saved_model_round_trips(tmp_path):
    X, y = load_dataset("breast_cancer")
    model = logistic_pipeline().fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_end_to_end_run_writes_report(tmp_path):
    report = run(tmp_path)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "churn_logistic.joblib").exists()
    cancer = report["breast_cancer_logistic"]
    assert cancer["accuracy"] > cancer["baseline_accuracy"]
    assert report["diabetes_ridge"]["rmse"] < report["diabetes_ridge"]["baseline_rmse"]
    assert report["wine_forest"]["accuracy"] > 0.9
