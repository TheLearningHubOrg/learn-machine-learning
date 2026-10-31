"""Pipeline factories.

Every model is a single Pipeline object, so preprocessing is fitted on the
training folds only and ships together with the model.
"""

from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import RANDOM_STATE


def logistic_pipeline(C: float = 1.0) -> Pipeline:
    """Scaled logistic regression: the baseline every classifier must beat."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2_000)),
        ]
    )


def forest_pipeline(n_estimators: int = 300) -> Pipeline:
    """Random forest. Trees do not need scaling, so the pipeline is one step."""
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """L2-regularised linear regression for continuous targets."""
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def _categorical_branch() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def tabular_preprocessor() -> ColumnTransformer:
    """Route numeric and categorical columns through different transforms."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", _categorical_branch(), make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    )


def churn_pipeline() -> Pipeline:
    """Mixed-type pipeline: impute, encode, scale, then classify."""
    return Pipeline(
        [
            ("prep", tabular_preprocessor()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced")),
        ]
    )


def boosting_pipeline() -> Pipeline:
    """Gradient boosting handles NaN natively, so numeric imputation is skipped."""
    prep = ColumnTransformer(
        [("cat", _categorical_branch(), make_column_selector(dtype_exclude="number"))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("prep", prep),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


def clustering_pipeline(n_clusters: int = 3, n_components: int = 2) -> Pipeline:
    """Scale, compress with PCA, then group with k-means."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init="auto", random_state=RANDOM_STATE)),
        ]
    )
