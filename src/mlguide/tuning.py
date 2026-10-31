"""Hyperparameter search over a whole pipeline."""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import RANDOM_STATE
from .pipelines import logistic_pipeline


def tune_logistic(X_train, y_train, *, folds: int = 5) -> GridSearchCV:
    """Grid-search the inverse regularisation strength of the logistic model.

    Parameter names use the ``<step>__<param>`` convention so the search reaches
    inside the pipeline, and the scaler is refitted on every training fold.
    """
    search = GridSearchCV(
        logistic_pipeline(),
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X_train, y_train)
