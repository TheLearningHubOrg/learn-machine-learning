"""Dataset loaders and splitting helpers.

Every loader returns pandas objects so column names survive through the
pipelines, which makes feature importances and debugging far easier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "diabetes": load_diabetes,
    "iris": load_iris,
    "wine": load_wine,
}


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one of the bundled scikit-learn datasets as (X, y) pandas objects."""
    try:
        loader = _LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_LOADERS)}") from exc
    bunch = loader(as_frame=True)
    return bunch.data, bunch.target


def split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
):
    """Hold out a test set. Stratify for classification so class ratios match."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state,
    )


def make_customer_frame(
    n_rows: int = 600, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic churn table with mixed types and missing values.

    The bundled datasets are all numeric and clean. Real tables are not, so this
    frame exists to exercise imputation, one-hot encoding and ColumnTransformer.
    """
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 72, size=n_rows).astype(float)
    monthly = rng.normal(70, 25, size=n_rows).clip(15, 150)
    tickets = rng.poisson(1.5, size=n_rows).astype(float)
    plan = rng.choice(["basic", "standard", "premium"], size=n_rows, p=[0.5, 0.3, 0.2])
    region = rng.choice(["north", "south", "east", "west"], size=n_rows)

    # Churn is more likely with short tenure, many tickets and the basic plan.
    logit = (
        -1.0
        - 0.05 * tenure
        + 0.6 * tickets
        + 0.01 * monthly
        + np.where(plan == "basic", 0.8, 0.0)
    )
    churned = rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))

    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly.round(2),
            "support_tickets": tickets,
            "plan": plan,
            "region": region,
        }
    )
    # Knock out about 5% of values in two columns to simulate messy source data.
    for column in ("monthly_charge", "region"):
        mask = rng.random(n_rows) < 0.05
        frame.loc[mask, column] = np.nan
    return frame, pd.Series(churned.astype(int), name="churned")
