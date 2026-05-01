"""sklearn preprocessing pipeline.

XGBoost natively handles missing values, so the numeric branch only enforces
dtype. The categorical branch imputes a sentinel and ordinal-encodes (XGBoost
splits on integer codes fine and we keep the column count linear in cardinality
rather than blowing up via one-hot)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from .feature_engineering import FEATURE_GROUPS

NUMERIC_FEATURES = FEATURE_GROUPS["numeric"]
CATEGORICAL_FEATURES = FEATURE_GROUPS["categorical"]


def _to_float32(X):
    return X.astype("float32")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("cast", FunctionTransformer(_to_float32, validate=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
