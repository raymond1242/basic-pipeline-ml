"""Data preprocessing module."""

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    "gender",
    "occupation",
    "chronotype",
    "mental_health_condition",
    "season",
    "day_type",
]


def load_data(path: str) -> pd.DataFrame:
    """Load a dataset from the given file path.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded dataframe.
    """
    return pd.read_csv(path)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with missing rows dropped.
    """
    return df.dropna()


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns, creating *_enc columns.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with new label-encoded columns appended.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            encoder = LabelEncoder()
            df[f"{col}_enc"] = encoder.fit_transform(df[col])
            logger.info("Encoded '%s' -> '%s_enc' (%d classes)", col, col, df[f"{col}_enc"].nunique())
    return df


def select_features(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Select only the required feature and target columns.

    Args:
        df: Input dataframe.
        feature_cols: List of feature column names to keep.
        target_col: Name of the target column.

    Returns:
        Dataframe with only the selected columns.
    """
    return df[feature_cols + [target_col]]


def preprocess(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Args:
        df: Raw input dataframe.
        feature_cols: List of feature column names to keep.
        target_col: Name of the target column.

    Returns:
        Cleaned, encoded, and filtered dataframe.
    """
    df = handle_missing(df)
    df = encode_features(df)
    df = select_features(df, feature_cols, target_col)
    return df
