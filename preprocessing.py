"""Data preprocessing module."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    """Encode categorical variables using one-hot encoding.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with categorical columns encoded.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features using StandardScaler.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with numeric columns standardized (mean=0, std=1).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Args:
        df: Raw input dataframe.

    Returns:
        Cleaned, encoded, and scaled dataframe.
    """
    df = handle_missing(df)
    df = encode_features(df)
    df = scale_features(df)
    return df
