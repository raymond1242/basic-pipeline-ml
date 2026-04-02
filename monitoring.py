"""Data monitoring module: raw and processed data quality checks."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 60 + "\n"


def monitor_raw(df: pd.DataFrame) -> None:
    """Analyze raw data quality before preprocessing.

    Logs shape, missing values per column, duplicate rows,
    and basic descriptive statistics.

    Args:
        df: Raw dataframe to inspect.
    """
    logger.info(_SEPARATOR)
    logger.info("MONITORING — RAW DATA")
    logger.info("Shape: %d rows x %d columns", df.shape[0], df.shape[1])

    total_missing = df.isnull().sum()
    if total_missing.sum() > 0:
        logger.warning("Missing values detected:")
        for col, count in total_missing[total_missing > 0].items():
            logger.warning("  %s: %d (%.1f%%)", col, count, 100 * count / len(df))
    else:
        logger.info("Missing values: 0")

    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning("Duplicate rows: %d (%.1f%%)", n_duplicates, 100 * n_duplicates / len(df))
    else:
        logger.info("Duplicate rows: 0")

    logger.info("Descriptive statistics:\n%s", df.describe())
    logger.info(_SEPARATOR)


def monitor_processed(df: pd.DataFrame, target_col: str | None = None) -> None:
    """Verify data quality after preprocessing.

    Logs shape, remaining missing values, data types,
    and optional target distribution.

    Args:
        df: Processed dataframe to inspect.
        target_col: Name of the target column. If provided,
            its class distribution is logged.
    """
    logger.info(_SEPARATOR)
    logger.info("MONITORING — PROCESSED DATA")
    logger.info("Shape: %d rows x %d columns", df.shape[0], df.shape[1])

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning("Remaining missing values: %d", remaining_nulls)
    else:
        logger.info("Remaining missing values: 0")

    logger.info("Data types:\n%s", df.dtypes.to_string())

    if target_col and target_col in df.columns:
        dist = df[target_col].value_counts(normalize=True)
        logger.info("Target distribution (%s):\n%s", target_col, dist.to_string())

    logger.info(_SEPARATOR)
