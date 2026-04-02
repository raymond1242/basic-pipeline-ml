"""Model training and evaluation module."""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET_COL = "target"  # replace with actual target column name


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into training and testing sets.

    Args:
        df: Input dataframe containing features and target.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_model(n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """Create and return a RandomForestClassifier.

    Args:
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.

    Returns:
        Configured RandomForestClassifier instance.
    """
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def train(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    """Train the model on the provided dataframe.

    Args:
        df: Preprocessed dataframe with features and target.

    Returns:
        Tuple of (trained model, metrics dict with X_test and y_test).
    """
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    model.fit(X_train, y_train)
    logger.info("Model trained on %d samples", len(X_train))
    return model, {"X_test": X_test, "y_test": y_test}


def evaluate(model: RandomForestClassifier, metrics: dict) -> None:
    """Log the classification report for the trained model.

    Args:
        model: Trained classifier.
        metrics: Dict containing X_test and y_test from training.
    """
    y_pred = model.predict(metrics["X_test"])
    report = classification_report(metrics["y_test"], y_pred)
    logger.info("EVALUATION REPORT\n%s", report)
