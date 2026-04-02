"""Model training and evaluation module."""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into training and testing sets with stratification.

    Args:
        df: Input dataframe containing features and target.
        target_col: Name of the target column.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler (fit on train only).

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def build_model(
    n_estimators: int = 200,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Create and return a RandomForestClassifier.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        min_samples_leaf: Minimum samples required at a leaf node.
        random_state: Seed for reproducibility.

    Returns:
        Configured RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def train(df: pd.DataFrame, target_col: str) -> tuple[RandomForestClassifier, dict]:
    """Train the model: split, scale, fit.

    Args:
        df: Preprocessed dataframe with features and target.
        target_col: Name of the target column.

    Returns:
        Tuple of (trained model, metrics dict with test data and scaler).
    """
    X_train, X_test, y_train, y_test = split_data(df, target_col)
    X_train_sc, X_test_sc, scaler = scale_data(X_train, X_test)

    model = build_model()
    model.fit(X_train_sc, y_train)
    logger.info("Model trained on %d samples", len(X_train))

    return model, {
        "X_test_sc": X_test_sc,
        "y_test": y_test,
        "scaler": scaler,
    }


def evaluate(model: RandomForestClassifier, metrics: dict) -> None:
    """Log the classification report and prediction probabilities.

    Args:
        model: Trained classifier.
        metrics: Dict containing X_test_sc and y_test from training.
    """
    y_pred = model.predict(metrics["X_test_sc"])
    y_prob = model.predict_proba(metrics["X_test_sc"])[:, 1]

    report = classification_report(metrics["y_test"], y_pred)
    logger.info("EVALUATION REPORT\n%s", report)
    logger.info("Prediction probabilities — min: %.3f, max: %.3f, mean: %.3f",
                y_prob.min(), y_prob.max(), y_prob.mean())
