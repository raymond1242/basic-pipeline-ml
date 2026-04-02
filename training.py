"""Módulo de entrenamiento y evaluación del modelo."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


TARGET_COL = "target"  # <-- reemplazar con columna objetivo real


def split_data(df: pd.DataFrame):
    """Divide en features (X) y etiqueta (y), luego train/test."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model():
    """Instancia y retorna el modelo."""
    return RandomForestClassifier(n_estimators=100, random_state=42)


def train(df: pd.DataFrame):
    """Entrena el modelo y retorna (modelo, métricas)."""
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    model.fit(X_train, y_train)
    metrics = {"X_test": X_test, "y_test": y_test}
    return model, metrics


def evaluate(model, metrics: dict) -> None:
    """Imprime el reporte de clasificación."""
    y_pred = model.predict(metrics["X_test"])
    print("REPORTE DE EVALUACIÓN")
    print(classification_report(metrics["y_test"], y_pred))
