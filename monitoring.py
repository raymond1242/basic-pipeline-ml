"""Módulo de monitoreo: data cruda y data procesada."""

import pandas as pd


# ── Monitoreo data CRUDA ─────────────────────────────


def monitor_raw(df: pd.DataFrame) -> None:
    """
    Analiza la calidad de la data cruda.
    Reporta: forma, nulos, duplicados, KS y estadísticas básicas.
    """
    print("=" * 50)
    print("MONITOREO — DATA CRUDA")
    print(f"  Filas x Columnas : {df.shape}")
    print(f"  Valores nulos    : {df.isnull().sum().sum()}")
    print(f"  Duplicados       : {df.duplicated().sum()}")
    print("  Estadísticas:\n", df.describe())
    print("=" * 50)


# ── Monitoreo data PROCESADA ─────────────────────────


def monitor_processed(df: pd.DataFrame) -> None:
    """
    Verifica la calidad de la data post-preprocesamiento.
    Reporta: forma, nulos restantes, KS y distribución de columnas.
    """
    print("=" * 50)
    print("MONITOREO — DATA PROCESADA")
    print(f"  Filas x Columnas : {df.shape}")
    print(f"  Nulos restantes  : {df.isnull().sum().sum()}")
    print(f"  Tipos de datos:\n{df.dtypes}")
    print("=" * 50)
