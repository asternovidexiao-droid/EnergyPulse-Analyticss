"""
Pruebas unitarias para el módulo de validación y calidad de datos (src/validation.py).
"""

import pandas as pd
import pytest

from src.config import REQUIRED_COLUMNS, TARGET_COLUMN
from src.validation import clean_dataframe, validate_dataframe


@pytest.fixture
def valid_sample_df():
    """Genera un DataFrame con 15 registros únicos completamente válido para pruebas."""
    return pd.DataFrame({
        "timestamp": [f"2026-01-01 {i:02d}:00:00" for i in range(15)],
        "building_id": ["BLD-101"] * 15,
        "building_type": ["Oficina"] * 15,
        "surface_area_sqm": [12000.0] * 15,
        "temperature_celsius": [18.0 + (i * 0.5) for i in range(15)],
        "humidity_percent": [50.0 + (i % 5) for i in range(15)],
        "occupancy_rate": [0.1 + (i * 0.05) for i in range(15)],
        "hvac_mode": ["Normal"] * 15,
        TARGET_COLUMN: [200.0 + (i * 10.0) for i in range(15)],
    })


def test_validate_none_and_empty_dataframe():
    """Verifica que objetos None o DataFrames vacíos sean rechazados con error explícito."""
    report_none = validate_dataframe(None)
    assert not report_none.is_valid
    assert len(report_none.errors) > 0

    report_empty = validate_dataframe(pd.DataFrame())
    assert not report_empty.is_valid
    assert any("vacío" in err.lower() for err in report_empty.errors)


def test_validate_missing_required_columns(valid_sample_df):
    """Verifica que se detecten y reporten las columnas obligatorias faltantes."""
    df_missing = valid_sample_df.drop(columns=["building_type", "temperature_celsius"])
    report = validate_dataframe(df_missing)

    assert not report.is_valid
    assert any("faltan columnas obligatorias" in err.lower() for err in report.errors)
    assert any("building_type" in err for err in report.errors)


def test_validate_valid_dataframe(valid_sample_df):
    """Verifica que un DataFrame conforme pase la validación exitosamente sin errores."""
    report = validate_dataframe(valid_sample_df)
    assert report.is_valid
    assert len(report.errors) == 0
    assert report.total_rows == len(valid_sample_df)
    assert report.total_columns == len(valid_sample_df.columns)


def test_validate_nulls_and_duplicates_warning(valid_sample_df):
    """Verifica la detección adecuada de nulos y duplicados."""
    df_with_issues = valid_sample_df.copy()
    # Insertar duplicado
    df_with_issues = pd.concat([df_with_issues, df_with_issues.iloc[0:1]], ignore_index=True)
    # Insertar nulo en temperatura
    df_with_issues.loc[2, "temperature_celsius"] = None

    report = validate_dataframe(df_with_issues)
    assert report.duplicate_rows == 1
    assert "temperature_celsius" in report.null_counts
    assert any("duplicadas" in w.lower() for w in report.warnings)


def test_clean_dataframe_drops_duplicates_and_normalizes(valid_sample_df):
    """Verifica que clean_dataframe elimine duplicados y normalice tipos numéricos."""
    duplicated_df = pd.concat([valid_sample_df, valid_sample_df.iloc[0:2]], ignore_index=True)
    assert len(duplicated_df) == len(valid_sample_df) + 2

    cleaned = clean_dataframe(duplicated_df, drop_duplicates=True)
    assert len(cleaned) == len(valid_sample_df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])
