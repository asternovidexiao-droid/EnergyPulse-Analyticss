"""
Pruebas unitarias para el módulo de carga y exportación de datos (src/data_loader.py).
"""

import io
import pandas as pd
import pytest

from src.data_loader import export_dataframe_to_csv, get_or_create_sample_data, load_csv_data


def test_load_csv_from_valid_buffer():
    """Verifica que un CSV válido en un buffer StringIO se lea y valide correctamente."""
    csv_text = (
        "timestamp,building_id,building_type,surface_area_sqm,temperature_celsius,"
        "humidity_percent,occupancy_rate,hvac_mode,energy_consumption_kwh\n"
        + "\n".join([
            f"2026-01-01 {i:02d}:00:00,BLD-1,Oficina,5000,21.5,50.0,0.8,Normal,150.0"
            for i in range(15)
        ])
    )
    buffer = io.StringIO(csv_text)
    df, report = load_csv_data(buffer, clean=True)

    assert report.is_valid
    assert len(df) == 15
    assert "energy_consumption_kwh" in df.columns


def test_load_csv_empty_file():
    """Verifica que un archivo vacío devuelva un DataFrame vacío y reporte de error controlado."""
    empty_buffer = io.StringIO("")
    df, report = load_csv_data(empty_buffer)

    assert not report.is_valid
    assert df.empty
    assert any("vacío" in err.lower() for err in report.errors)


def test_load_csv_corrupt_delimiters():
    """Verifica que errores severos de análisis no generen caídas no controladas."""
    corrupt_buffer = io.StringIO("col1,col2\n1,2,3,4,5,6\n\"unclosed quote,")
    df, report = load_csv_data(corrupt_buffer)

    assert not report.is_valid or df.empty or len(report.errors) > 0


def test_get_or_create_sample_data():
    """Verifica la carga del dataset de muestra integrado."""
    df, report = get_or_create_sample_data()

    assert report.is_valid
    assert len(df) >= 1000
    assert set(["building_id", "energy_consumption_kwh", "temperature_celsius"]).issubset(df.columns)


def test_export_dataframe_to_csv():
    """Verifica la serialización a CSV en formato UTF-8 con BOM reproducible."""
    sample_df = pd.DataFrame({
        "building_id": ["BLD-1", "BLD-2"],
        "energy_consumption_kwh": [100.5, 200.7],
    })

    csv_bytes = export_dataframe_to_csv(sample_df)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

    # Re-leer desde los bytes exportados
    reloaded_df = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(reloaded_df) == 2
    assert list(reloaded_df["building_id"]) == ["BLD-1", "BLD-2"]
