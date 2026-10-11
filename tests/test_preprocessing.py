"""
Pruebas unitarias para ingeniería de características y filtrado (src/preprocessing.py).
"""

import pandas as pd
import pytest

from src.preprocessing import (
    build_ml_preprocessor,
    compute_energy_kpis,
    extract_temporal_features,
    filter_dataset,
)


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00:00", periods=48, freq="h"),
        "building_id": ["BLD-1"] * 24 + ["BLD-2"] * 24,
        "building_type": ["Oficina"] * 24 + ["Hospital"] * 24,
        "surface_area_sqm": [10000.0] * 24 + [20000.0] * 24,
        "temperature_celsius": [15.0 + (i % 15) for i in range(48)],
        "humidity_percent": [50.0] * 48,
        "occupancy_rate": [0.5] * 48,
        "hvac_mode": ["Normal"] * 48,
        "energy_consumption_kwh": [100.0 + i * 2 for i in range(48)],
    })


def test_extract_temporal_features(base_df):
    """Verifica la correcta extracción de horas, días de la semana y flags punta."""
    df_temp = extract_temporal_features(base_df)

    assert "hour" in df_temp.columns
    assert "day_of_week" in df_temp.columns
    assert "is_weekend" in df_temp.columns
    assert "is_peak_hour" in df_temp.columns
    assert df_temp["hour"].min() >= 0
    assert df_temp["hour"].max() <= 23


def test_compute_energy_kpis(base_df):
    """Verifica el cálculo de intensidad de energía, emisiones y costes."""
    df_kpi = compute_energy_kpis(base_df, cost_per_kwh=0.20, co2_per_kwh=0.40)

    assert "energy_intensity_kwh_sqm" in df_kpi.columns
    assert "estimated_cost_usd" in df_kpi.columns
    assert "carbon_emissions_kg" in df_kpi.columns
    assert "cooling_degree_delta" in df_kpi.columns

    # Verificar proporcionalidad
    first_row = df_kpi.iloc[0]
    expected_cost = round(first_row["energy_consumption_kwh"] * 0.20, 2)
    assert first_row["estimated_cost_usd"] == expected_cost


def test_filter_dataset(base_df):
    """Verifica el filtrado multidimensional por tipo de edificio y rango de temperatura."""
    filtered_type = filter_dataset(base_df, building_types=["Hospital"])
    assert len(filtered_type) == 24
    assert (filtered_type["building_type"] == "Hospital").all()

    filtered_temp = filter_dataset(base_df, temp_range=(20.0, 25.0))
    assert (filtered_temp["temperature_celsius"] >= 20.0).all()
    assert (filtered_temp["temperature_celsius"] <= 25.0).all()


def test_build_ml_preprocessor(base_df):
    """Verifica que el ColumnTransformer escale numéricas y codifique categóricas."""
    num_cols = ["surface_area_sqm", "temperature_celsius"]
    cat_cols = ["building_type", "hvac_mode"]

    preprocessor = build_ml_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    transformed = preprocessor.fit_transform(base_df[num_cols + cat_cols])

    assert transformed.shape[0] == len(base_df)
    assert transformed.shape[1] >= len(num_cols) + len(cat_cols)
