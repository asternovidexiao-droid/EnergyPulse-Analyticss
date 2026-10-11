"""
Pruebas unitarias para análisis estadístico y detección de anomalías (src/analysis.py).
"""

import pandas as pd
import pytest

from src.analysis import (
    compute_comprehensive_statistics,
    compute_correlation_matrix,
    compute_hourly_load_profiles,
    compute_summary_kpis,
    detect_outliers_iqr,
)


@pytest.fixture
def analytics_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00:00", periods=72, freq="h"),
        "hour": [i % 24 for i in range(72)],
        "building_id": ["BLD-1"] * 72,
        "building_type": ["Oficina"] * 72,
        "surface_area_sqm": [10000.0] * 72,
        "temperature_celsius": [15.0 + (i % 20) for i in range(72)],
        "occupancy_rate": [0.5] * 72,
        "energy_consumption_kwh": [100.0 + (i % 20) * 10 for i in range(72)],
        "estimated_cost_usd": [15.0] * 72,
        "carbon_emissions_kg": [35.0] * 72,
        "solar_pv_kwh": [10.0] * 72,
    })


def test_compute_summary_kpis(analytics_df):
    """Verifica el cálculo de KPIs ejecutivos globales."""
    kpis = compute_summary_kpis(analytics_df)

    assert kpis["total_consumption_kwh"] > 0
    assert kpis["peak_demand_kwh"] > 0
    assert kpis["mean_consumption_kwh"] > 0
    assert kpis["active_buildings_count"] == 1


def test_compute_comprehensive_statistics(analytics_df):
    """Verifica que la tabla estadística devuelva todas las métricas requeridas."""
    stats = compute_comprehensive_statistics(analytics_df, columns=["energy_consumption_kwh", "temperature_celsius"])

    assert len(stats) == 2
    cols_expected = ["Variable", "Media", "Desv. Estándar", "Mínimo", "Q25 (P25)", "Mediana (Q50)", "Q75 (P75)", "Máximo", "Rango IQR"]
    for col in cols_expected:
        assert col in stats.columns


def test_compute_correlation_matrix(analytics_df):
    """Verifica que la matriz de correlación devuelva valores entre -1 y 1."""
    corr = compute_correlation_matrix(analytics_df, columns=["temperature_celsius", "energy_consumption_kwh"])

    assert not corr.empty
    assert corr.loc["temperature_celsius", "temperature_celsius"] == 1.0
    assert -1.0 <= corr.loc["temperature_celsius", "energy_consumption_kwh"] <= 1.0


def test_detect_outliers_iqr(analytics_df):
    """Verifica que el método IQR detecte anomalías artificialmente introducidas."""
    df_with_outlier = analytics_df.copy()
    # Inyectar outlier extremo
    df_with_outlier.loc[10, "energy_consumption_kwh"] = 9999.0

    outliers = detect_outliers_iqr(df_with_outlier, column="energy_consumption_kwh", factor=1.5)

    assert len(outliers) >= 1
    assert 9999.0 in outliers["energy_consumption_kwh"].values
    assert "deviation_kwh" in outliers.columns


def test_compute_hourly_load_profiles(analytics_df):
    """Verifica la agregación de la curva de carga horaria."""
    profiles = compute_hourly_load_profiles(analytics_df)

    assert not profiles.empty
    assert "mean_consumption" in profiles.columns
    assert "hour" in profiles.columns
    assert set(profiles["hour"].unique()) == set(range(24))
