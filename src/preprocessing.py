"""
Módulo de preprocesamiento, ingeniería de características y filtrado para EnergyPulse Analytics.
Implementa transformaciones reproducibles para análisis exploratorio y pipelines de Machine Learning.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DEFAULT_CO2_KG_PER_KWH,
    DEFAULT_COST_PER_KWH_USD,
    PEAK_HOURS,
    TARGET_COLUMN,
)


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae componentes temporales útiles a partir de la columna 'timestamp'.

    Args:
        df: DataFrame que contiene la columna 'timestamp' (datetime o string).

    Returns:
        DataFrame con nuevas columnas temporales enriquecidas.
    """
    df_out = df.copy()
    if "timestamp" not in df_out.columns:
        return df_out

    # Asegurar formato datetime
    timestamps = pd.to_datetime(df_out["timestamp"], errors="coerce")
    df_out["timestamp"] = timestamps

    # Componentes temporales
    df_out["hour"] = timestamps.dt.hour
    df_out["day_of_week"] = timestamps.dt.dayofweek
    df_out["day_name"] = timestamps.dt.day_name(locale="es_ES") if hasattr(timestamps.dt, "day_name") else timestamps.dt.day_name()
    # Fallback si locale no está instalado
    day_map = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    df_out["day_name_es"] = timestamps.dt.dayofweek.map(day_map)
    df_out["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    df_out["is_peak_hour"] = timestamps.dt.hour.isin(PEAK_HOURS).astype(int)

    return df_out


def compute_energy_kpis(
    df: pd.DataFrame,
    cost_per_kwh: float = DEFAULT_COST_PER_KWH_USD,
    co2_per_kwh: float = DEFAULT_CO2_KG_PER_KWH,
) -> pd.DataFrame:
    """
    Calcula indicadores de negocio (intensidad energética EUI, costes y huella de carbono).

    Args:
        df: DataFrame con las columnas energéticas.
        cost_per_kwh: Tarifa unitaria en USD por kWh.
        co2_per_kwh: Factor de emisión en kg CO2 por kWh.

    Returns:
        DataFrame enriquecido con métricas de sostenibilidad y costes.
    """
    df_out = df.copy()

    # Intensidad de Consumo Energético (kWh/m²)
    if TARGET_COLUMN in df_out.columns and "surface_area_sqm" in df_out.columns:
        # Evitar división por cero
        areas = df_out["surface_area_sqm"].replace(0, np.nan)
        df_out["energy_intensity_kwh_sqm"] = (df_out[TARGET_COLUMN] / areas).round(4)

    # Coste estimado en USD
    if TARGET_COLUMN in df_out.columns:
        df_out["estimated_cost_usd"] = (df_out[TARGET_COLUMN] * cost_per_kwh).round(2)
        df_out["carbon_emissions_kg"] = (df_out[TARGET_COLUMN] * co2_per_kwh).round(2)

    # Grados-hora de refrigeración y calefacción (CDH / HDH)
    if "temperature_celsius" in df_out.columns:
        df_out["cooling_degree_delta"] = np.maximum(0.0, df_out["temperature_celsius"] - 22.0).round(2)
        df_out["heating_degree_delta"] = np.maximum(0.0, 16.0 - df_out["temperature_celsius"]).round(2)

    return df_out


def filter_dataset(
    df: pd.DataFrame,
    building_types: Optional[List[str]] = None,
    building_ids: Optional[List[str]] = None,
    hvac_modes: Optional[List[str]] = None,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    temp_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Aplica filtros combinados al conjunto de datos de manera vectorizada y eficiente.

    Args:
        df: DataFrame a filtrar.
        building_types: Lista de tipos de edificio permitidos.
        building_ids: Lista de IDs de edificio permitidos.
        hvac_modes: Modos de climatización permitidos.
        date_range: Tupla con (fecha_inicio, fecha_fin).
        temp_range: Tupla con (temperatura_min, temperatura_max).

    Returns:
        DataFrame filtrado.
    """
    filtered = df.copy()

    if building_types and "building_type" in filtered.columns:
        filtered = filtered[filtered["building_type"].isin(building_types)]

    if building_ids and "building_id" in filtered.columns:
        filtered = filtered[filtered["building_id"].isin(building_ids)]

    if hvac_modes and "hvac_mode" in filtered.columns:
        filtered = filtered[filtered["hvac_mode"].isin(hvac_modes)]

    if date_range and "timestamp" in filtered.columns and len(date_range) == 2:
        start_date, end_date = date_range
        # Asegurar comparación justa de fechas
        filtered = filtered[
            (filtered["timestamp"] >= pd.to_datetime(start_date)) &
            (filtered["timestamp"] <= pd.to_datetime(end_date))
        ]

    if temp_range and "temperature_celsius" in filtered.columns and len(temp_range) == 2:
        min_temp, max_temp = temp_range
        filtered = filtered[
            (filtered["temperature_celsius"] >= min_temp) &
            (filtered["temperature_celsius"] <= max_temp)
        ]

    return filtered.reset_index(drop=True)


def build_ml_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer de Scikit-Learn para preprocesar de forma
    reproducible variables numéricas y categóricas sin fuga de datos (data leakage).

    Args:
        numeric_features: Lista de nombres de columnas cuantitativas.
        categorical_features: Lista de nombres de columnas cualitativas.

    Returns:
        ColumnTransformer configurado.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
