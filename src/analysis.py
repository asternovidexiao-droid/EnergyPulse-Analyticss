"""
Módulo de análisis estadístico y analítica descriptiva para EnergyPulse Analytics.
Implementa cálculo de KPIs globales, perfiles horarios, matrices de correlación,
resúmenes estadísticos detallados y detección matemática de anomalías (IQR / Z-score).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN


def compute_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula los indicadores clave de rendimiento (KPIs) ejecutivos del conjunto de datos.

    Args:
        df: DataFrame enriquecido con columnas energéticas.

    Returns:
        Diccionario con métricas agregadas clave.
    """
    if df.empty or TARGET_COLUMN not in df.columns:
        return {
            "total_consumption_kwh": 0.0,
            "peak_demand_kwh": 0.0,
            "mean_consumption_kwh": 0.0,
            "median_consumption_kwh": 0.0,
            "total_cost_usd": 0.0,
            "total_co2_kg": 0.0,
            "total_solar_kwh": 0.0,
            "solar_coverage_pct": 0.0,
            "active_buildings_count": 0,
        }

    total_kwh = float(df[TARGET_COLUMN].sum())
    peak_kwh = float(df[TARGET_COLUMN].max())
    mean_kwh = float(df[TARGET_COLUMN].mean())
    median_kwh = float(df[TARGET_COLUMN].median())

    cost_val = float(df["estimated_cost_usd"].sum()) if "estimated_cost_usd" in df.columns else total_kwh * 0.165
    co2_val = float(df["carbon_emissions_kg"].sum()) if "carbon_emissions_kg" in df.columns else total_kwh * 0.385
    solar_kwh = float(df["solar_pv_kwh"].sum()) if "solar_pv_kwh" in df.columns else 0.0

    solar_pct = (solar_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0
    active_bld = int(df["building_id"].nunique()) if "building_id" in df.columns else 1

    return {
        "total_consumption_kwh": round(total_kwh, 2),
        "peak_demand_kwh": round(peak_kwh, 2),
        "mean_consumption_kwh": round(mean_kwh, 2),
        "median_consumption_kwh": round(median_kwh, 2),
        "total_cost_usd": round(cost_val, 2),
        "total_co2_kg": round(co2_val, 2),
        "total_solar_kwh": round(solar_kwh, 2),
        "solar_coverage_pct": round(solar_pct, 2),
        "active_buildings_count": active_bld,
    }


def compute_comprehensive_statistics(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calcula una tabla estadística exhaustiva con medidas de tendencia central,
    dispersión, forma de distribución (asimetría, curtosis) y rango intercuartílico (IQR).

    Args:
        df: DataFrame de entrada.
        columns: Lista opcional de columnas numéricas a analizar.

    Returns:
        pd.DataFrame formateado con estadísticas completas por columna.
    """
    if df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    if columns:
        cols_to_use = [c for c in columns if c in num_df.columns]
        num_df = num_df[cols_to_use]

    if num_df.empty:
        return pd.DataFrame()

    stats_list = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue

        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25

        stats_list.append({
            "Variable": col,
            "Registros Válidos": int(len(s)),
            "Media": round(float(s.mean()), 2),
            "Desv. Estándar": round(float(s.std()), 2),
            "Mínimo": round(float(s.min()), 2),
            "Q25 (P25)": round(q25, 2),
            "Mediana (Q50)": round(float(s.median()), 2),
            "Q75 (P75)": round(q75, 2),
            "Máximo": round(float(s.max()), 2),
            "Rango IQR": round(iqr, 2),
            "Asimetría (Skew)": round(float(s.skew()), 2) if len(s) > 2 else 0.0,
            "Curtosis (Kurt)": round(float(s.kurt()), 2) if len(s) > 3 else 0.0,
        })

    return pd.DataFrame(stats_list)


def compute_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de Pearson entre las variables numéricas seleccionadas.

    Args:
        df: DataFrame analizado.
        columns: Columnas a incluir. Si es None, incluye todas las numéricas relevantes.

    Returns:
        DataFrame con la matriz de correlación simétrica.
    """
    if df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    # Filtrar columnas auxiliares de poco valor interpretativo directo
    ignore_cols = ["hour", "day_of_week", "is_weekend", "is_peak_hour"]
    candidate_cols = [c for c in num_df.columns if c not in ignore_cols]

    if columns:
        candidate_cols = [c for c in columns if c in candidate_cols]

    if len(candidate_cols) < 2:
        return pd.DataFrame()

    return num_df[candidate_cols].corr(method="pearson").round(3)


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str = TARGET_COLUMN,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Identifica registros atípicos (outliers) utilizando el método no paramétrico
    del Rango Intercuartílico (IQR de Tukey).

    Args:
        df: DataFrame analizado.
        column: Nombre de la columna numérica a inspeccionar.
        factor: Multiplicador del IQR (típicamente 1.5 para outliers suaves, 3.0 para extremos).

    Returns:
        DataFrame conteniendo únicamente las filas identificadas como anómalas,
        con columnas descriptivas del umbral.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < 4:
        return pd.DataFrame()

    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25

    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)

    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_df = df[outliers_mask].copy()

    outliers_df["outlier_lower_bound"] = round(lower_bound, 2)
    outliers_df["outlier_upper_bound"] = round(upper_bound, 2)
    outliers_df["deviation_kwh"] = (
        np.maximum(0, outliers_df[column] - upper_bound) +
        np.maximum(0, lower_bound - outliers_df[column])
    ).round(2)

    return outliers_df.reset_index(drop=True)


def compute_hourly_load_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el perfil de curva de carga horario promedio y desviación estándar
    agrupado por tipo de edificio y hora del día (0-23).

    Args:
        df: DataFrame con 'hour', 'building_type' y target.

    Returns:
        DataFrame con la curva de carga horaria promedio.
    """
    if df.empty or "hour" not in df.columns or TARGET_COLUMN not in df.columns:
        return pd.DataFrame()

    group_cols = ["hour"]
    if "building_type" in df.columns:
        group_cols.append("building_type")

    agg_df = df.groupby(group_cols)[TARGET_COLUMN].agg(
        mean_consumption="mean",
        std_consumption="std",
        median_consumption="median",
        p90_consumption=lambda x: x.quantile(0.90),
    ).reset_index()

    agg_df["mean_consumption"] = agg_df["mean_consumption"].round(2)
    agg_df["std_consumption"] = agg_df["std_consumption"].round(2)
    agg_df["median_consumption"] = agg_df["median_consumption"].round(2)
    agg_df["p90_consumption"] = agg_df["p90_consumption"].round(2)

    return agg_df
