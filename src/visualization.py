"""
Módulo de visualización interactiva para EnergyPulse Analytics.
Construye gráficos interactivos con Plotly: series temporales, matrices de correlación,
curvas de carga horaria, diagramas de caja, análisis de anomalías y evaluación de ML.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import TARGET_COLUMN

# Paleta de colores profesional
THEME_TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#0284c7"    # Sky 600
COLOR_SECONDARY = "#0f766e"  # Teal 700
COLOR_ACCENT = "#f59e0b"     # Amber 500
COLOR_DANGER = "#ef4444"     # Red 500
COLOR_SUCCESS = "#10b981"    # Emerald 500


def plot_time_series_consumption(
    df: pd.DataFrame,
    building_id: Optional[str] = None,
    color_by: str = "building_type",
) -> go.Figure:
    """
    Genera un gráfico interactivo de serie temporal del consumo eléctrico (kWh).

    Args:
        df: DataFrame con 'timestamp' y target.
        building_id: ID específico para filtrar (opcional).
        color_by: Columna para segmentar por colores.

    Returns:
        Figura de Plotly.
    """
    plot_df = df.copy()
    if building_id and "building_id" in plot_df.columns:
        plot_df = plot_df[plot_df["building_id"] == building_id]

    if "timestamp" not in plot_df.columns or TARGET_COLUMN not in plot_df.columns:
        return go.Figure()

    # Si hay demasiados puntos, muestrear o ordenar
    plot_df = plot_df.sort_values(by="timestamp")

    fig = px.line(
        plot_df,
        x="timestamp",
        y=TARGET_COLUMN,
        color=color_by if color_by in plot_df.columns else None,
        title="Evolución Temporal de la Demanda Eléctrica (kWh)",
        labels={"timestamp": "Fecha y Hora", TARGET_COLUMN: "Consumo Energético (kWh)"},
        template=THEME_TEMPLATE,
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_hourly_load_curve(df_hourly: pd.DataFrame) -> go.Figure:
    """
    Grafica la curva de carga horaria promedio por tipología de edificio con banda de confianza.

    Args:
        df_hourly: DataFrame resultante de `compute_hourly_load_profiles`.

    Returns:
        Figura de Plotly.
    """
    if df_hourly.empty:
        return go.Figure()

    fig = go.Figure()
    has_bld = "building_type" in df_hourly.columns

    if has_bld:
        for b_type in df_hourly["building_type"].unique():
            sub = df_hourly[df_hourly["building_type"] == b_type].sort_values(by="hour")
            fig.add_trace(go.Scatter(
                x=sub["hour"],
                y=sub["mean_consumption"],
                mode="lines+markers",
                name=f"{b_type} (Media)",
                hovertemplate="Hora: %{x}h<br>Media: %{y:.1f} kWh<extra></extra>",
            ))
    else:
        sub = df_hourly.sort_values(by="hour")
        fig.add_trace(go.Scatter(
            x=sub["hour"],
            y=sub["mean_consumption"],
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=3),
            name="Consumo Promedio Global",
        ))

    fig.update_layout(
        title="Curva de Carga Horaria Promedio (00:00 - 23:00 h)",
        xaxis=dict(title="Hora del Día", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Consumo Promedio (kWh)"),
        template=THEME_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_temperature_vs_consumption(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de dispersión entre temperatura ambiente y demanda eléctrica con curva de tendencia.

    Args:
        df: DataFrame con temperatura y target.

    Returns:
        Figura de Plotly.
    """
    if "temperature_celsius" not in df.columns or TARGET_COLUMN not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df,
        x="temperature_celsius",
        y=TARGET_COLUMN,
        color="hvac_mode" if "hvac_mode" in df.columns else None,
        hover_data=["building_type", "occupancy_rate"] if "building_type" in df.columns else None,
        trendline="ols",
        title="Respuesta Térmica: Temperatura Exterior vs Consumo Eléctrico",
        labels={
            "temperature_celsius": "Temperatura Exterior (°C)",
            TARGET_COLUMN: "Consumo Energético (kWh)",
            "hvac_mode": "Modo Climatización",
        },
        template=THEME_TEMPLATE,
        opacity=0.65,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_distribution_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Genera un histograma con estimación de densidad y caja marginal.

    Args:
        df: DataFrame con los datos.
        column: Columna numérica a graficar.

    Returns:
        Figura de Plotly.
    """
    if column not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=40,
        title=f"Distribución Frecuencial y Cuartiles: {column}",
        labels={column: column, "count": "Frecuencia (Horas)"},
        color_discrete_sequence=[COLOR_PRIMARY],
        template=THEME_TEMPLATE,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_categorical_comparison(df: pd.DataFrame, cat_col: str, num_col: str = TARGET_COLUMN) -> go.Figure:
    """
    Diagrama de cajas (Box Plot) interactivo para comparar una métrica numérica entre categorías.

    Args:
        df: DataFrame de entrada.
        cat_col: Columna categórica (e.g., 'building_type' o 'hvac_mode').
        num_col: Columna numérica continua.

    Returns:
        Figura de Plotly.
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return go.Figure()

    fig = px.box(
        df,
        x=cat_col,
        y=num_col,
        color=cat_col,
        points="outliers",
        title=f"Comparativa de {num_col} según {cat_col}",
        labels={cat_col: cat_col, num_col: f"{num_col} (kWh)"},
        template=THEME_TEMPLATE,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Genera un mapa de calor (Heatmap) interactivo con valores numéricos visibles.

    Args:
        corr_matrix: DataFrame con coeficientes de correlación de Pearson.

    Returns:
        Figura de Plotly.
    """
    if corr_matrix.empty:
        return go.Figure()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Matriz de Correlación Lineal (Pearson)",
        template=THEME_TEMPLATE,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_anomaly_detection(df: pd.DataFrame, outliers_df: pd.DataFrame) -> go.Figure:
    """
    Grafica la serie temporal destacando visualmente los puntos anómalos detectados por IQR.

    Args:
        df: DataFrame completo con 'timestamp' y target.
        outliers_df: Subconjunto con registros anómalos.

    Returns:
        Figura de Plotly.
    """
    if "timestamp" not in df.columns or TARGET_COLUMN not in df.columns:
        return go.Figure()

    fig = go.Figure()

    # Traza normal de consumo
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[TARGET_COLUMN],
        mode="lines",
        line=dict(color="#94a3b8", width=1),
        name="Consumo Nominal",
        opacity=0.7,
    ))

    # Traza de anomalías detectadas
    if not outliers_df.empty:
        fig.add_trace(go.Scatter(
            x=outliers_df["timestamp"],
            y=outliers_df[TARGET_COLUMN],
            mode="markers",
            marker=dict(color=COLOR_DANGER, size=8, symbol="circle-open-dot", line=dict(width=2)),
            name=f"Anomalías Detectadas ({len(outliers_df)})",
            hovertemplate="Fecha: %{x}<br>Consumo Anómalo: %{y:.1f} kWh<extra></extra>",
        ))

    fig.update_layout(
        title="Detección de Anomalías de Carga Fuera de Rango (Método IQR)",
        xaxis=dict(title="Fecha y Hora"),
        yaxis=dict(title="Demanda (kWh)"),
        template=THEME_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_real_vs_predicted(y_real: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Gráfico de dispersión entre valores reales y predichos con línea de referencia diagonal ideal (y = x).

    Args:
        y_real: Array con etiquetas reales.
        y_pred: Array con predicciones del modelo.

    Returns:
        Figura de Plotly.
    """
    min_val = min(float(np.min(y_real)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_real)), float(np.max(y_pred)))

    fig = go.Figure()

    # Puntos predichos vs reales
    fig.add_trace(go.Scatter(
        x=y_real,
        y=y_pred,
        mode="markers",
        marker=dict(color=COLOR_PRIMARY, size=5, opacity=0.5),
        name="Predicciones",
        hovertemplate="Real: %{x:.1f} kWh<br>Predicho: %{y:.1f} kWh<extra></extra>",
    ))

    # Línea ideal y = x
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLOR_DANGER, dash="dash", width=2),
        name="Referencia Perfecta (y = x)",
    ))

    fig.update_layout(
        title="Evaluación de Precisión: Valores Reales vs Predichos (Test Set)",
        xaxis=dict(title="Consumo Real (kWh)"),
        yaxis=dict(title="Consumo Predicho por el Modelo (kWh)"),
        template=THEME_TEMPLATE,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_residuals_distribution(residuals: np.ndarray) -> go.Figure:
    """
    Grafica la distribución de los errores residuales (y_real - y_pred).

    Args:
        residuals: Array de errores residuales.

    Returns:
        Figura de Plotly.
    """
    fig = px.histogram(
        x=residuals,
        nbins=40,
        marginal="rug",
        title="Distribución de Errores Residuales (y_real - y_pred)",
        labels={"x": "Error Residual (kWh)", "count": "Frecuencia"},
        color_discrete_sequence=[COLOR_SECONDARY],
        template=THEME_TEMPLATE,
    )
    # Línea vertical en 0 (error nulo)
    fig.add_vline(x=0.0, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Error Cero")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_feature_importances(df_importances: pd.DataFrame) -> go.Figure:
    """
    Grafica la importancia de variables del modelo en orden descendente.

    Args:
        df_importances: DataFrame con columnas 'Característica' y 'Importancia'.

    Returns:
        Figura de Plotly.
    """
    if df_importances.empty:
        return go.Figure()

    top_df = df_importances.head(12).sort_values(by="Importancia", ascending=True)

    fig = px.bar(
        top_df,
        x="Importancia",
        y="Característica",
        orientation="h",
        text_auto=".3f",
        title="Top Variables Predictoras más Influyentes",
        labels={"Importancia": "Importancia Relativa", "Característica": "Variable"},
        color="Importancia",
        color_continuous_scale="Blues",
        template=THEME_TEMPLATE,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
