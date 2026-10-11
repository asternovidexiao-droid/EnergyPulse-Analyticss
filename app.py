"""
EnergyPulse Analytics - Plataforma Profesional de Inteligencia Energética y Predicción de Consumo
Aplicación interactiva Streamlit para análisis exploratorio, control de calidad,
detección de anomalías y modelado predictivo con Machine Learning.
"""

import sys
from pathlib import Path

# Garantizar resolución correcta de rutas
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import (
    compute_comprehensive_statistics,
    compute_correlation_matrix,
    compute_hourly_load_profiles,
    compute_summary_kpis,
    detect_outliers_iqr,
)
from src.config import (
    DEFAULT_CO2_KG_PER_KWH,
    DEFAULT_COST_PER_KWH_USD,
    TARGET_COLUMN,
    VALID_CATEGORIES,
)
from src.data_loader import (
    export_dataframe_to_csv,
    get_or_create_sample_data,
    load_csv_data,
)
from src.modeling import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    predict_scenario,
    train_energy_model,
)
from src.preprocessing import (
    compute_energy_kpis,
    extract_temporal_features,
    filter_dataset,
)
from src.visualization import (
    plot_anomaly_detection,
    plot_categorical_comparison,
    plot_correlation_heatmap,
    plot_distribution_histogram,
    plot_feature_importances,
    plot_hourly_load_curve,
    plot_real_vs_predicted,
    plot_residuals_distribution,
    plot_temperature_vs_consumption,
    plot_time_series_consumption,
)

# -----------------------------------------------------------------------------
# Configuración general de la página Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyPulse Analytics | Smart Building AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Estilos CSS personalizados para UI/UX profesional
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Estilos generales y tipografía */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 1.0rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0284c7;
        margin: 0.2rem 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    .status-badge-ok {
        background-color: #ecfdf5;
        color: #059669;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: #fffbeb;
        color: #d97706;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Inicialización de Estado de Sesión (Session State)
# -----------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    sample_df, sample_rep = get_or_create_sample_data()
    st.session_state.raw_df = sample_df
    st.session_state.validation_report = sample_rep
    st.session_state.data_source_name = "Conjunto de datos de muestra (60 días, 5 edificios)"

if "model_result" not in st.session_state:
    st.session_state.model_result = None

# -----------------------------------------------------------------------------
# Barra Lateral (Sidebar): Gestión de Archivos y Filtros Globales
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?auto=format&fit=crop&w=600&q=80",
        caption="EnergyPulse AI System",
        use_container_width=True,
    )
    st.subheader("⚡ Gestión de Datos")

    # Botón para restaurar dataset de muestra
    if st.button("🔄 Cargar Dataset de Muestra", use_container_width=True):
        sample_df, sample_rep = get_or_create_sample_data()
        st.session_state.raw_df = sample_df
        st.session_state.validation_report = sample_rep
        st.session_state.data_source_name = "Conjunto de datos de muestra (60 días, 5 edificios)"
        st.session_state.model_result = None
        st.success("¡Datos de ejemplo cargados con éxito!")
        st.rerun()

    # Subida de archivo CSV
    uploaded_file = st.file_uploader(
        "Subir archivo CSV propio",
        type=["csv"],
        help="El archivo debe contener telemetría con columnas: timestamp, building_id, building_type, surface_area_sqm, temperature_celsius, humidity_percent, occupancy_rate, hvac_mode, energy_consumption_kwh.",
    )

    if uploaded_file is not None:
        try:
            up_df, up_rep = load_csv_data(uploaded_file, clean=True)
            st.session_state.raw_df = up_df
            st.session_state.validation_report = up_rep
            st.session_state.data_source_name = f"Archivo cargado: {uploaded_file.name}"
            st.session_state.model_result = None
        except Exception as e:
            st.error(f"Error crítico al leer el archivo: {str(e)}")

    st.markdown("---")
    st.subheader("⚙️ Configuración Económica")
    custom_cost = st.number_input(
        "Tarifa Eléctrica ($ / kWh)",
        min_value=0.01,
        max_value=2.00,
        value=DEFAULT_COST_PER_KWH_USD,
        step=0.01,
        format="%.3f",
    )
    custom_co2 = st.number_input(
        "Factor de Emisión ($kg\\ CO_2$ / kWh)",
        min_value=0.01,
        max_value=2.00,
        value=DEFAULT_CO2_KG_PER_KWH,
        step=0.01,
        format="%.3f",
    )

    st.markdown("---")
    st.caption("EnergyPulse Analytics v1.0 • Motor Analítico")


# -----------------------------------------------------------------------------
# Encabezado Principal
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ EnergyPulse Analytics</h1>
        <p>Plataforma Integral de Inteligencia Operativa, Diagnóstico Térmico y Pronóstico Predictivo de Consumo Energético en Edificaciones</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Validación y Diagnóstico Inicial de Datos
# -----------------------------------------------------------------------------
raw_data = st.session_state.raw_df
validation_report = st.session_state.validation_report

if not validation_report.is_valid:
    st.error("🚨 **El archivo de datos contiene errores críticos que impiden el análisis:**")
    for err in validation_report.errors:
        st.markdown(f"- {err}")
    st.info("💡 Por favor, cargue un archivo con las columnas requeridas o pulse el botón **'Cargar Dataset de Muestra'** en la barra lateral.")
    st.stop()

# Procesamiento de características de tiempo y KPIs
enriched_df = extract_temporal_features(raw_data)
enriched_df = compute_energy_kpis(enriched_df, cost_per_kwh=custom_cost, co2_per_kwh=custom_co2)

# Advertencias no bloqueantes
if validation_report.warnings:
    with st.expander(f"⚠️ Diagnóstico de Calidad: {len(validation_report.warnings)} Advertencias Detectadas", expanded=False):
        for w in validation_report.warnings:
            st.markdown(f"- {w}")

# -----------------------------------------------------------------------------
# Filtros Interactivos Superiores
# -----------------------------------------------------------------------------
with st.expander("🔎 Filtros Interactivos de Análisis", expanded=True):
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    available_types = sorted(list(enriched_df["building_type"].dropna().unique())) if "building_type" in enriched_df.columns else []
    selected_types = col_f1.multiselect(
        "Tipología de Edificio",
        options=available_types,
        default=available_types,
    )

    available_blds = sorted(list(enriched_df["building_id"].dropna().unique())) if "building_id" in enriched_df.columns else []
    selected_blds = col_f2.multiselect(
        "Instalación (ID)",
        options=available_blds,
        default=available_blds,
    )

    available_hvac = sorted(list(enriched_df["hvac_mode"].dropna().unique())) if "hvac_mode" in enriched_df.columns else []
    selected_hvac = col_f3.multiselect(
        "Modo de Climatización",
        options=available_hvac,
        default=available_hvac,
    )

    # Rango de fechas
    if "timestamp" in enriched_df.columns and pd.api.types.is_datetime64_any_dtype(enriched_df["timestamp"]):
        min_date = enriched_df["timestamp"].min().date()
        max_date = enriched_df["timestamp"].max().date()
        date_sel = col_f4.date_input(
            "Ventana Temporal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_sel, tuple) and len(date_sel) == 2:
            date_range = (pd.Timestamp(date_sel[0]), pd.Timestamp(date_sel[1]) + pd.Timedelta(days=1))
        else:
            date_range = None
    else:
        date_range = None
        col_f4.write("Fechas no disponibles")

# Aplicar filtros
df_filtered = filter_dataset(
    enriched_df,
    building_types=selected_types,
    building_ids=selected_blds,
    hvac_modes=selected_hvac,
    date_range=date_range,
)

if df_filtered.empty:
    st.warning("⚠️ No se encontraron registros que coincidan con la combinación de filtros seleccionada. Ajuste los criterios de búsqueda.")
    st.stop()

# -----------------------------------------------------------------------------
# Pestañas Principales de la Aplicación
# -----------------------------------------------------------------------------
tab_summary, tab_data, tab_eda, tab_ml, tab_sim, tab_docs = st.tabs([
    "📊 Panel Ejecutivo & KPIs",
    "📁 Gestión y Exploración de Datos",
    "🔍 Análisis Exploratorio (EDA)",
    "🤖 Modelado Predictivo (ML)",
    "🔮 Simulador 'What-If'",
    "📖 Metadatos y Documentación",
])

# -----------------------------------------------------------------------------
# TAB 1: Panel Ejecutivo & KPIs
# -----------------------------------------------------------------------------
with tab_summary:
    kpi_dict = compute_summary_kpis(df_filtered)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Demanda Total</div>
                <div class="kpi-value">{kpi_dict['total_consumption_kwh']:,.1f}</div>
                <div class="kpi-subtext">kWh consumidos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Demanda Punta (Peak)</div>
                <div class="kpi-value">{kpi_dict['peak_demand_kwh']:,.1f}</div>
                <div class="kpi-subtext">kW máx. instantáneo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Coste Estimado</div>
                <div class="kpi-value">${kpi_dict['total_cost_usd']:,.2f}</div>
                <div class="kpi-subtext">a ${custom_cost:.3f}/kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Huella de Carbono</div>
                <div class="kpi-value">{kpi_dict['total_co2_kg']:,.1f}</div>
                <div class="kpi-subtext">kg CO₂eq generados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Aporte Solar Fotovoltaico</div>
                <div class="kpi-value">{kpi_dict['solar_coverage_pct']:.1f}%</div>
                <div class="kpi-subtext">{kpi_dict['total_solar_kwh']:,.1f} kWh generados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Vista gráfica rápida del Panel
    c_graph1, c_graph2 = st.columns([3, 2])
    with c_graph1:
        st.plotly_chart(
            plot_time_series_consumption(df_filtered),
            use_container_width=True,
        )
    with c_graph2:
        df_hourly = compute_hourly_load_profiles(df_filtered)
        st.plotly_chart(
            plot_hourly_load_curve(df_hourly),
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: Gestión y Exploración de Datos
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("📁 Exploración Tabular y Control de Calidad")
    st.info(f"**Origen de Datos Actual:** {st.session_state.data_source_name} | **Registros Filtrados:** {len(df_filtered):,} de {len(enriched_df):,}")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Total Columnas", len(df_filtered.columns))
    col_meta2.metric("Filas Duplicadas", validation_report.duplicate_rows)
    col_meta3.metric("Valores Nulos Totales", sum(validation_report.null_counts.values()))
    col_meta4.metric("Instalaciones Únicas", df_filtered["building_id"].nunique() if "building_id" in df_filtered.columns else 1)

    # Selector de columnas para visualizar
    available_cols = list(df_filtered.columns)
    selected_display_cols = st.multiselect(
        "Seleccionar columnas a visualizar:",
        options=available_cols,
        default=[c for c in ["timestamp", "building_id", "building_type", "surface_area_sqm", "temperature_celsius", "occupancy_rate", "hvac_mode", TARGET_COLUMN, "estimated_cost_usd"] if c in available_cols],
    )

    st.dataframe(
        df_filtered[selected_display_cols],
        use_container_width=True,
        height=380,
    )

    # Resumen de tipos de datos y nulos
    with st.expander("🔍 Detalle Técnico de Columnas y Tipos de Datos", expanded=False):
        types_summary = pd.DataFrame({
            "Columna": df_filtered.columns,
            "Tipo Dtype": [str(t) for t in df_filtered.dtypes],
            "Nulos Detectados": [df_filtered[c].isna().sum() for c in df_filtered.columns],
            "Valores Únicos": [df_filtered[c].nunique() for c in df_filtered.columns],
        })
        st.table(types_summary)

    # Descarga de datos
    st.markdown("---")
    csv_bytes = export_dataframe_to_csv(df_filtered)
    st.download_button(
        label="📥 Descargar Datos Filtrados en Formato CSV",
        data=csv_bytes,
        file_name="energypulse_filtered_data.csv",
        mime="text/csv",
        help="Exporta los registros actuales con codificación UTF-8 para análisis externo.",
    )

# -----------------------------------------------------------------------------
# TAB 3: Análisis Exploratorio (EDA)
# -----------------------------------------------------------------------------
with tab_eda:
    st.subheader("🔍 Análisis Estadístico y Exploratorio Interactivo")

    eda_choice = st.radio(
        "Seleccione el enfoque analítico:",
        [
            "📈 Curvas de Carga & Perfiles Horarios",
            "🌡️ Respuesta Térmica & Meteorología",
            "📊 Distribución de Variables & Estadísticas",
            "🏢 Comparativa por Tipología & HVAC",
            "🔥 Matriz de Correlación Multivariable",
            "⚠️ Detección Matemática de Anomalías (IQR)",
        ],
        horizontal=True,
    )

    if eda_choice == "📈 Curvas de Carga & Perfiles Horarios":
        st.markdown("#### Análisis de la Curva de Demanda Horaria")
        st.caption("Permite identificar los patrones de consumo durante la jornada diaria (00:00 a 23:00 h) y los horarios de mayor exigencia de la red eléctrica.")
        df_hourly = compute_hourly_load_profiles(df_filtered)
        st.plotly_chart(plot_hourly_load_curve(df_hourly), use_container_width=True)
        st.dataframe(df_hourly, use_container_width=True, height=250)

    elif eda_choice == "🌡️ Respuesta Térmica & Meteorología":
        st.markdown("#### Sensibilidad Térmica: Temperatura Exterior vs Demanda")
        st.caption("Determina el impacto de la temperatura ambiente en el encendido de sistemas de climatización (HVAC). Los edificios incrementan su consumo tanto por enfriamiento en días cálidos (>22°C) como por calefacción en días fríos (<16°C).")
        st.plotly_chart(plot_temperature_vs_consumption(df_filtered), use_container_width=True)

    elif eda_choice == "📊 Distribución de Variables & Estadísticas":
        st.markdown("#### Estadísticas Descriptivas Exhaustivas")
        num_cols = list(df_filtered.select_dtypes(include=[np.number]).columns)
        stats_table = compute_comprehensive_statistics(df_filtered, columns=num_cols)
        st.dataframe(stats_table, use_container_width=True)

        col_hist_var = st.selectbox("Seleccione variable para histograma y caja:", options=num_cols, index=num_cols.index(TARGET_COLUMN) if TARGET_COLUMN in num_cols else 0)
        st.plotly_chart(plot_distribution_histogram(df_filtered, col_hist_var), use_container_width=True)

    elif eda_choice == "🏢 Comparativa por Tipología & HVAC":
        st.markdown("#### Distribución de Demanda según Factores Operativos")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if "building_type" in df_filtered.columns:
                st.plotly_chart(plot_categorical_comparison(df_filtered, "building_type"), use_container_width=True)
        with col_c2:
            if "hvac_mode" in df_filtered.columns:
                st.plotly_chart(plot_categorical_comparison(df_filtered, "hvac_mode"), use_container_width=True)

    elif eda_choice == "🔥 Matriz de Correlación Multivariable":
        st.markdown("#### Correlaciones Lineales de Pearson")
        st.caption("Identifica qué variables numéricas guardan una relación lineal directa o inversa con la demanda de energía.")
        corr_matrix = compute_correlation_matrix(df_filtered)
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

    elif eda_choice == "⚠️ Detección Matemática de Anomalías (IQR)":
        st.markdown("#### Detección de Consumos Inusuales y Fuera de Rango (Método IQR)")
        st.caption("Aplica la regla estadística de Tukey ($Q_1 - 1.5 \\times IQR$ y $Q_3 + 1.5 \\times IQR$) para aislar horas con consumos anómalos (posibles fallos en sensores, fugas energéticas o eventos imprevistos).")

        iqr_mult = st.slider("Factor Multiplicador IQR:", min_value=1.0, max_value=3.5, value=1.5, step=0.1)
        outliers = detect_outliers_iqr(df_filtered, column=TARGET_COLUMN, factor=iqr_mult)

        st.plotly_chart(plot_anomaly_detection(df_filtered, outliers), use_container_width=True)

        if not outliers.empty:
            st.warning(f"Se detectaron **{len(outliers):,}** registros anómalos ({len(outliers)/len(df_filtered):.2%}).")
            st.dataframe(
                outliers[["timestamp", "building_id", "building_type", TARGET_COLUMN, "outlier_upper_bound", "deviation_kwh"]].head(50),
                use_container_width=True,
            )
        else:
            st.success("No se encontraron anomalías bajo el umbral configurado.")

# -----------------------------------------------------------------------------
# TAB 4: Modelado Predictivo (Machine Learning)
# -----------------------------------------------------------------------------
with tab_ml:
    st.subheader("🤖 Modelado Predictivo de Carga Eléctrica")
    st.caption("Entrena un modelo supervisado de regresión con preprocesamiento formal (escalado y codificación One-Hot en Pipeline) y evaluación frente a un modelo base (Dummy Regressor).")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    model_choice = col_m1.selectbox(
        "Algoritmo de Regresión",
        options=["Random Forest", "Gradient Boosting", "Ridge", "Linear Regression"],
    )
    test_ratio = col_m2.slider(
        "Tamaño Conjunto Test (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    ) / 100.0

    n_trees = col_m3.slider(
        "Nº de Estimadores (Árboles)",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
        disabled=(model_choice in ["Ridge", "Linear Regression"]),
    )
    max_d = col_m4.slider(
        "Profundidad Máxima (Max Depth)",
        min_value=3,
        max_value=25,
        value=12,
        step=1,
        disabled=(model_choice in ["Ridge", "Linear Regression"]),
    )

    if st.button("🚀 Entrenar y Evaluar Modelo", type="primary", use_container_width=True):
        with st.spinner(f"Entrenando {model_choice} sobre los datos filtrados..."):
            try:
                res = train_energy_model(
                    df=df_filtered,
                    model_type=model_choice,
                    test_size=test_ratio,
                    n_estimators=n_trees,
                    max_depth=max_d,
                    random_state=42,
                )
                st.session_state.model_result = res
                st.success(f"¡Modelo {model_choice} entrenado exitosamente con {res.sample_size_train:,} muestras de entrenamiento y {res.sample_size_test:,} de evaluación!")
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {str(e)}")

    res = st.session_state.model_result
    if res is not None:
        st.markdown("---")
        st.markdown("### 🏆 Comparativa de Desempeño: Modelo Propuesto vs Modelo Base")

        c_met1, c_met2, c_met3, c_met4 = st.columns(4)
        c_met1.metric(
            "MAE (Error Absoluto Medio)",
            f"{res.test_metrics.mae:.2f} kWh",
            delta=f"{res.baseline_metrics.mae - res.test_metrics.mae:+.2f} kWh vs Base",
            delta_color="normal",
        )
        c_met2.metric(
            "RMSE (Raíz Error Cuadrático)",
            f"{res.test_metrics.rmse:.2f} kWh",
            delta=f"{res.baseline_metrics.rmse - res.test_metrics.rmse:+.2f} kWh vs Base",
            delta_color="normal",
        )
        c_met3.metric(
            "Coeficiente R²",
            f"{res.test_metrics.r2:.4f}",
            delta=f"{res.test_metrics.r2 - res.baseline_metrics.r2:+.4f} vs Base",
            delta_color="normal",
        )
        c_met4.metric(
            "MAPE (Error % Absoluto)",
            f"{res.test_metrics.mape * 100:.2f}%",
            delta=f"{(res.baseline_metrics.mape - res.test_metrics.mape) * 100:+.2f}% vs Base",
            delta_color="normal",
        )

        # Gráficos diagnósticos del Modelo
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.plotly_chart(plot_real_vs_predicted(res.y_test_real, res.y_test_pred), use_container_width=True)
        with col_plot2:
            st.plotly_chart(plot_residuals_distribution(res.residuals), use_container_width=True)

        if not res.feature_importances.empty:
            st.markdown("### 🔍 Importancia de Características (Feature Importance)")
            st.plotly_chart(plot_feature_importances(res.feature_importances), use_container_width=True)
            with st.expander("Ver tabla completa de importancias"):
                st.dataframe(res.feature_importances, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: Simulador "What-If"
# -----------------------------------------------------------------------------
with tab_sim:
    st.subheader("🔮 Simulador de Escenarios Hipotéticos 'What-If'")
    st.caption("Ajuste los parámetros operativos y meteorológicos para estimar en tiempo real la demanda prevista, el coste asociado y la huella de carbono.")

    # Verificar si hay un modelo entrenado; si no, entrenar uno ligero por defecto
    if st.session_state.model_result is None:
        st.info("💡 Entrenando automáticamente un modelo inicial con los datos actuales para habilitar el simulador...")
        try:
            res_default = train_energy_model(df_filtered, model_type="Random Forest", n_estimators=60, random_state=42)
            st.session_state.model_result = res_default
        except Exception as e:
            st.warning(f"No fue posible preentrenar el modelo: {str(e)}")

    if st.session_state.model_result is not None:
        active_pipeline = st.session_state.model_result.pipeline

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_bld_type = st.selectbox("Tipología del Inmueble:", options=VALID_CATEGORIES["building_type"], index=0)
            sim_area = st.slider("Superficie Construida (m²):", min_value=1000, max_value=50000, value=12500, step=500)
            sim_temp = st.slider("Temperatura Exterior (°C):", min_value=-10.0, max_value=45.0, value=26.0, step=0.5)

        with sim_c2:
            sim_occ = st.slider("Tasa de Ocupación Humana:", min_value=0.0, max_value=1.0, value=0.75, step=0.05, format="%.2f")
            sim_hvac = st.selectbox("Modo del Sistema HVAC:", options=VALID_CATEGORIES["hvac_mode"], index=2)
            sim_humidity = st.slider("Humedad Relativa (%):", min_value=10.0, max_value=100.0, value=50.0, step=5.0)

        with sim_c3:
            sim_hour = st.slider("Hora del Día (0-23 h):", min_value=0, max_value=23, value=14, step=1)
            sim_is_weekend = st.radio("Día de la Semana:", options=["Día Laborable (L-V)", "Fin de Semana (S-D)"], horizontal=True)
            weekend_val = 1 if "Fin de Semana" in sim_is_weekend else 0

        # Preparar diccionario de simulación
        scenario_data = {
            "surface_area_sqm": float(sim_area),
            "temperature_celsius": float(sim_temp),
            "humidity_percent": float(sim_humidity),
            "occupancy_rate": float(sim_occ),
            "hour": int(sim_hour),
            "is_weekend": int(weekend_val),
            "building_type": sim_bld_type,
            "hvac_mode": sim_hvac,
        }

        pred_result = predict_scenario(
            pipeline=active_pipeline,
            scenario_input=scenario_data,
            cost_per_kwh=custom_cost,
            co2_factor=custom_co2,
        )

        st.markdown("---")
        st.markdown("### 📊 Resultado de la Simulación")

        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Demanda Prevista</div>
                    <div class="kpi-value">{pred_result['predicted_consumption_kwh']:,.1f}</div>
                    <div class="kpi-subtext">kWh para esa hora</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sp2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Coste Estimado</div>
                    <div class="kpi-value">${pred_result['estimated_cost_usd']:,.2f}</div>
                    <div class="kpi-subtext">coste horario previsto</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sp3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Emisiones Estimadas</div>
                    <div class="kpi-value">{pred_result['carbon_emissions_kg']:,.1f}</div>
                    <div class="kpi-subtext">kg CO₂eq generados</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recomendaciones de eficiencia energética inteligentes
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("💡 **Recomendaciones de Optimización Operativa:**")
        if sim_hvac == "High" and sim_occ < 0.5:
            st.warning("⚠️ **Potencial de Ahorro:** El sistema HVAC está en modo 'High' con una ocupación inferior al 50%. Ajustar el sistema a modo 'Eco' o 'Normal' reduciría el consumo previsto entre un 20% y un 35%.")
        elif sim_temp > 28.0 and sim_hvac == "Normal":
            st.info("ℹ️ **Alerta de Confort Térmico:** La temperatura exterior es de {:.1f}°C. Se recomienda preclimatizar en horarios valle para evitar picos tarifarios durante las horas punta.".format(sim_temp))
        else:
            st.success("✅ **Operación Eficiente:** La configuración seleccionada mantiene una correlación adecuada entre aforo, modo de climatización y temperatura ambiente.")
    else:
        st.warning("Debe entrenar un modelo en la pestaña 'Modelado Predictivo (ML)' para habilitar el simulador.")

# -----------------------------------------------------------------------------
# TAB 6: Metadatos y Documentación
# -----------------------------------------------------------------------------
with tab_docs:
    st.subheader("📖 Documentación del Dataset, Métodos y Limitaciones")

    st.markdown(
        """
        ### 1. Descripción del Problema
        La optimización del consumo eléctrico en edificios comerciales e industriales representa uno de los mayores desafíos para la descarbonización urbana y la gestión de costes operativos. **EnergyPulse Analytics** aborda este problema integrando:
        - Auditoría automatizada de telemetría de contadores inteligentes (*Smart Meters*).
        - Descomposición de factores térmicos, ocupacionales y estacionales.
        - Modelos de aprendizaje automático para anticipar la demanda punta (*Peak Shaving*).
        - Evaluación del impacto financiero y ambiental en tiempo real.

        ### 2. Diccionario de Variables
        | Variable | Unidad | Tipo | Descripción |
        | :--- | :--- | :--- | :--- |
        | `timestamp` | Datetime | Temporal | Fecha y hora local de la medición horaria. |
        | `building_id` | Texto | Categórico | Identificador del edificio analizado (`BLD-101` a `BLD-105`). |
        | `building_type` | Categoría | Categórico | Tipología de uso (`Oficina`, `Hospital`, `Centro Comercial`, `Escuela`, `Residencial`). |
        | `surface_area_sqm` | $m^2$ | Continuo | Superficie total climatizada construida. |
        | `temperature_celsius` | °C | Continuo | Temperatura meteorológica exterior ambiente. |
        | `humidity_percent` | % | Continuo | Humedad relativa del aire exterior. |
        | `occupancy_rate` | Ratio [0, 1] | Continuo | Fracción del aforo máximo ocupado durante la hora. |
        | `hvac_mode` | Categoría | Categórico | Estado operativo del sistema HVAC (`Off`, `Eco`, `Normal`, `High`). |
        | `solar_pv_kwh` | kWh | Continuo | Autogeneración eléctrica fotovoltaica in-situ. |
        | `energy_consumption_kwh` | kWh | Continuo | **Variable Objetivo (Target)**: Demanda eléctrica neta horaria. |

        ### 3. Rigor Metodológico y Prevención de Data Leakage
        - **Pipeline Cerrado**: El preprocesamiento (escalado estándar de variables cuantitativas y codificación One-Hot de categorías) se ajusta exclusivamente con el conjunto de entrenamiento (`X_train`) y se aplica sin refit a los datos de prueba (`X_test`).
        - **Comparación con Modelo Base**: Todos los modelos se evalúan contra un estimador ingenuo (*DummyRegressor* con estrategia de media), calculando explícitamente la ganancia en MAE, RMSE y $R^2$.
        - **Detección de Anomalías**: Se implementa el método del Rango Intercuartílico (IQR de Tukey), insensible a valores extremos y sin suposición estricta de normalidad gaussiana.

        ### 4. Limitaciones Conocidas
        - Las predicciones en condiciones climáticas extremas no observadas en los datos (e.g., olas de calor >45°C) deben interpretarse como extrapolaciones teóricas.
        - Las tarifas eléctricas y los factores de emisión son promedios configurables; en redes con tarificación horaria dinámica (*Time-of-Use*), se recomienda ajustar las tarifas en la barra lateral.
        """
    )
