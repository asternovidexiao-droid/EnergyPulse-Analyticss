"""
Módulo de modelado predictivo y Machine Learning para EnergyPulse Analytics.
Implementa pipelines libres de data leakage, comparación contra modelo base,
cálculo de métricas estandarizadas (MAE, RMSE, R², MAPE), importancia de características
y simulador de escenarios "What-If".
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_CO2_KG_PER_KWH, DEFAULT_COST_PER_KWH_USD, TARGET_COLUMN
from src.preprocessing import build_ml_preprocessor


@dataclass
class ModelMetrics:
    """Métricas de evaluación del modelo."""
    mae: float
    rmse: float
    r2: float
    mape: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "MAE (kWh)": round(self.mae, 2),
            "RMSE (kWh)": round(self.rmse, 2),
            "R² Score": round(self.r2, 4),
            "MAPE (%)": round(self.mape * 100.0, 2),
        }


@dataclass
class TrainingResult:
    """Contenedor integral de los resultados del entrenamiento y evaluación."""
    model_name: str
    pipeline: Pipeline
    features: List[str]
    train_metrics: ModelMetrics
    test_metrics: ModelMetrics
    baseline_metrics: ModelMetrics
    y_test_real: np.ndarray
    y_test_pred: np.ndarray
    feature_importances: pd.DataFrame
    residuals: np.ndarray
    sample_size_train: int
    sample_size_test: int


# Variables por defecto utilizadas para el modelado
DEFAULT_NUMERIC_FEATURES = [
    "surface_area_sqm",
    "temperature_celsius",
    "humidity_percent",
    "occupancy_rate",
    "hour",
    "is_weekend",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "building_type",
    "hvac_mode",
]


def train_energy_model(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.20,
    random_state: int = 42,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = 12,
) -> TrainingResult:
    """
    Entrena un pipeline de Machine Learning completo evitando fuga de datos.
    Compara el modelo seleccionado con un baseline determinista (DummyRegressor o Linear).

    Args:
        df: DataFrame limpio con datos de telemetría y características.
        model_type: Tipo de algoritmo ('Random Forest', 'Gradient Boosting', 'Ridge', 'Linear').
        test_size: Proporción de datos reservada para el conjunto de prueba (e.g. 0.20).
        random_state: Semilla aleatoria para reproducibilidad.
        numeric_features: Lista de columnas numéricas a usar como predictores.
        categorical_features: Lista de columnas categóricas a codificar.
        n_estimators: Número de árboles para modelos de ensamble.
        max_depth: Profundidad máxima de los árboles.

    Returns:
        TrainingResult con modelos entrenados, métricas completas y arrays de evaluación.
    """
    if df.empty or TARGET_COLUMN not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna objetivo '{TARGET_COLUMN}'.")

    # Si faltan componentes temporales como 'hour' o 'is_weekend', generarlos al vuelo
    df_ready = df.copy()
    if "hour" not in df_ready.columns and "timestamp" in df_ready.columns:
        timestamps = pd.to_datetime(df_ready["timestamp"], errors="coerce")
        df_ready["hour"] = timestamps.dt.hour
        df_ready["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)

    num_cols = [c for c in (numeric_features or DEFAULT_NUMERIC_FEATURES) if c in df_ready.columns]
    cat_cols = [c for c in (categorical_features or DEFAULT_CATEGORICAL_FEATURES) if c in df_ready.columns]

    all_features = num_cols + cat_cols
    if not all_features:
        raise ValueError("No se encontraron variables predictoras válidas en el conjunto de datos.")

    # Filtrar nulos en los predictores y en el target
    valid_data = df_ready[all_features + [TARGET_COLUMN]].dropna()
    if len(valid_data) < 20:
        raise ValueError("Cantidad insuficiente de filas válidas tras descartar valores nulos (< 20).")

    X = valid_data[all_features]
    y = valid_data[TARGET_COLUMN].values

    # División rigurosa entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 1. Pipeline de Preprocesamiento
    preprocessor = build_ml_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)

    # 2. Selección del Estimador
    if model_type == "Random Forest":
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "Gradient Boosting":
        regressor = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    elif model_type == "Ridge":
        regressor = Ridge(alpha=1.0, random_state=random_state)
    else:  # Linear Regression por defecto
        regressor = LinearRegression()

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    pipeline.fit(X_train, y_train)

    # 3. Modelo Base (Baseline) para comparación honesta
    baseline_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", DummyRegressor(strategy="mean"))]
    )
    baseline_pipeline.fit(X_train, y_train)
    y_test_base_pred = baseline_pipeline.predict(X_test)

    # 4. Predicciones y Métricas
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Evitar consumos negativos físicamente imposibles
    y_test_pred = np.maximum(0.0, y_test_pred)
    y_train_pred = np.maximum(0.0, y_train_pred)

    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        # Para evitar división por cero en MAPE
        non_zero_mask = y_true > 1e-4
        if np.sum(non_zero_mask) > 0:
            mape = float(mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask]))
        else:
            mape = 0.0
        return ModelMetrics(mae=mae, rmse=rmse, r2=r2, mape=mape)

    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    baseline_metrics = calculate_metrics(y_test, y_test_base_pred)

    # 5. Importancia de Variables (Feature Importances)
    importances_df = _extract_feature_importances(pipeline, num_cols, cat_cols)

    # 6. Residuales
    residuals = y_test - y_test_pred

    return TrainingResult(
        model_name=model_type,
        pipeline=pipeline,
        features=all_features,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        y_test_real=y_test,
        y_test_pred=y_test_pred,
        feature_importances=importances_df,
        residuals=residuals,
        sample_size_train=len(X_train),
        sample_size_test=len(X_test),
    )


def _extract_feature_importances(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Extrae y mapea nombres transformados a la importancia de características."""
    try:
        regressor = pipeline.named_steps["regressor"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Obtener nombres de columnas tras One-Hot Encoding
        cat_encoder = preprocessor.named_transformers_["cat"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            ohe_names = list(cat_encoder.get_feature_names_out(categorical_features))
        else:
            ohe_names = [f"cat_{i}" for i in range(len(categorical_features))]

        feature_names = numeric_features + ohe_names

        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
        elif hasattr(regressor, "coef_"):
            importances = np.abs(regressor.coef_)
        else:
            # Si no soporta importancias directas
            return pd.DataFrame()

        if len(importances) == len(feature_names):
            df_imp = pd.DataFrame({
                "Característica": feature_names,
                "Importancia": importances,
            }).sort_values(by="Importancia", ascending=False).reset_index(drop=True)
            df_imp["Importancia Relativa (%)"] = (
                (df_imp["Importancia"] / df_imp["Importancia"].sum()) * 100.0
            ).round(2)
            return df_imp
    except Exception:
        pass

    return pd.DataFrame()


def predict_scenario(
    pipeline: Pipeline,
    scenario_input: Dict[str, Any],
    cost_per_kwh: float = DEFAULT_COST_PER_KWH_USD,
    co2_factor: float = DEFAULT_CO2_KG_PER_KWH,
) -> Dict[str, float]:
    """
    Predice el consumo energético para un escenario hipotético 'What-If'.

    Args:
        pipeline: Pipeline entrenado.
        scenario_input: Diccionario con los valores del escenario.
        cost_per_kwh: Tarifa eléctrica unitaria ($/kWh).
        co2_factor: Factor de emisión (kg CO2/kWh).

    Returns:
        Diccionario con predicción de kWh, coste estimado y huella de carbono.
    """
    input_df = pd.DataFrame([scenario_input])
    predicted_kwh = float(pipeline.predict(input_df)[0])
    predicted_kwh = max(0.0, round(predicted_kwh, 2))

    cost_usd = round(predicted_kwh * cost_per_kwh, 2)
    carbon_kg = round(predicted_kwh * co2_factor, 2)

    return {
        "predicted_consumption_kwh": predicted_kwh,
        "estimated_cost_usd": cost_usd,
        "carbon_emissions_kg": carbon_kg,
    }
